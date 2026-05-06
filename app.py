import os
import sys
import uuid
import asyncio
import shutil
import tempfile

# ── Project base ──
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Redirect ALL model caches to local models dir (never C drive) ──
os.environ["HF_HOME"]               = MODELS_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(MODELS_DIR, "hub")
os.environ["TRANSFORMERS_CACHE"]    = os.path.join(MODELS_DIR, "transformers")
os.environ["TORCH_HOME"]            = os.path.join(MODELS_DIR, "torch")
os.environ["XDG_CACHE_HOME"]        = MODELS_DIR
os.environ["SPEECHBRAIN_CACHE"]     = os.path.join(MODELS_DIR, "speechbrain")

# ── Windows symlink fix ──
# SpeechBrain always tries os.symlink when linking model files.
# Windows blocks symlinks without admin / Developer Mode, so we patch
# os.symlink to fall back to a plain file copy when it fails.
_orig_symlink = os.symlink
def _safe_symlink(src, dst, target_is_directory=False, *, dir_fd=None):
    try:
        _orig_symlink(src, dst, target_is_directory=target_is_directory, dir_fd=dir_fd)
    except OSError:
        if os.path.isdir(src):
            if not os.path.exists(dst):
                shutil.copytree(src, dst)
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
os.symlink = _safe_symlink

sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.tts_generate import generate_all, get_command_path, SPEAKERS, COMMANDS
from src.audio_mixer import mix_audio
from src.speaker_id import (
    enroll_speakers, enroll_real_speaker,
    list_enrolled_speakers, delete_speaker, identify_speaker,
)
from src.user_profiles import (
    get_profile, save_profile, list_profiles,
    update_workspace_section, update_smart_home,
    add_calendar_event, add_note, add_reminder,
)
from src.llm_engine import set_llm_mode, get_llm_mode, set_gemini_key, get_status as llm_status
from src.pipeline import run_pipeline

app = FastAPI(title="Multi-User Smart Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MIXED_DIR = os.path.join(DATA_DIR, "mixed")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)


@app.get("/")
def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)


@app.get("/style.css")
def serve_css():
    return FileResponse(os.path.join(FRONTEND_DIR, "style.css"))


@app.get("/script.js")
def serve_js():
    return FileResponse(os.path.join(FRONTEND_DIR, "script.js"))


@app.post("/generate_demo")
async def generate_demo(
    num_speakers: int = Form(2),
    overlap_type: str = Form("partial"),
    noise_level: str = Form("low"),
    noise_type: str = Form("white"),
    cmd_indices: str = Form("0,0,0"),  # comma-separated cmd index per speaker
):
    try:
        # Generate TTS voices
        await generate_all()

        # Enroll speakers
        enroll_speakers()

        speakers_list = list(SPEAKERS.keys())[:num_speakers]
        idx_list = [int(x) for x in cmd_indices.split(",")]

        speaker_files = []
        for i, spk in enumerate(speakers_list):
            cmd_idx = idx_list[i] if i < len(idx_list) else 0
            cmd_idx = min(cmd_idx, len(COMMANDS[spk]) - 1)
            path = get_command_path(spk, cmd_idx)
            if not os.path.exists(path):
                raise HTTPException(status_code=500, detail=f"TTS file missing for {spk}. Run generate first.")
            text = COMMANDS[spk][cmd_idx]
            speaker_files.append((spk, path, text))

        result = mix_audio(
            speaker_files=speaker_files,
            overlap_type=overlap_type,
            noise_level=noise_level,
            noise_type=noise_type,
            output_prefix="demo_mixed",
        )

        return JSONResponse({
            "status": "ok",
            "mixed_path": result["mixed_path"],
            "ground_truth_path": result["ground_truth_path"],
            "ground_truth": result["ground_truth"],
            "clean_paths": result["clean_paths"],
            "audio_url": "/audio/mixed/demo_mixed.wav",
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_demo")
async def process_demo(
    num_speakers: int = Form(2),
    overlap_type: str = Form("partial"),
    noise_level: str = Form("low"),
    noise_type: str = Form("white"),
    demo_oracle: bool = Form(True),
    cmd_indices: str = Form("0,0,0"),
):
    try:
        # Generate if not exists
        await generate_all()
        enroll_speakers()

        speakers_list = list(SPEAKERS.keys())[:num_speakers]
        idx_list = [int(x) for x in cmd_indices.split(",")]

        speaker_files = []
        for i, spk in enumerate(speakers_list):
            cmd_idx = idx_list[i] if i < len(idx_list) else 0
            cmd_idx = min(cmd_idx, len(COMMANDS[spk]) - 1)
            path = get_command_path(spk, cmd_idx)
            text = COMMANDS[spk][cmd_idx]
            speaker_files.append((spk, path, text))

        mix_result = mix_audio(
            speaker_files=speaker_files,
            overlap_type=overlap_type,
            noise_level=noise_level,
            noise_type=noise_type,
            output_prefix="demo_mixed",
        )

        report = run_pipeline(
            input_audio_path=mix_result["mixed_path"],
            ground_truth_path=mix_result["ground_truth_path"],
            clean_paths=mix_result["clean_paths"],
            demo_oracle=demo_oracle,
            num_speakers=num_speakers,
        )

        # Add audio URLs for frontend
        for r in report["results"]:
            fname = os.path.basename(r["stream_path"])
            r["audio_url"] = f"/audio/separated/{fname}"

        return JSONResponse(report)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_audio")
async def process_audio(
    file: UploadFile = File(...),
    num_speakers: int = Form(2),
    demo_oracle: bool = Form(True),
):
    try:
        ext = os.path.splitext(file.filename)[1] or ".wav"
        tmp_path = os.path.join(UPLOADS_DIR, f"{uuid.uuid4()}{ext}")
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Ensure speakers enrolled
        enroll_speakers()

        report = run_pipeline(
            input_audio_path=tmp_path,
            demo_oracle=demo_oracle,
            num_speakers=num_speakers,
        )

        for r in report["results"]:
            fname = os.path.basename(r["stream_path"])
            r["audio_url"] = f"/audio/separated/{fname}"

        return JSONResponse(report)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audio/mixed/{filename}")
def serve_mixed_audio(filename: str):
    path = os.path.join(MIXED_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="audio/wav")


@app.get("/audio/separated/{filename}")
def serve_separated_audio(filename: str):
    path = os.path.join(OUTPUTS_DIR, "separated", filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="audio/wav")


@app.get("/audio/{folder}/{filename}")
def serve_audio_generic(folder: str, filename: str):
    # Security: only allow data and outputs folders
    allowed = {
        "enrolled": os.path.join(DATA_DIR, "enrolled_speakers"),
        "generated": os.path.join(DATA_DIR, "generated"),
        "mixed": MIXED_DIR,
        "separated": os.path.join(OUTPUTS_DIR, "separated"),
    }
    base = allowed.get(folder)
    if not base:
        raise HTTPException(status_code=403, detail="Folder not allowed")
    path = os.path.join(base, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="audio/wav")


@app.get("/commands")
def get_commands():
    return JSONResponse(COMMANDS)


# ── Real Speaker Enrollment ────────────────────────────────────────────────

@app.get("/speakers")
def get_speakers():
    """List all enrolled speakers."""
    try:
        speakers = list_enrolled_speakers()
        return JSONResponse({"speakers": speakers})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enroll_speaker")
async def enroll_speaker_route(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """
    Enroll a real human speaker.
    POST with form fields: name (str) + files (1–5 audio files).
    """
    try:
        if not name.strip():
            raise HTTPException(status_code=400, detail="Name is required.")
        if len(files) < 1:
            raise HTTPException(status_code=400, detail="At least 1 audio file is required.")

        # Save uploaded files to temp dir
        tmp_paths = []
        for f in files:
            ext = os.path.splitext(f.filename)[1] or ".wav"
            p = os.path.join(UPLOADS_DIR, f"{uuid.uuid4()}{ext}")
            with open(p, "wb") as out:
                shutil.copyfileobj(f.file, out)
            tmp_paths.append(p)

        result = enroll_real_speaker(name.strip(), tmp_paths)
        return JSONResponse({"status": "ok", "enrolled": result})
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/speakers/{name}")
def delete_speaker_route(name: str):
    """Delete an enrolled speaker."""
    ok = delete_speaker(name)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Speaker '{name}' not found.")
    return JSONResponse({"status": "deleted", "name": name})


@app.post("/identify_voice")
async def identify_voice_route(file: UploadFile = File(...)):
    """
    Upload a single audio clip — returns who the system thinks is speaking.
    Works with any enrolled speaker (TTS or real human).
    """
    try:
        ext = os.path.splitext(file.filename)[1] or ".wav"
        tmp_path = os.path.join(UPLOADS_DIR, f"{uuid.uuid4()}{ext}")
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = identify_speaker(tmp_path)

        # Also return all similarities for transparency
        import pickle, numpy as np
        db_path = os.path.join(OUTPUTS_DIR, "speaker_db.pkl")
        all_sims = {}
        if os.path.exists(db_path):
            from src.speaker_id import _compute_embedding_model, _cosine_sim
            with open(db_path, "rb") as f:
                db = pickle.load(f)
            emb = _compute_embedding_model(tmp_path)
            if emb is not None:
                for spk, entry in db.items():
                    ref = entry.get("embedding")
                    if ref is not None:
                        all_sims[spk] = round(_cosine_sim(emb, ref), 4)

        return JSONResponse({
            "identified_speaker": result["speaker"],
            "confidence": result["confidence"],
            "fallback": result["fallback"],
            "all_similarities": all_sims,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Profile routes ────────────────────────────────────────────────────────

@app.get("/profiles")
def get_profiles():
    return JSONResponse({"profiles": list_profiles()})

@app.get("/profiles/{name}")
def get_profile_route(name: str):
    p = get_profile(name)
    if not p:
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found.")
    return JSONResponse(p)

@app.put("/profiles/{name}")
async def update_profile_route(name: str, request: Request):
    body = await request.json()
    p = get_profile(name)
    if not p:
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found.")
    p.update(body)
    save_profile(p)
    return JSONResponse({"status": "ok"})

@app.put("/profiles/{name}/workspace/{section}")
async def update_section(name: str, section: str, request: Request):
    try:
        body = await request.json()
        data = body.get("data")
        update_workspace_section(name, section, data)
        return JSONResponse({"status": "ok"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/profiles/{name}/calendar")
async def add_event(name: str, request: Request):
    try:
        body = await request.json()
        event = add_calendar_event(name, body)
        return JSONResponse({"status": "ok", "event": event})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/profiles/{name}/notes")
async def add_note_route(name: str, request: Request):
    try:
        body = await request.json()
        note = add_note(name, body.get("text", ""))
        return JSONResponse({"status": "ok", "note": note})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/profiles/{name}/reminders")
async def add_reminder_route(name: str, request: Request):
    try:
        body = await request.json()
        r = add_reminder(name, body.get("text", ""), body.get("datetime"))
        return JSONResponse({"status": "ok", "reminder": r})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/profiles/{name}/smart_home/{device}")
async def toggle_smart_home(name: str, device: str, request: Request):
    try:
        body = await request.json()
        state = update_smart_home(name, device, body.get("state"))
        return JSONResponse({"status": "ok", "smart_home": state})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/profiles/{name}/create")
async def create_profile(name: str, request: Request):
    from src.user_profiles import DEFAULTS, save_profile as _save
    import copy
    body = await request.json()
    new_profile = {
        "name": name,
        "display_name": body.get("display_name", name),
        "avatar_color": body.get("avatar_color", "#7c3aed"),
        "avatar_emoji": body.get("avatar_emoji", "👤"),
        "type": "real",
        "room": body.get("room", f"{name}'s room"),
        "workspace": copy.deepcopy(list(DEFAULTS.values())[0]["workspace"]),
    }
    new_profile["workspace"]["calendar"] = []
    new_profile["workspace"]["notes"] = []
    new_profile["workspace"]["reminders"] = []
    _save(new_profile)
    return JSONResponse({"status": "ok", "profile": new_profile})


# ── LLM routes ─────────────────────────────────────────────────────────────

@app.get("/llm/status")
def llm_status_route():
    return JSONResponse(llm_status())

@app.post("/llm/mode")
async def set_mode(request: Request):
    body = await request.json()
    mode = body.get("mode")
    if mode not in ("offline", "gemini"):
        raise HTTPException(status_code=400, detail="mode must be 'offline' or 'gemini'")
    set_llm_mode(mode)
    return JSONResponse({"status": "ok", "mode": mode})

@app.post("/llm/gemini_key")
async def set_key(request: Request):
    body = await request.json()
    key = body.get("key", "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="Key cannot be empty.")
    set_gemini_key(key)
    return JSONResponse({"status": "ok"})

@app.post("/llm/chat")
async def llm_chat(request: Request):
    """Direct LLM chat for a user — used by profile page."""
    from src.llm_engine import generate as llm_generate
    from src.user_profiles import get_context_summary
    try:
        body      = await request.json()
        name      = body.get("name", "User")
        message   = body.get("message", "")
        mode      = body.get("mode")
        context   = get_context_summary(name)
        result    = llm_generate(name, context, message, "chat", mode=mode)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Profiles page ──────────────────────────────────────────────────────────

@app.get("/profiles-page")
def profiles_page():
    p = os.path.join(FRONTEND_DIR, "profiles.html")
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="Profiles page not found")
    return FileResponse(p)

@app.get("/profiles.css")
def profiles_css():
    return FileResponse(os.path.join(FRONTEND_DIR, "profiles.css"))

@app.get("/profiles.js")
def profiles_js():
    return FileResponse(os.path.join(FRONTEND_DIR, "profiles.js"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
