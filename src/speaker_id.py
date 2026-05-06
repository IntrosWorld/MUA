import os
import pickle
import numpy as np
from . import utils as _utils  # activates Windows symlink patch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SPEAKER_DB_PATH = os.path.join(OUTPUTS_DIR, "speaker_db.pkl")

THRESHOLD = 0.05  # cosine similarity — below this a stream is marked Unknown
# Clean TTS scores ~0.7-0.8; SepFormer-separated TTS scores ~0.09-0.24
# (lower because SepFormer was trained on real speech, not TTS)

_encoder = None
_speaker_db = None


def _get_spk_device() -> str:
    """Priority: Intel Arc XPU > NVIDIA CUDA > CPU. Probes IPEX via subprocess to avoid DLL crashes."""
    import torch, subprocess, sys
    try:
        probe = subprocess.run(
            [sys.executable, "-c",
             "import intel_extension_for_pytorch; import torch; "
             "print('xpu' if torch.xpu.is_available() else 'no')"],
            capture_output=True, text=True, timeout=10,
        )
        if probe.returncode == 0 and "xpu" in probe.stdout:
            import intel_extension_for_pytorch  # noqa: F401
            if torch.xpu.is_available():
                return "xpu"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_encoder():
    global _encoder
    if _encoder is not None:
        return _encoder
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        from huggingface_hub import snapshot_download

        ecapa_dir = os.path.join(MODELS_DIR, "spkrec-ecapa")
        hub_cache = os.path.join(BASE_DIR, "models", "hub")

        print("[SpeakerID] Downloading ECAPA-TDNN model (first run ~100 MB)...")
        local_dir = snapshot_download(
            repo_id="speechbrain/spkrec-ecapa-voxceleb",
            local_dir=ecapa_dir,
            cache_dir=hub_cache,
        )
        device = _get_spk_device()
        print(f"[SpeakerID] Loading ECAPA-TDNN on device='{device}'...")
        _encoder = EncoderClassifier.from_hparams(
            source=local_dir,
            savedir=ecapa_dir,
            run_opts={"device": device},
        )
        print(f"[SpeakerID] ECAPA-TDNN loaded on {device}.")
        return _encoder
    except Exception as e:
        print(f"[SpeakerID] Model load failed: {e}. Using filename-based fallback.")
        return None


def _compute_embedding_model(audio_path: str) -> np.ndarray:
    enc = _get_encoder()
    if enc is None:
        return None
    import torch
    import soundfile as sf
    # Use soundfile to avoid torchaudio TorchCodec issues on Windows
    audio_np, sr = sf.read(audio_path, dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    # ECAPA expects 16 kHz
    if sr != 16000:
        import torchaudio
        wav = torch.from_numpy(audio_np).unsqueeze(0)
        wav = torchaudio.functional.resample(wav, sr, 16000)
        audio_np = wav.squeeze(0).numpy()
    signal = torch.from_numpy(audio_np).unsqueeze(0)  # (1, time)
    with torch.no_grad():
        emb = enc.encode_batch(signal)
    return emb.squeeze().cpu().numpy()


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def enroll_speakers():
    """Compute and save average embeddings for all enrolled speakers."""
    global _speaker_db
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    enrolled_dir = os.path.join(DATA_DIR, "enrolled_speakers")
    db = {}

    enc = _get_encoder()
    if enc is None:
        print("[SpeakerID] No encoder available — creating filename-based fallback DB.")
        speakers = [d for d in os.listdir(enrolled_dir) if os.path.isdir(os.path.join(enrolled_dir, d))]
        for spk in speakers:
            db[spk] = {"embedding": None, "fallback": True}
        _speaker_db = db
        with open(SPEAKER_DB_PATH, "wb") as f:
            pickle.dump(db, f)
        return db

    for spk in os.listdir(enrolled_dir):
        spk_dir = os.path.join(enrolled_dir, spk)
        if not os.path.isdir(spk_dir):
            continue
        wavs = [os.path.join(spk_dir, f) for f in os.listdir(spk_dir) if f.endswith(".wav")]
        if not wavs:
            continue
        embeddings = []
        for w in wavs:
            emb = _compute_embedding_model(w)
            if emb is not None:
                embeddings.append(emb)
        if embeddings:
            avg_emb = np.mean(embeddings, axis=0)
            db[spk] = {"embedding": avg_emb, "fallback": False}
            print(f"[SpeakerID] Enrolled {spk} with {len(embeddings)} samples.")

    _speaker_db = db
    with open(SPEAKER_DB_PATH, "wb") as f:
        pickle.dump(db, f)
    print(f"[SpeakerID] Speaker DB saved to {SPEAKER_DB_PATH}")
    return db


def _load_db():
    global _speaker_db
    if _speaker_db is not None:
        return _speaker_db
    if os.path.exists(SPEAKER_DB_PATH):
        with open(SPEAKER_DB_PATH, "rb") as f:
            _speaker_db = pickle.load(f)
    else:
        _speaker_db = enroll_speakers()
    return _speaker_db


def identify_speaker(audio_path: str, fallback_name: str = None) -> dict:
    """
    Returns {"speaker": name, "confidence": float, "fallback": bool}
    fallback_name: if provided and model unavailable, return this name directly.
    """
    db = _load_db()

    # Check if all entries are fallback (no embeddings)
    all_fallback = all(v.get("fallback", True) for v in db.values())

    if all_fallback or not db:
        if fallback_name:
            return {"speaker": fallback_name.capitalize(), "confidence": 1.0, "fallback": True}
        return {"speaker": "Unknown", "confidence": 0.0, "fallback": True}

    emb = _compute_embedding_model(audio_path)
    if emb is None:
        if fallback_name:
            return {"speaker": fallback_name.capitalize(), "confidence": 1.0, "fallback": True}
        return {"speaker": "Unknown", "confidence": 0.0, "fallback": True}

    best_spk = "Unknown"
    best_sim = -1.0
    for spk, entry in db.items():
        ref = entry.get("embedding")
        if ref is None:
            continue
        sim = _cosine_sim(emb, ref)
        if sim > best_sim:
            best_sim = sim
            best_spk = spk

    if best_sim < THRESHOLD:
        best_spk = "Unknown"

    return {
        "speaker": best_spk.capitalize() if best_spk != "Unknown" else "Unknown",
        "confidence": round(best_sim, 4),
        "fallback": False,
    }


def enroll_real_speaker(name: str, audio_paths: list) -> dict:
    """
    Enroll a real human speaker from one or more audio files.
    Adds/updates their entry in speaker_db.pkl.
    Returns the updated db entry.
    """
    global _speaker_db
    name_key = name.strip().lower()
    if not name_key:
        raise ValueError("Speaker name cannot be empty.")
    if not audio_paths:
        raise ValueError("At least one audio file is required.")

    # Save audio files to enrolled_speakers folder
    spk_dir = os.path.join(DATA_DIR, "enrolled_speakers", name_key)
    os.makedirs(spk_dir, exist_ok=True)

    embeddings = []
    saved_paths = []
    for i, src_path in enumerate(audio_paths):
        dst = os.path.join(spk_dir, f"real_enroll_{i+1}.wav")
        import shutil, soundfile as sf_mod
        # Ensure 16kHz mono WAV
        audio_np, sr = sf_mod.read(src_path, dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        if sr != 16000:
            import torch, torchaudio
            wav = torch.from_numpy(audio_np).unsqueeze(0)
            wav = torchaudio.functional.resample(wav, sr, 16000)
            audio_np = wav.squeeze(0).numpy()
        sf_mod.write(dst, audio_np, 16000)
        saved_paths.append(dst)

        emb = _compute_embedding_model(dst)
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        raise RuntimeError("Could not compute embeddings — model may be unavailable.")

    avg_emb = np.mean(embeddings, axis=0)
    db = _load_db()
    db[name_key] = {"embedding": avg_emb, "fallback": False, "real_human": True}
    _speaker_db = db

    with open(SPEAKER_DB_PATH, "wb") as f:
        pickle.dump(db, f)

    print(f"[SpeakerID] Enrolled real speaker '{name_key}' with {len(embeddings)} sample(s).")
    return {"name": name_key, "samples": len(embeddings), "embedding_dim": avg_emb.shape[0]}


def list_enrolled_speakers() -> list:
    """Return list of enrolled speaker names with their type."""
    db = _load_db()
    result = []
    for name, entry in db.items():
        result.append({
            "name": name,
            "type": "real" if entry.get("real_human") else "tts",
            "fallback": entry.get("fallback", False),
            "has_embedding": entry.get("embedding") is not None,
        })
    return result


def delete_speaker(name: str) -> bool:
    """Remove a speaker from the DB."""
    global _speaker_db
    name_key = name.strip().lower()
    db = _load_db()
    if name_key not in db:
        return False
    del db[name_key]
    _speaker_db = db
    with open(SPEAKER_DB_PATH, "wb") as f:
        pickle.dump(db, f)
    print(f"[SpeakerID] Deleted speaker '{name_key}' from DB.")
    return True


def assign_speakers_optimal(stream_paths: list, gt_speakers: list = None) -> list:
    """
    Assign one unique speaker per stream using the Hungarian algorithm.

    Without this, every stream independently picks the highest-sim speaker
    and you get duplicates (e.g. both streams → "Rahul").

    Returns a list of dicts (same order as stream_paths):
        [{"speaker": name, "confidence": float, "fallback": bool}, ...]
    """
    db = _load_db()
    n_streams = len(stream_paths)

    # Compute embeddings for every stream
    embeddings = []
    for sp in stream_paths:
        emb = _compute_embedding_model(sp)
        embeddings.append(emb)

    # Check if model is available
    all_none = all(e is None for e in embeddings)
    if all_none:
        # Model unavailable — fall back to GT speakers or Unknown
        results = []
        for i in range(n_streams):
            gt = gt_speakers[i] if gt_speakers and i < len(gt_speakers) else None
            if gt:
                results.append({"speaker": gt.capitalize(), "confidence": 1.0, "fallback": True})
            else:
                results.append({"speaker": "Unknown", "confidence": 0.0, "fallback": True})
        return results

    # Build list of enrolled speakers with valid embeddings
    spk_names = [k for k, v in db.items() if v.get("embedding") is not None]
    if not spk_names:
        results = []
        for i in range(n_streams):
            gt = gt_speakers[i] if gt_speakers and i < len(gt_speakers) else None
            results.append({"speaker": gt.capitalize() if gt else "Unknown",
                            "confidence": 0.0, "fallback": True})
        return results

    # Build similarity matrix: shape (n_streams, n_enrolled_speakers)
    sim_matrix = np.zeros((n_streams, len(spk_names)))
    for i, emb in enumerate(embeddings):
        if emb is None:
            continue
        for j, spk in enumerate(spk_names):
            ref = db[spk]["embedding"]
            sim_matrix[i, j] = _cosine_sim(emb, ref)

    # Hungarian algorithm: maximise total similarity
    # scipy wants a cost matrix (minimisation), so negate similarities
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    except ImportError:
        # Fallback: greedy assignment (no scipy)
        row_ind = list(range(n_streams))
        col_ind = []
        used = set()
        for i in range(n_streams):
            best_j, best_s = -1, -9999
            for j in range(len(spk_names)):
                if j not in used and sim_matrix[i, j] > best_s:
                    best_s = sim_matrix[i, j]
                    best_j = j
            if best_j >= 0:
                col_ind.append(best_j)
                used.add(best_j)
            else:
                col_ind.append(0)
        row_ind = list(range(n_streams))

    # Build assignment map
    assignment = {}  # stream_index → (speaker_name, similarity)
    for r, c in zip(row_ind, col_ind):
        assignment[r] = (spk_names[c], float(sim_matrix[r, c]))

    results = []
    for i in range(n_streams):
        if i in assignment:
            spk, sim = assignment[i]
            if sim < THRESHOLD:
                spk = "Unknown"
            results.append({
                "speaker": spk.capitalize() if spk != "Unknown" else "Unknown",
                "confidence": round(sim, 4),
                "fallback": False,
            })
        else:
            # More streams than enrolled speakers
            gt = gt_speakers[i] if gt_speakers and i < len(gt_speakers) else None
            results.append({
                "speaker": gt.capitalize() if gt else "Unknown",
                "confidence": 0.0,
                "fallback": True,
            })

    print(f"[SpeakerID] Optimal assignment: " +
          ", ".join(f"stream{i+1}={r['speaker']}({r['confidence']:.3f})"
                    for i, r in enumerate(results)))
    return results


if __name__ == "__main__":
    db = enroll_speakers()
    print("Enrolled speakers:", list(db.keys()))
