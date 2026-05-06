"""
Speech Separation Module
------------------------
Uses SpeechBrain SepFormer trained on LibriMix (clean 2-speaker speech).

Model progression:
  - wham     = noisy reverberant speech            → poor on clean TTS
  - wsj02mix = clean WSJ0 speech, ~100 speakers    → decent on TTS
  - libri2mix= clean LibriSpeech, 2000+ speakers   → best generalisation (current)

Two modes:
  REAL (demo_oracle=False): SepFormer actually separates voices
  ORACLE (demo_oracle=True): Uses pre-generated clean sources directly (demo cheat)

GPU support (priority order):
  1. Intel Arc / Intel GPU via IPEX (XPU)  — install: see README
  2. NVIDIA GPU via CUDA
  3. CPU fallback (always available)
"""

import os
import shutil
import numpy as np
import soundfile as sf
from . import utils as _utils   # activates Windows symlink patch

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "separated")
MODELS_DIR  = os.path.join(BASE_DIR, "models", "sepformer-libri2mix")
OUTPUT_SR   = 16000   # we always write 16 kHz output

# SepFormer LibriMix: clean speech, 8 kHz model, trained on 2000+ LibriSpeech speakers
# Much better generalisation than wsj02mix (only ~100 WSJ0 speakers)
SEP_REPO_ID = "speechbrain/sepformer-libri2mix"
SEP_MODEL_SR = 8000


# ── Device selection ───────────────────────────────────────────────────────────

def _get_torch_device() -> str:
    """
    Detect best available compute device.
    Priority: Intel Arc (XPU via IPEX) > NVIDIA (CUDA) > CPU

    IPEX import is tested in a subprocess first because a DLL version mismatch
    causes Python itself to crash (exit 127) rather than raising ImportError.
    To enable Arc GPU:
      1. Install Intel oneAPI Base Toolkit (provides runtime DLLs)
      2. pip install intel-extension-for-pytorch
             --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
    """
    import torch, subprocess, sys

    # 1. Intel Arc GPU via IPEX — test import safely in a subprocess
    try:
        probe = subprocess.run(
            [sys.executable, "-c",
             "import intel_extension_for_pytorch as i; import torch; "
             "print('xpu' if torch.xpu.is_available() else 'no')"],
            capture_output=True, text=True, timeout=15,
        )
        if probe.returncode == 0 and "xpu" in probe.stdout:
            import intel_extension_for_pytorch as ipex  # noqa: F401
            if torch.xpu.is_available():
                name = torch.xpu.get_device_name(0)
                print(f"[Device] Intel Arc GPU via IPEX: {name}")
                return "xpu"
    except Exception:
        pass

    # 2. NVIDIA GPU via CUDA
    if torch.cuda.is_available():
        print(f"[Device] NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"

    # 3. CPU fallback
    print("[Device] Using CPU (no GPU runtime detected)")
    return "cpu"


# ── Oracle (demo cheat) ────────────────────────────────────────────────────────

def _oracle_separation(clean_paths: list) -> list:
    """Copy pre-generated clean TTS sources as 'separated' outputs."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_paths = []
    for i, src in enumerate(clean_paths):
        dst = os.path.join(OUTPUTS_DIR, f"speaker_{i+1}.wav")
        shutil.copy(src, dst)
        out_paths.append(dst)
    print(f"[Separation] Oracle mode — using {len(out_paths)} clean TTS sources (no real separation).")
    return out_paths


# ── Real SepFormer model ───────────────────────────────────────────────────────

_sep_model = None   # cached after first load


def _get_sep_model(device: str):
    """Load (and cache) the SepFormer WSJ0-2mix model."""
    global _sep_model
    if _sep_model is not None:
        return _sep_model

    from speechbrain.inference.separation import SepformerSeparation
    from huggingface_hub import snapshot_download

    os.makedirs(MODELS_DIR, exist_ok=True)
    hub_cache = os.path.join(BASE_DIR, "models", "hub")

    print(f"[Separation] Downloading SepFormer-WSJ02mix (clean speech, ~300 MB)…")
    local_dir = snapshot_download(
        repo_id=SEP_REPO_ID,
        local_dir=MODELS_DIR,
        cache_dir=hub_cache,
    )

    print(f"[Separation] Loading SepFormer on device='{device}'…")
    _sep_model = SepformerSeparation.from_hparams(
        source=local_dir,
        savedir=MODELS_DIR,
        run_opts={"device": device},
    )
    print("[Separation] SepFormer-WSJ02mix loaded.")
    return _sep_model


def _model_separation(mixed_path: str, num_speakers: int) -> list:
    """Run SepFormer WSJ0-2mix on the mixed audio and return separated stream paths."""
    import torch
    import torchaudio

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    device = _get_torch_device()
    model  = _get_sep_model(device)

    # Read mixed audio
    audio_np, file_sr = sf.read(mixed_path, dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)

    # Resample to model's expected sample rate (8 kHz)
    waveform = torch.from_numpy(audio_np).unsqueeze(0)   # (1, time)
    if file_sr != SEP_MODEL_SR:
        waveform = torchaudio.functional.resample(waveform, file_sr, SEP_MODEL_SR)

    # Move to target device
    waveform = waveform.to(device)

    print(f"[Separation] Running SepFormer-WSJ02mix ({device}) on {os.path.basename(mixed_path)}…")
    with torch.no_grad():
        est_sources = model.separate_batch(waveform)
    # est_sources: (batch=1, time, num_sources) → (time, num_sources)
    est_sources = est_sources.squeeze(0).cpu()

    n_out = min(num_speakers, est_sources.shape[1])
    out_paths = []
    for i in range(n_out):
        src_np = est_sources[:, i].numpy()   # (time,)

        # Upsample back to 16 kHz for ASR / speaker ID
        if SEP_MODEL_SR != OUTPUT_SR:
            src_t  = torch.from_numpy(src_np).unsqueeze(0)
            src_t  = torchaudio.functional.resample(src_t, SEP_MODEL_SR, OUTPUT_SR)
            src_np = src_t.squeeze(0).numpy()

        # Normalise loudness to avoid clipping
        peak = np.abs(src_np).max()
        if peak > 0:
            src_np = src_np / peak * 0.9

        out_path = os.path.join(OUTPUTS_DIR, f"speaker_{i+1}.wav")
        sf.write(out_path, src_np, OUTPUT_SR)
        out_paths.append(out_path)

    print(f"[Separation] Done — {len(out_paths)} streams separated on {device}.")
    return out_paths


# ── Naive fallback (last resort) ───────────────────────────────────────────────

def _naive_fallback(mixed_path: str, num_speakers: int) -> list:
    """Duplicate mixed audio for each slot — NOT real separation."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    audio, sr = sf.read(mixed_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    out_paths = []
    for i in range(num_speakers):
        p = os.path.join(OUTPUTS_DIR, f"speaker_{i+1}.wav")
        sf.write(p, audio, sr)
        out_paths.append(p)
    print(f"[Separation] WARNING: Naive fallback — duplicated mixed audio for {num_speakers} slots.")
    return out_paths


# ── Public entry point ─────────────────────────────────────────────────────────

def separate(
    mixed_path: str,
    clean_paths: list = None,
    num_speakers: int = 2,
    demo_oracle: bool = False,
) -> dict:
    """
    Returns:
        {
          "separated_paths": [str, ...],
          "mode": "oracle" | "sepformer_wsj02mix" | "naive_fallback",
          "warning": str | None
        }
    """
    # Oracle mode
    if demo_oracle:
        if clean_paths:
            paths = _oracle_separation(clean_paths)
            return {"separated_paths": paths, "mode": "oracle", "warning": None}
        else:
            paths = _naive_fallback(mixed_path, num_speakers)
            return {
                "separated_paths": paths,
                "mode": "naive_fallback",
                "warning": "Oracle requested but no clean sources available; mixed audio duplicated.",
            }

    # Real model mode
    try:
        paths = _model_separation(mixed_path, num_speakers)
        return {"separated_paths": paths, "mode": "sepformer_wsj02mix", "warning": None}
    except RuntimeError as e:
        print(f"[Separation] Model failed: {e}")
        if clean_paths:
            print("[Separation] Falling back to oracle clean sources.")
            paths = _oracle_separation(clean_paths)
            return {
                "separated_paths": paths,
                "mode": "oracle",
                "warning": f"SepFormer unavailable ({e}); used oracle clean sources.",
            }
        paths = _naive_fallback(mixed_path, num_speakers)
        return {
            "separated_paths": paths,
            "mode": "naive_fallback",
            "warning": f"SepFormer unavailable ({e}); mixed audio duplicated.",
        }
