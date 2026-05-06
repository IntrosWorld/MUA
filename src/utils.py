import os
import json
import shutil
import soundfile as sf
import numpy as np

# ── Windows symlink fix (applied once at import time) ──
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

SR = 16000


def get_audio_duration(path: str) -> float:
    try:
        info = sf.info(path)
        return info.duration
    except Exception:
        return 0.0


def load_ground_truth(gt_path: str) -> dict:
    try:
        with open(gt_path, "r") as f:
            return json.load(f)
    except Exception:
        return {"sources": []}


def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def ensure_wav_mono_16k(path: str, out_path: str = None) -> str:
    """Ensure audio is mono 16 kHz WAV. Returns output path."""
    if out_path is None:
        out_path = path
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=SR)
    sf.write(out_path, data, SR)
    return out_path


def normalize_text(text: str) -> str:
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
