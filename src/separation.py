"""
Speech Separation Module
------------------------
Supports multiple SepFormer models — user picks at runtime.

Available models
  sepformer_whamr16k   SepFormer WHAMR! 16 K  (native 16 kHz, reverb-robust)  ← default
  sepformer_libri2mix  SepFormer LibriMix      (8 kHz → upsampled, 2 000+ spk)

Post-processing: Ideal Ratio Mask (IRM) applied to the original mixed signal
  IRM_i(t,f) = |S_i(t,f)|^2 / Σ|S_j(t,f)|^2  → applied to mix STFT
  Fallback   : Wiener spectral subtraction (if mix audio unavailable)

Pipeline
  REAL   (demo_oracle=False): chosen SepFormer → IRM refinement
  ORACLE (demo_oracle=True) : pre-generated clean sources copied directly

GPU support: Intel Arc XPU > NVIDIA CUDA > CPU
"""

import os
import shutil
import numpy as np
import soundfile as sf
from . import utils as _utils   # activates Windows symlink patch

BASE_DIR    = os.environ.get("MUA_DATA_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "separated")
SEP_MODELS  = os.path.join(BASE_DIR, "models")
OUTPUT_SR   = 16000

# ── Model registry ────────────────────────────────────────────────────────────
SEPARATION_MODELS = {
    "sepformer_whamr16k": {
        "repo_id":  "speechbrain/sepformer-whamr16k",
        "local_dir": os.path.join(SEP_MODELS, "sepformer-whamr16k"),
        "native_sr": 16000,
        "label":    "SepFormer WHAMR 16K",
        "desc":     "Native 16 kHz | reverb & noise robust | recommended",
    },
    "sepformer_libri2mix": {
        "repo_id":  "speechbrain/sepformer-libri2mix",
        "local_dir": os.path.join(SEP_MODELS, "sepformer-libri2mix"),
        "native_sr": 8000,
        "label":    "SepFormer LibriMix",
        "desc":     "8 kHz | 2000+ LibriSpeech speakers | clean speech",
    },
}
DEFAULT_MODEL = "sepformer_whamr16k"

# Per-model cache: { model_name -> SepformerSeparation instance }
_sep_model_cache: dict = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def list_models() -> list:
    """Return model info dicts for the API / UI."""
    out = []
    for key, cfg in SEPARATION_MODELS.items():
        downloaded = os.path.isdir(cfg["local_dir"]) and any(
            f.endswith((".pt", ".yaml", ".bin"))
            for f in os.listdir(cfg["local_dir"])
            if os.path.isfile(os.path.join(cfg["local_dir"], f))
        ) if os.path.isdir(cfg["local_dir"]) else False
        out.append({
            "id":          key,
            "label":       cfg["label"],
            "desc":        cfg["desc"],
            "native_sr":   cfg["native_sr"],
            "downloaded":  downloaded,
            "default":     key == DEFAULT_MODEL,
        })
    return out


def _get_torch_device() -> str:
    import torch, subprocess, sys
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
                return "xpu"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Oracle ────────────────────────────────────────────────────────────────────

def _oracle_separation(clean_paths: list) -> list:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_paths = []
    for i, src in enumerate(clean_paths):
        dst = os.path.join(OUTPUTS_DIR, f"speaker_{i+1}.wav")
        shutil.copy(src, dst)
        out_paths.append(dst)
    print(f"[Separation] Oracle mode — {len(out_paths)} clean sources copied.")
    return out_paths


# ── IRM post-processing ───────────────────────────────────────────────────────

def _irm_refine(streams: list, mixed_audio: np.ndarray, sr: int,
                beta: float = 2.0, floor: float = 1e-8) -> list:
    """
    Ideal Ratio Mask applied to the original mixed signal.

    IRM_i(t,f) = |S_i(t,f)|^beta / ( Σ_j |S_j(t,f)|^beta + floor )
    Apply mask to S_mix → clean speaker i waveform.
    """
    from scipy.signal import stft, istft as sci_istft

    n_fft = 512
    hop   = 128

    _, _, S_mix = stft(mixed_audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)

    stream_stfts = []
    for s in streams:
        _, _, S = stft(s, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        stream_stfts.append(S)

    total_power = sum(np.abs(S) ** beta for S in stream_stfts)
    total_power = np.maximum(total_power, floor)

    refined = []
    for i, S_i in enumerate(stream_stfts):
        mask_i  = np.abs(S_i) ** beta / total_power
        S_clean = mask_i * S_mix
        _, audio_clean = sci_istft(S_clean, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)

        n = len(streams[i])
        if len(audio_clean) > n:
            audio_clean = audio_clean[:n]
        elif len(audio_clean) < n:
            audio_clean = np.pad(audio_clean, (0, n - len(audio_clean)))

        peak = np.abs(audio_clean).max()
        if peak > 0:
            audio_clean = audio_clean / peak * 0.9

        refined.append(audio_clean.astype(np.float32))
    return refined


def _wiener_filter_streams(streams: list, sr: int, alpha: float = 1.5) -> list:
    """Fallback Wiener filter when original mix is unavailable."""
    from scipy.signal import stft, istft as sci_istft

    n_fft = 512
    hop   = 128
    cleaned = []
    for i, target in enumerate(streams):
        noise = sum(other for j, other in enumerate(streams) if j != i)
        _, _, S_t = stft(target, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        _, _, S_n = stft(noise,  fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        P_t = np.abs(S_t) ** 2
        P_n = np.abs(S_n) ** 2
        gain = np.maximum(P_t - alpha * P_n, 1e-10 * P_t) / (P_t + 1e-10)
        _, audio = sci_istft(gain * S_t, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        if len(audio) > len(target):
            audio = audio[:len(target)]
        elif len(audio) < len(target):
            audio = np.pad(audio, (0, len(target) - len(audio)))
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.9
        cleaned.append(audio.astype(np.float32))
    return cleaned


# ── Model loading ─────────────────────────────────────────────────────────────

def _get_sep_model(model_name: str, device: str):
    """Load (and cache) a SepFormer model by registry key."""
    if model_name in _sep_model_cache:
        return _sep_model_cache[model_name]

    cfg = SEPARATION_MODELS.get(model_name)
    if cfg is None:
        raise ValueError(f"Unknown separation model: {model_name!r}. "
                         f"Valid: {list(SEPARATION_MODELS)}")

    from speechbrain.inference.separation import SepformerSeparation
    from huggingface_hub import snapshot_download

    local_dir = cfg["local_dir"]
    os.makedirs(local_dir, exist_ok=True)
    hub_cache = os.path.join(SEP_MODELS, "hub")

    print(f"[Separation] Loading {cfg['label']}…")
    downloaded = snapshot_download(
        repo_id=cfg["repo_id"],
        local_dir=local_dir,
        cache_dir=hub_cache,
    )
    model = SepformerSeparation.from_hparams(
        source=downloaded,
        savedir=local_dir,
        run_opts={"device": device},
    )
    _sep_model_cache[model_name] = model
    print(f"[Separation] {cfg['label']} loaded on {device}.")
    return model


# ── Core separation ───────────────────────────────────────────────────────────

def _sepformer_separation(mixed_path: str, num_speakers: int,
                          model_name: str = DEFAULT_MODEL) -> list:
    import torch, torchaudio
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    cfg    = SEPARATION_MODELS[model_name]
    native = cfg["native_sr"]
    device = _get_torch_device()
    model  = _get_sep_model(model_name, device)

    audio_np, file_sr = sf.read(mixed_path, dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)

    # Waveform for the model (resample to model's native SR if needed)
    waveform = torch.from_numpy(audio_np).unsqueeze(0)
    if file_sr != native:
        waveform = torchaudio.functional.resample(waveform, file_sr, native)
    waveform = waveform.to(device)

    # Keep a 16 kHz copy of the mix for IRM post-processing
    if file_sr != OUTPUT_SR:
        mix_16k = torchaudio.functional.resample(
            torch.from_numpy(audio_np).unsqueeze(0), file_sr, OUTPUT_SR
        ).squeeze(0).numpy()
    else:
        mix_16k = audio_np.copy()

    print(f"[Separation] Running {cfg['label']} ({device})…")
    with torch.no_grad():
        est_sources = model.separate_batch(waveform)
    est_sources = est_sources.squeeze(0).cpu()   # (samples, n_src)

    n_out = min(num_speakers, est_sources.shape[1])
    raw_streams = []
    for i in range(n_out):
        src_np = est_sources[:, i].numpy()
        # Upsample to 16 kHz if model runs at lower SR
        if native != OUTPUT_SR:
            src_t  = torch.from_numpy(src_np).unsqueeze(0)
            src_t  = torchaudio.functional.resample(src_t, native, OUTPUT_SR)
            src_np = src_t.squeeze(0).numpy()
        peak = np.abs(src_np).max()
        if peak > 0:
            src_np = src_np / peak * 0.9
        raw_streams.append(src_np.astype(np.float32))

    print(f"[Separation] {cfg['label']} done. Applying IRM post-processing…")
    try:
        min_len       = min(len(mix_16k), min(len(s) for s in raw_streams))
        refined       = _irm_refine([s[:min_len] for s in raw_streams],
                                    mix_16k[:min_len], OUTPUT_SR, beta=2.0)
    except Exception as e:
        print(f"[Separation] IRM failed ({e}), trying Wiener filter…")
        try:
            refined = _wiener_filter_streams(raw_streams, OUTPUT_SR, alpha=1.5)
        except Exception as e2:
            print(f"[Separation] Wiener also failed ({e2}), using raw streams.")
            refined = raw_streams

    out_paths = []
    for i, src_np in enumerate(refined):
        out_path = os.path.join(OUTPUTS_DIR, f"speaker_{i+1}.wav")
        sf.write(out_path, src_np, OUTPUT_SR)
        out_paths.append(out_path)

    print(f"[Separation] Done — {len(out_paths)} streams on {device} "
          f"({cfg['label']} + IRM).")
    return out_paths


def _naive_fallback(mixed_path: str, num_speakers: int) -> list:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    audio, sr = sf.read(mixed_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    out_paths = []
    for i in range(num_speakers):
        p = os.path.join(OUTPUTS_DIR, f"speaker_{i+1}.wav")
        sf.write(p, audio, sr)
        out_paths.append(p)
    print("[Separation] WARNING: Naive fallback — duplicated mixed audio.")
    return out_paths


# ── Public API ────────────────────────────────────────────────────────────────

def separate(
    mixed_path:   str,
    clean_paths:  list = None,
    num_speakers: int  = 2,
    demo_oracle:  bool = False,
    sep_model:    str  = DEFAULT_MODEL,
) -> dict:
    """
    Separate mixed audio into per-speaker streams.

    Args:
        mixed_path   : path to the mixed WAV file
        clean_paths  : list of clean source WAVs (oracle / fallback)
        num_speakers : expected number of speakers
        demo_oracle  : if True, copy clean_paths directly (skip ML)
        sep_model    : model key from SEPARATION_MODELS (default: sepformer_whamr16k)

    Returns:
        {
          "separated_paths": [str, ...],
          "mode":    "oracle" | "<model_key>" | "naive_fallback",
          "warning": str | None,
          "model_label": str,
        }
    """
    # Validate / normalise model key
    if sep_model not in SEPARATION_MODELS:
        print(f"[Separation] Unknown model {sep_model!r}, using {DEFAULT_MODEL}.")
        sep_model = DEFAULT_MODEL

    model_label = SEPARATION_MODELS[sep_model]["label"]

    if demo_oracle:
        if clean_paths:
            paths = _oracle_separation(clean_paths)
            return {"separated_paths": paths, "mode": "oracle",
                    "model_label": "Oracle", "warning": None}
        paths = _naive_fallback(mixed_path, num_speakers)
        return {
            "separated_paths": paths,
            "mode": "naive_fallback",
            "model_label": "Naive fallback",
            "warning": "Oracle requested but no clean sources available.",
        }

    try:
        paths = _sepformer_separation(mixed_path, num_speakers, sep_model)
        return {"separated_paths": paths, "mode": sep_model,
                "model_label": model_label, "warning": None}
    except Exception as e:
        print(f"[Separation] {model_label} failed: {e}")
        if clean_paths:
            paths = _oracle_separation(clean_paths)
            return {"separated_paths": paths, "mode": "oracle",
                    "model_label": "Oracle (fallback)",
                    "warning": f"{model_label} failed ({e}); used clean oracle sources."}
        paths = _naive_fallback(mixed_path, num_speakers)
        return {"separated_paths": paths, "mode": "naive_fallback",
                "model_label": "Naive fallback",
                "warning": f"{model_label} failed ({e}); mixed audio duplicated."}
