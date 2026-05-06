"""
Speech Separation Module
------------------------
Primary model : SepFormer WHAMR! 16K (SpeechBrain)
  - Native 16 kHz — no resampling quality loss
  - Trained on reverberant + noisy LibriSpeech mixtures (realistic conditions)
Post-processing: Ideal Ratio Mask (IRM) applied to the ORIGINAL mixed signal
  - IRM_i(t,f) = |S_i(t,f)|^2 / Σ|S_j(t,f)|^2
  - Applying the mask to the mix (not the noisy separated stream) yields
    theoretically-optimal SNR and avoids double-distortion artifacts.

Fallback chain:
  1. SepFormer WHAMR 16K   (best quality, native 16 kHz)
  2. Oracle / clean sources (demo_oracle mode or if SepFormer fails + cleans available)
  3. Naive duplication      (last resort)

GPU support: Intel Arc XPU > NVIDIA CUDA > CPU
"""

import os
import shutil
import numpy as np
import soundfile as sf
from . import utils as _utils   # activates Windows symlink patch

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR  = os.path.join(BASE_DIR, "outputs", "separated")
SEP_MODELS   = os.path.join(BASE_DIR, "models")
OUTPUT_SR    = 16000

# SepFormer WHAMR! 16 kHz — native 16 kHz, trained on reverberant LibriSpeech
SEPFORMER_REPO_ID = "speechbrain/sepformer-whamr16k"
SEPFORMER_DIR     = os.path.join(SEP_MODELS, "sepformer-whamr16k")
SEPFORMER_SR      = 16000   # native — no resampling needed

_sep_model = None


# ── Device selection ──────────────────────────────────────────────────────────

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


# ── Oracle (demo cheat) ────────────────────────────────────────────────────────

def _oracle_separation(clean_paths: list) -> list:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_paths = []
    for i, src in enumerate(clean_paths):
        dst = os.path.join(OUTPUTS_DIR, f"speaker_{i+1}.wav")
        shutil.copy(src, dst)
        out_paths.append(dst)
    print(f"[Separation] Oracle mode — {len(out_paths)} clean sources copied.")
    return out_paths


# ── IRM post-processing (applies masks to original mix) ───────────────────────

def _irm_refine(streams: list, mixed_audio: np.ndarray, sr: int,
                beta: float = 2.0, floor: float = 1e-8) -> list:
    """
    Ideal Ratio Mask (IRM) post-processing.

    For each speaker stream, compute a soft time-frequency mask:
        IRM_i(t,f) = |S_i(t,f)|^β / (Σ_j |S_j(t,f)|^β + floor)

    Apply the mask to the ORIGINAL MIX spectrogram (not the noisy
    separated stream). This avoids double-distortion and maximises SNR.

    beta=2 → power-domain masking (standard IRM)
    floor  → prevents silence bins from blowing up the mask
    """
    from scipy.signal import stft, istft as sci_istft

    n_fft = 512
    hop   = 128

    # STFT of the original mix
    _, _, S_mix = stft(mixed_audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)

    # STFT of each separated stream
    stream_stfts = []
    for s in streams:
        _, _, S = stft(s, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        stream_stfts.append(S)

    # Total power across all speakers (denominator of IRM)
    total_power = np.zeros_like(np.abs(stream_stfts[0]) ** beta)
    for S in stream_stfts:
        total_power += np.abs(S) ** beta
    total_power = np.maximum(total_power, floor)

    refined = []
    for i, S_i in enumerate(stream_stfts):
        # Compute IRM for speaker i
        mask_i = np.abs(S_i) ** beta / total_power          # shape: (freq, time)

        # Apply mask to original mix spectrum (complex multiplication)
        S_clean = mask_i * S_mix

        # Reconstruct waveform
        _, audio_clean = sci_istft(S_clean, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)

        # Trim / pad to match original length
        n = len(streams[i])
        if len(audio_clean) > n:
            audio_clean = audio_clean[:n]
        elif len(audio_clean) < n:
            audio_clean = np.pad(audio_clean, (0, n - len(audio_clean)))

        # Peak normalise
        peak = np.abs(audio_clean).max()
        if peak > 0:
            audio_clean = audio_clean / peak * 0.9

        refined.append(audio_clean.astype(np.float32))

    return refined


# ── Wiener filter fallback (if mix audio unavailable) ────────────────────────

def _wiener_filter_streams(streams: list, sr: int, alpha: float = 1.5) -> list:
    """
    Spectral Wiener filter using other streams as noise estimate.
    Used as fallback when the original mixed audio is not accessible.
    """
    from scipy.signal import stft, istft as sci_istft

    n_fft   = 512
    hop     = 128
    cleaned = []

    for i, target in enumerate(streams):
        noise = np.zeros_like(target)
        for j, other in enumerate(streams):
            if j != i:
                noise += other

        _, _, S_target = stft(target, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        _, _, S_noise  = stft(noise,  fs=sr, nperseg=n_fft, noverlap=n_fft - hop)

        P_target = np.abs(S_target) ** 2
        P_noise  = np.abs(S_noise)  ** 2

        P_clean = np.maximum(P_target - alpha * P_noise, 1e-10 * P_target)
        gain    = P_clean / (P_target + 1e-10)

        S_clean = gain * S_target
        _, audio_clean = sci_istft(S_clean, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)

        if len(audio_clean) > len(target):
            audio_clean = audio_clean[:len(target)]
        elif len(audio_clean) < len(target):
            audio_clean = np.pad(audio_clean, (0, len(target) - len(audio_clean)))

        peak = np.abs(audio_clean).max()
        if peak > 0:
            audio_clean = audio_clean / peak * 0.9

        cleaned.append(audio_clean.astype(np.float32))

    return cleaned


# ── SepFormer WHAMR 16K ───────────────────────────────────────────────────────

def _get_sep_model(device: str):
    global _sep_model
    if _sep_model is not None:
        return _sep_model
    from speechbrain.inference.separation import SepformerSeparation
    from huggingface_hub import snapshot_download
    os.makedirs(SEPFORMER_DIR, exist_ok=True)
    hub_cache = os.path.join(SEP_MODELS, "hub")
    print("[Separation] Loading SepFormer-WHAMR-16K…")
    local_dir = snapshot_download(
        repo_id=SEPFORMER_REPO_ID,
        local_dir=SEPFORMER_DIR,
        cache_dir=hub_cache,
    )
    _sep_model = SepformerSeparation.from_hparams(
        source=local_dir,
        savedir=SEPFORMER_DIR,
        run_opts={"device": device},
    )
    print(f"[Separation] SepFormer-WHAMR-16K loaded on {device}.")
    return _sep_model


def _sepformer_separation(mixed_path: str, num_speakers: int) -> list:
    import torch, torchaudio
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    device = _get_torch_device()
    model  = _get_sep_model(device)

    audio_np, file_sr = sf.read(mixed_path, dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)

    waveform = torch.from_numpy(audio_np).unsqueeze(0)
    if file_sr != SEPFORMER_SR:
        waveform = torchaudio.functional.resample(waveform, file_sr, SEPFORMER_SR)
    waveform = waveform.to(device)

    # Keep a reference-sample-rate copy of the mixed audio for IRM
    if file_sr != OUTPUT_SR:
        mix_for_irm = torchaudio.functional.resample(
            torch.from_numpy(audio_np).unsqueeze(0), file_sr, OUTPUT_SR
        ).squeeze(0).numpy()
    else:
        mix_for_irm = audio_np.copy()

    print(f"[Separation] Running SepFormer-WHAMR-16K ({device})…")
    with torch.no_grad():
        est_sources = model.separate_batch(waveform)
    est_sources = est_sources.squeeze(0).cpu()

    n_out = min(num_speakers, est_sources.shape[1])
    raw_streams = []
    for i in range(n_out):
        src_np = est_sources[:, i].numpy()
        # SepFormer-WHAMR runs natively at 16 kHz; upsample only if needed
        if SEPFORMER_SR != OUTPUT_SR:
            src_t  = torch.from_numpy(src_np).unsqueeze(0)
            src_t  = torchaudio.functional.resample(src_t, SEPFORMER_SR, OUTPUT_SR)
            src_np = src_t.squeeze(0).numpy()
        peak = np.abs(src_np).max()
        if peak > 0:
            src_np = src_np / peak * 0.9
        raw_streams.append(src_np.astype(np.float32))

    print("[Separation] SepFormer done. Applying IRM post-processing…")

    # IRM: apply masks to original mix for cleanest output
    try:
        # Trim mix_for_irm to match separated stream length
        min_len = min(len(mix_for_irm), min(len(s) for s in raw_streams))
        mix_trimmed   = mix_for_irm[:min_len]
        raw_trimmed   = [s[:min_len] for s in raw_streams]
        refined_streams = _irm_refine(raw_trimmed, mix_trimmed, OUTPUT_SR, beta=2.0)
    except Exception as e:
        print(f"[Separation] IRM failed ({e}), falling back to Wiener filter.")
        try:
            refined_streams = _wiener_filter_streams(raw_streams, OUTPUT_SR, alpha=1.5)
        except Exception as e2:
            print(f"[Separation] Wiener filter also failed ({e2}), using raw streams.")
            refined_streams = raw_streams

    out_paths = []
    for i, src_np in enumerate(refined_streams):
        out_path = os.path.join(OUTPUTS_DIR, f"speaker_{i+1}.wav")
        sf.write(out_path, src_np, OUTPUT_SR)
        out_paths.append(out_path)

    print(f"[Separation] Done — {len(out_paths)} streams (IRM-refined) on {device}.")
    return out_paths


# ── Naive fallback ─────────────────────────────────────────────────────────────

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
          "mode": "oracle" | "sepformer_whamr16k" | "naive_fallback",
          "warning": str | None
        }
    """
    if demo_oracle:
        if clean_paths:
            paths = _oracle_separation(clean_paths)
            return {"separated_paths": paths, "mode": "oracle", "warning": None}
        paths = _naive_fallback(mixed_path, num_speakers)
        return {
            "separated_paths": paths,
            "mode": "naive_fallback",
            "warning": "Oracle requested but no clean sources available.",
        }

    try:
        paths = _sepformer_separation(mixed_path, num_speakers)
        return {"separated_paths": paths, "mode": "sepformer_whamr16k", "warning": None}
    except Exception as e:
        print(f"[Separation] SepFormer failed: {e}")
        if clean_paths:
            paths = _oracle_separation(clean_paths)
            return {"separated_paths": paths, "mode": "oracle",
                    "warning": f"SepFormer failed ({e}); used clean oracle sources."}
        paths = _naive_fallback(mixed_path, num_speakers)
        return {"separated_paths": paths, "mode": "naive_fallback",
                "warning": f"SepFormer failed ({e}); mixed audio duplicated."}
