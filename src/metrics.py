import numpy as np
import soundfile as sf


def compute_xrt(processing_time: float, audio_duration: float) -> float:
    if audio_duration <= 0:
        return 0.0
    return round(processing_time / audio_duration, 4)


def compute_wer(reference: str, hypothesis: str) -> float:
    try:
        import jiwer
        transforms = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(),
        ])
        result = jiwer.wer(
            reference,
            hypothesis,
            reference_transform=transforms,
            hypothesis_transform=transforms,
        )
        return round(result, 4)
    except Exception as e:
        print(f"[Metrics] WER computation failed: {e}")
        return None


def compute_si_snr(reference_path: str, estimated_path: str) -> float:
    """Scale-Invariant SNR between clean reference and separated estimate."""
    try:
        ref, sr1 = sf.read(reference_path, dtype="float32")
        est, sr2 = sf.read(estimated_path, dtype="float32")
        if ref.ndim > 1:
            ref = ref.mean(axis=1)
        if est.ndim > 1:
            est = est.mean(axis=1)
        # Match lengths
        min_len = min(len(ref), len(est))
        if min_len < 100:
            return None
        ref = ref[:min_len]
        est = est[:min_len]
        # Remove DC
        ref = ref - ref.mean()
        est = est - est.mean()
        # SI-SNR
        dot = np.dot(est, ref)
        ref_energy = np.dot(ref, ref) + 1e-8
        s_target = (dot / ref_energy) * ref
        e_noise = est - s_target
        si_snr = 10 * np.log10((np.dot(s_target, s_target) + 1e-8) / (np.dot(e_noise, e_noise) + 1e-8))
        return round(float(si_snr), 2)
    except Exception as e:
        print(f"[Metrics] SI-SNR computation failed: {e}")
        return None


def count_model_params(model) -> str:
    try:
        import torch
        total = sum(p.numel() for p in model.parameters())
        return f"{total:,}"
    except Exception:
        return "N/A"


def build_report(
    audio_duration: float,
    processing_time: float,
    wer_pairs: list = None,         # list of (reference, hypothesis)
    si_snr_pairs: list = None,      # list of (ref_path, est_path)
    separation_mode: str = "oracle",
    model=None,
) -> dict:
    report = {
        "audio_duration": round(audio_duration, 3),
        "processing_time": round(processing_time, 3),
        "xRT": compute_xrt(processing_time, audio_duration),
        "wer": None,
        "si_snr": None,
        "model_params": "Oracle/demo separation mode, parameter count not applicable."
        if separation_mode == "oracle"
        else count_model_params(model),
        "separation_mode": separation_mode,
    }

    if wer_pairs:
        refs = " | ".join(r for r, _ in wer_pairs)
        hyps = " | ".join(h for _, h in wer_pairs)
        report["wer"] = compute_wer(refs, hyps)

    if si_snr_pairs:
        snrs = [compute_si_snr(r, e) for r, e in si_snr_pairs if r and e]
        snrs = [v for v in snrs if v is not None]
        report["si_snr"] = round(np.mean(snrs), 2) if snrs else None

    return report
