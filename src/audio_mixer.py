import os
import json
import numpy as np
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MIXED_DIR = os.path.join(DATA_DIR, "mixed")
SR = 16000


def _load_wav(path: str) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=SR)
    return data


def _generate_white_noise(length: int, level: float = 0.02) -> np.ndarray:
    return np.random.randn(length).astype(np.float32) * level


def _generate_fan_noise(length: int, level: float = 0.015) -> np.ndarray:
    t = np.linspace(0, length / SR, length, dtype=np.float32)
    noise = (
        np.sin(2 * np.pi * 50 * t) * 0.5
        + np.sin(2 * np.pi * 120 * t) * 0.3
        + np.random.randn(length).astype(np.float32) * 0.2
    )
    return noise * level


def _mix_sources(sources: list, overlap_type: str) -> tuple:
    """Return (mixed, padded_sources) all at length of longest."""
    if overlap_type == "full":
        max_len = max(len(s) for s in sources)
        padded = [np.pad(s, (0, max_len - len(s))) for s in sources]
        mixed = sum(padded)
    elif overlap_type == "partial":
        # Each source starts at 40% offset from the previous
        offset = 0
        segments = []
        for s in sources:
            start = int(offset)
            segments.append((start, s))
            offset += len(s) * 0.6
        total_len = max(start + len(s) for start, s in segments)
        padded = []
        mixed = np.zeros(total_len, dtype=np.float32)
        for start, s in segments:
            arr = np.zeros(total_len, dtype=np.float32)
            arr[start: start + len(s)] = s
            padded.append(arr)
            mixed += arr
    else:  # no overlap — sequential
        total_len = sum(len(s) for s in sources)
        padded = []
        mixed = np.zeros(total_len, dtype=np.float32)
        cursor = 0
        for s in sources:
            arr = np.zeros(total_len, dtype=np.float32)
            arr[cursor: cursor + len(s)] = s
            padded.append(arr)
            mixed += arr
            cursor += len(s)
    return mixed, padded


def _noise_level_to_amplitude(level: str) -> float:
    return {"none": 0.0, "low": 0.005, "medium": 0.02, "high": 0.06}.get(level, 0.0)


def mix_audio(
    speaker_files: list,  # list of (speaker_name, wav_path, text)
    overlap_type: str = "partial",
    noise_level: str = "low",
    noise_type: str = "white",
    output_prefix: str = "mixed",
) -> dict:
    os.makedirs(MIXED_DIR, exist_ok=True)

    sources_raw = [_load_wav(path) for _, path, _ in speaker_files]
    mixed, padded = _mix_sources(sources_raw, overlap_type)
    total_len = len(mixed)

    # Add noise
    amp = _noise_level_to_amplitude(noise_level)
    if amp > 0:
        if noise_type == "fan":
            noise = _generate_fan_noise(total_len, amp)
        else:
            noise = _generate_white_noise(total_len, amp)
        mixed = mixed + noise

    # Normalize
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed = mixed / peak * 0.9

    mixed_path = os.path.join(MIXED_DIR, f"{output_prefix}.wav")
    sf.write(mixed_path, mixed, SR)

    clean_paths = []
    ground_truth_sources = []
    for idx, (spk, _, text) in enumerate(speaker_files):
        clean_arr = padded[idx]
        # Normalize clean source relative to same scale
        if peak > 0:
            clean_arr = clean_arr / peak * 0.9
        clean_path = os.path.join(MIXED_DIR, f"clean_source_{idx+1}.wav")
        sf.write(clean_path, clean_arr, SR)
        clean_paths.append(clean_path)
        ground_truth_sources.append({
            "speaker": spk.capitalize(),
            "text": text,
            "file": clean_path,
        })

    gt = {"sources": ground_truth_sources}
    gt_path = os.path.join(MIXED_DIR, f"{output_prefix}_ground_truth.json")
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)

    print(f"[Mixer] Mixed audio saved to {mixed_path}")
    return {
        "mixed_path": mixed_path,
        "clean_paths": clean_paths,
        "ground_truth_path": gt_path,
        "ground_truth": gt,
    }


if __name__ == "__main__":
    from tts_generate import run_generate_all, get_command_path
    run_generate_all()
    files = [
        ("mohit", get_command_path("mohit", 0), "Hey assistant, turn on my bedroom light."),
        ("rahul", get_command_path("rahul", 0), "Hey assistant, play my study playlist."),
        ("priya", get_command_path("priya", 0), "Hey assistant, remind me to call home."),
    ]
    result = mix_audio(files, overlap_type="partial", noise_level="low")
    print(result)
