import os
import numpy as np
import soundfile as sf

SR = 16000


def _energy_vad(audio: np.ndarray, frame_ms: int = 30, threshold_ratio: float = 0.05) -> list:
    frame_len = int(SR * frame_ms / 1000)
    energies = []
    for i in range(0, len(audio) - frame_len, frame_len):
        frame = audio[i: i + frame_len]
        energies.append(np.sqrt(np.mean(frame ** 2)))

    if not energies:
        return [{"start": 0.0, "end": len(audio) / SR}]

    threshold = max(energies) * threshold_ratio
    segments = []
    in_speech = False
    seg_start = 0

    for i, e in enumerate(energies):
        t = i * frame_ms / 1000
        if not in_speech and e > threshold:
            in_speech = True
            seg_start = t
        elif in_speech and e <= threshold:
            in_speech = False
            segments.append({"start": seg_start, "end": t})

    if in_speech:
        segments.append({"start": seg_start, "end": len(audio) / SR})

    # Merge segments closer than 300 ms
    merged = []
    for seg in segments:
        if merged and seg["start"] - merged[-1]["end"] < 0.3:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))

    return merged if merged else [{"start": 0.0, "end": len(audio) / SR}]


def detect_speech_segments(audio_path: str) -> list:
    """Return list of {start, end} dicts in seconds."""
    try:
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        segments = _energy_vad(audio)
        print(f"[VAD] Detected {len(segments)} speech segment(s) in {os.path.basename(audio_path)}")
        return segments
    except Exception as e:
        print(f"[VAD] Error: {e} — returning full audio as one segment")
        return [{"start": 0.0, "end": 10.0}]


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        segs = detect_speech_segments(sys.argv[1])
        for s in segs:
            print(f"  {s['start']:.2f}s — {s['end']:.2f}s")
