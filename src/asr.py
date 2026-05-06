"""
ASR module — faster-whisper large-v3
Model downloads automatically to <project>/models/whisper/ (~1.6 GB int8)
"""

import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

_whisper_model    = None
_whisper_available = None

# Change model size here: "tiny" | "base" | "small" | "medium" | "large-v2" | "large-v3"
WHISPER_MODEL_SIZE = "large-v3"


def _get_whisper():
    global _whisper_model, _whisper_available
    if _whisper_available is not None:
        return _whisper_model

    try:
        from faster_whisper import WhisperModel
        model_dir = os.path.join(MODELS_DIR, "whisper")
        os.makedirs(model_dir, exist_ok=True)

        print(f"[ASR] Loading faster-whisper {WHISPER_MODEL_SIZE} "
              f"(first run downloads ~1.6 GB to {model_dir})...")
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8",          # fastest on CPU, still very accurate
            download_root=model_dir,
            cpu_threads=max(4, os.cpu_count() or 4),  # use all cores
            num_workers=1,
        )
        _whisper_available = True
        print(f"[ASR] faster-whisper {WHISPER_MODEL_SIZE} loaded.")
        return _whisper_model
    except Exception as e:
        print(f"[ASR] faster-whisper unavailable: {e}")
        _whisper_available = False
        return None


def transcribe(audio_path: str, fallback_text: str = None) -> dict:
    """
    Transcribe audio with faster-whisper large-v3.
    Returns {"text": str, "fallback": bool, "warning": str | None}
    """
    model = _get_whisper()
    if model is None:
        if fallback_text:
            print(f"[ASR] Using ground-truth fallback for {os.path.basename(audio_path)}")
            return {
                "text": fallback_text,
                "fallback": True,
                "warning": "ASR model unavailable; using ground-truth transcript.",
            }
        return {
            "text": "",
            "fallback": True,
            "warning": "ASR model unavailable and no fallback text provided.",
        }

    try:
        segments, info = model.transcribe(
            audio_path,
            language="en",
            beam_size=10,                  # more beams → better accuracy
            best_of=5,
            patience=2.0,                  # explore more hypotheses
            vad_filter=True,               # skip silence regions
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=200,
            ),
            condition_on_previous_text=False,  # avoid hallucination loops
            no_speech_threshold=0.5,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            word_timestamps=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if not text:
            # Retry without VAD filter if nothing came back
            segments2, _ = model.transcribe(
                audio_path,
                language="en",
                beam_size=10,
                best_of=5,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            text = " ".join(seg.text.strip() for seg in segments2).strip()
        return {"text": text, "fallback": False, "warning": None}
    except Exception as e:
        print(f"[ASR] Transcription error: {e}")
        if fallback_text:
            return {
                "text": fallback_text,
                "fallback": True,
                "warning": f"ASR error ({e}); using ground-truth fallback.",
            }
        return {"text": "", "fallback": True, "warning": str(e)}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = transcribe(sys.argv[1])
        print(result)
