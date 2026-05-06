import os
import time
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

from .vad import detect_speech_segments
from .separation import separate
from .speaker_id import identify_speaker, enroll_speakers, assign_speakers_optimal
from .asr import transcribe
from .intent import detect_intent
from .response_engine import generate_response as _generate_response
from .metrics import build_report, compute_xrt
from .utils import get_audio_duration, load_ground_truth, save_json


def run_pipeline(
    input_audio_path: str,
    ground_truth_path: str = None,
    clean_paths: list = None,
    demo_oracle: bool = True,
    num_speakers: int = None,
    use_llm: bool = True,
    llm_mode: str = None,
) -> dict:
    t_start = time.time()
    audio_duration = get_audio_duration(input_audio_path)

    # Load ground truth if available
    gt = {"sources": []}
    if ground_truth_path and os.path.exists(ground_truth_path):
        gt = load_ground_truth(ground_truth_path)

    gt_sources = gt.get("sources", [])
    if num_speakers is None:
        num_speakers = max(len(gt_sources), 2)

    # Step 1: VAD
    print("[Pipeline] Running VAD...")
    speech_segments = detect_speech_segments(input_audio_path)

    # Step 2: Ensure speakers are enrolled
    db_path = os.path.join(OUTPUTS_DIR, "speaker_db.pkl")
    if not os.path.exists(db_path):
        print("[Pipeline] Enrolling speakers...")
        enroll_speakers()

    # Step 3: Separation
    print("[Pipeline] Running separation...")
    sep_result = separate(
        mixed_path=input_audio_path,
        clean_paths=clean_paths,
        num_speakers=num_speakers,
        demo_oracle=demo_oracle,
    )
    separated_paths = sep_result["separated_paths"]
    sep_mode = sep_result["mode"]
    sep_warning = sep_result.get("warning")

    # Step 4: Per-speaker processing
    results = []
    wer_pairs = []
    si_snr_pairs = []

    # --- Optimal speaker assignment across ALL streams at once ---
    # (prevents two streams both matching the same speaker)
    gt_speaker_list = []
    for i in range(len(separated_paths)):
        ge = gt_sources[i] if i < len(gt_sources) else {}
        gt_speaker_list.append(ge.get("speaker", "").lower() if ge else None)

    if demo_oracle and any(gt_speaker_list):
        # Oracle mode: use ground-truth speaker labels directly
        id_results = []
        for gs in gt_speaker_list:
            if gs:
                id_results.append({"speaker": gs.capitalize(), "confidence": 1.0, "fallback": True})
            else:
                id_results.append({"speaker": "Unknown", "confidence": 0.0, "fallback": True})
    else:
        print("[Pipeline] Running optimal speaker assignment...")
        id_results = assign_speakers_optimal(
            separated_paths,
            gt_speakers=gt_speaker_list if gt_speaker_list else None,
        )

    for i, spk_path in enumerate(separated_paths):
        print(f"[Pipeline] Processing speaker stream {i+1}...")

        gt_entry  = gt_sources[i] if i < len(gt_sources) else {}
        id_result = id_results[i]

        # ASR
        gt_text = gt_entry.get("text") if gt_entry else None
        asr_fallback = gt_text if demo_oracle else None
        asr_result = transcribe(spk_path, fallback_text=asr_fallback)
        transcript = asr_result["text"]

        # Intent
        intent_result = detect_intent(transcript)

        # Response (LLM or rule-based)
        resp_result = _generate_response(
            speaker=id_result["speaker"],
            intent=intent_result["intent"],
            slots=intent_result.get("slots", {}),
            transcript=transcript,
            use_llm=use_llm,
            llm_mode=llm_mode,
        )
        response = resp_result["response"]

        # Collect for metrics
        if gt_text and transcript:
            wer_pairs.append((gt_text, transcript))

        clean_ref = gt_entry.get("file") if gt_entry else None
        if clean_ref and os.path.exists(clean_ref):
            si_snr_pairs.append((clean_ref, spk_path))

        results.append({
            "stream": os.path.basename(spk_path),
            "stream_path": spk_path,
            "identified_speaker": id_result["speaker"],
            "speaker_confidence": id_result["confidence"],
            "speaker_fallback": id_result.get("fallback", False),
            "transcript": transcript,
            "asr_fallback": asr_result.get("fallback", False),
            "asr_warning": asr_result.get("warning"),
            "intent": intent_result["intent"],
            "is_command": intent_result["is_command"],
            "wake_word_detected": intent_result.get("wake_word_detected", False),
            "slots": intent_result.get("slots", {}),
            "response": response,
            "response_source": resp_result.get("source", "rule"),
            "llm_mode": resp_result.get("llm_mode"),
            "llm_model": resp_result.get("model"),
        })

    processing_time = time.time() - t_start

    # Step 5: Metrics
    metrics = build_report(
        audio_duration=audio_duration,
        processing_time=processing_time,
        wer_pairs=wer_pairs if wer_pairs else None,
        si_snr_pairs=si_snr_pairs if si_snr_pairs else None,
        separation_mode=sep_mode,
    )

    report = {
        "input_audio": input_audio_path,
        "detected_speakers": len(results),
        "speech_segments": speech_segments,
        "separation_mode": sep_mode,
        "separation_warning": sep_warning,
        "results": results,
        "metrics": metrics,
    }

    # Save report
    report_path = os.path.join(OUTPUTS_DIR, "reports", "latest_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    save_json(report, report_path)
    print(f"[Pipeline] Done. Report saved to {report_path}")
    return report
