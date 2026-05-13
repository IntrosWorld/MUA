# Real-Time Multi-User Smart Assistant for Dynamic and Noisy Smart Environments (MultiVox)

A hackathon prototype that separates overlapping voices, identifies speakers, transcribes speech, detects smart-home commands, and returns personalized responses — all through a modern web UI.

---

## Quick Start

### 1. Create a virtual environment (inside this folder)

```bash
cd multi_user_smart_assistant
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch` and `speechbrain` are large. If you only want the demo oracle mode (no real model), the minimum working set is:  
> `fastapi uvicorn[standard] python-multipart soundfile librosa numpy edge-tts faster-whisper jiwer pydub`

### 3. Run the server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open in browser

```
http://127.0.0.1:8000
```

---

## How to Use the UI

1. **Configure** speakers (2 or 3), noise level, overlap type, and separation mode.
2. **Select commands** for each speaker from the dropdowns.
3. Click **Generate Demo Audio** — this creates TTS voices for Mohit, Rahul, and Priya and mixes them.
4. Click **Process Demo Audio** — the full pipeline runs and results appear.
5. Alternatively, **Upload Audio** your own WAV/MP3 and click **Process Uploaded**.

---

## Architecture

```
Mixed Audio (WAV)
      │
      ▼
  VAD (energy threshold)
      │
      ▼
  Speech Separation
  ┌─────────────────────────────────┐
  │  Oracle mode  │  SepFormer      │
  │  (demo)       │  (real model)   │
  └─────────────────────────────────┘
      │
      ▼  (per separated stream)
  Speaker ID (ECAPA-TDNN cosine similarity)
      │
  ASR (faster-whisper base)
      │
  Intent Detection (rule-based)
      │
  Personalized Response Engine
      │
      ▼
  JSON report + Web UI
```

---

## Synthetic Users (TTS)

| User  | TTS Voice              | Room              | Playlist                  |
|-------|------------------------|-------------------|---------------------------|
| Mohit | en-IN-PrabhatNeural    | Mohit's bedroom   | Mohit's rock playlist     |
| Rahul | en-US-GuyNeural        | Rahul's study room| Rahul's focus playlist    |
| Priya | en-IN-NeerjaNeural     | Priya's room      | Priya's relaxing playlist |

TTS voices are generated via **edge-tts** (Microsoft Azure Neural TTS, free, no API key needed).

---

## Demo Oracle vs Real Separation

| Mode           | Description                                                      |
|----------------|------------------------------------------------------------------|
| **Oracle**     | Uses the clean TTS-generated sources directly. Always works, WER/SI-SNR calculable. |
| **Real Model** | Attempts to load SpeechBrain SepFormer. Downloads ~300 MB. Falls back to naive if unavailable. |

---

## Using Real Recorded Voices

Replace the TTS enrollment files in `data/enrolled_speakers/<name>/` with real WAV recordings (16 kHz mono) and re-run speaker enrollment. The system will automatically recompute embeddings.

---

## KPIs

| Metric | Description |
|--------|-------------|
| **xRT** | `processing_time / audio_duration` — lower means faster than real-time |
| **WER** | Word Error Rate comparing ASR output to ground truth (via `jiwer`) |
| **SI-SNR** | Scale-Invariant Signal-to-Noise Ratio between clean source and separated output |
| **Model Params** | PyTorch parameter count for the separation model (N/A in oracle mode) |

---

## Model Downloads (all go to `models/` in this directory)

| Model | Size | Purpose |
|-------|------|---------|
| faster-whisper base | ~150 MB | ASR |
| speechbrain/spkrec-ecapa-voxceleb | ~100 MB | Speaker embeddings |
| speechbrain/sepformer-whamr | ~300 MB | Speech separation (optional) |

All downloads are redirected to `multi_user_smart_assistant/models/` — nothing goes to the C drive.

---

## Project Structure

```
multi_user_smart_assistant/
├── app.py                  FastAPI backend
├── requirements.txt
├── README.md
├── models/                 All model downloads (local, not C drive)
├── data/
│   ├── enrolled_speakers/  Enrollment WAV files per user
│   ├── generated/          TTS command WAVs
│   ├── mixed/              Mixed audio + ground truth JSON
│   └── noise/              (optional) noise samples
├── outputs/
│   ├── separated/          Separated speaker streams
│   ├── reports/            Pipeline JSON reports
│   └── speaker_db.pkl      Speaker embedding database
├── src/
│   ├── tts_generate.py
│   ├── audio_mixer.py
│   ├── vad.py
│   ├── separation.py
│   ├── speaker_id.py
│   ├── asr.py
│   ├── intent.py
│   ├── response_engine.py
│   ├── metrics.py
│   ├── pipeline.py
│   └── utils.py
└── frontend/
    ├── index.html
    ├── style.css
    └── script.js
```
