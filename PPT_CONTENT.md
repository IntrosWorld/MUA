# PPT Content — Multi-User Smart Assistant
---

## SLIDE 1 — [Leave as per college template]

**Title:** Multi-User Smart Voice Assistant
**Subtitle:** Separating, Identifying & Responding to Multiple Speakers in Real Time

| Field | Value |
|---|---|
| Team Name | [Your Team Name] |
| Team Members | [Name 1], [Name 2], [Name 3] |
| Institution | [College Name] |
| Department | [Dept] |
| Date | May 2026 |

---

## SLIDE 2 — Problem Statement

### Selected Problem:
**Building a Smart Voice Assistant That Understands and Responds to Multiple Users Simultaneously**

### Problem Understanding:
Every voice assistant that exists today — Alexa, Google Assistant, Siri — is designed for **one speaker at a time**. The moment two people speak at the same time or in quick succession, the system breaks. It mishears, misidentifies, or simply ignores one of them entirely.

**The real world does not work that way.** In a shared home, meeting room, or classroom:
- Multiple people issue commands simultaneously
- The assistant cannot tell *who* said what
- Everyone gets the same generic response — no personalization

**Specific pain points:**
- **Cocktail party problem:** Isolating one voice from a noisy mix of voices is an unsolved problem in consumer devices
- **No speaker identity resolution:** No current consumer assistant matches a voice stream to a specific enrolled user's profile in a multi-speaker scenario
- **No per-user response routing:** Even if two people speak, the assistant responds once — to no one in particular
- **Privacy gap:** Cloud-only LLMs send every command off-device, regardless of sensitivity

**Why we selected this:**
This problem sits at the intersection of signal processing, deep learning, NLP, and systems engineering. No complete, open-source, end-to-end multi-user pipeline exists. We wanted to build one.

---

## SLIDE 3 — Proposed Solution

### How We Are Solving It:

We built a **7-stage end-to-end AI pipeline** that takes a single mixed audio file containing multiple speakers and produces individual transcripts, identified names, intents, and personalized responses — one per speaker.

**The pipeline (all stages implemented and running):**

```
 Mixed Audio (multiple voices)
         │
         ▼
 ┌──────────────────┐
 │  1. VAD          │  → Detect speech vs silence (energy-based framing)
 └────────┬─────────┘
          ▼
 ┌──────────────────┐
 │  2. Separation   │  → Split mixed audio into N per-speaker WAV streams
 │  (SepFormer +    │     User picks model: WHAMR 16K or LibriMix
 │   IRM masking)   │     IRM applied to original mix (not noisy output)
 └────────┬─────────┘
          ▼
 ┌──────────────────┐
 │  3. Speaker ID   │  → Match each stream to an enrolled user
 │  (ECAPA-TDNN +   │     192-dim embeddings + cosine similarity
 │   Hungarian)     │     Hungarian algorithm: optimal 1-to-1 assignment
 └────────┬─────────┘
          ▼
 ┌──────────────────┐
 │  4. ASR          │  → faster-Whisper large-v3 (INT8, ~1.6 GB)
 └────────┬─────────┘     Transcript per speaker
          ▼
 ┌──────────────────┐
 │  5. Intent NLU   │  → Wake word detection + command + slot extraction
 └────────┬─────────┘     (device, action, schedule, reminder)
          ▼
 ┌──────────────────┐
 │  6. LLM Response │  → OpenVINO INT4 Llama 3.2 3B (Arc GPU)
 │  (tiered chain)  │     → CPU Transformers fallback
 └────────┬─────────┘     → Google Gemini API fallback
          ▼
 ┌──────────────────┐
 │  7. Web UI       │  → FastAPI backend + HTML/CSS/JS frontend
 └──────────────────┘     Per-speaker cards: name, audio, transcript,
                          intent, response. Live metrics dashboard.
```

**Delivered as:** A fully working local web application. Open browser → generate demo audio → process → see per-speaker results. Also supports real audio upload and live speaker enrollment.

---

## SLIDE 4 — Proposed Solution – Technical Details

### Diagram to Draw: Layered Architecture (draw.io / Mermaid recommended)

```
┌──────────────────────────────────────────────────────┐
│                  BROWSER (Web UI)                    │
│  Pills: speakers | overlap | noise | sep model       │
│  Buttons: Generate Demo | Process | Upload           │
│  Output: per-speaker cards + metrics strip           │
└──────────────────┬───────────────────────────────────┘
                   │ HTTP POST (multipart/form-data)
┌──────────────────▼───────────────────────────────────┐
│             FASTAPI BACKEND (Python)                 │
│  /process_demo  /process_audio  /speakers            │
│  /api/separation_models  /enroll  /chat              │
└──────────────────┬───────────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │   pipeline.py        │
        │   run_pipeline()     │
        └──────┬───────┬───────┘
               │       │
    ┌──────────▼─┐  ┌──▼────────────────────────────┐
    │ vad.py     │  │ separation.py                  │
    │ Energy VAD │  │ SepFormer WHAMR 16K (default)  │
    └────────────┘  │ SepFormer LibriMix (alternate) │
                    │ IRM post-processing on mix      │
                    └──────────────┬─────────────────┘
                                   │
                    ┌──────────────▼─────────────────┐
                    │ speaker_id.py                   │
                    │ ECAPA-TDNN → 192-d embedding    │
                    │ Cosine sim vs enrolled DB        │
                    │ Hungarian optimal assignment     │
                    └──────────────┬─────────────────┘
                                   │
               ┌───────────────────▼────────────────┐
               │ asr.py                              │
               │ faster-Whisper large-v3 INT8        │
               │ beam_size=5, VAD filter             │
               └──────────────┬──────────────────────┘
                              │
               ┌──────────────▼────────────────────┐
               │ intent.py                          │
               │ Wake word + regex slot extraction  │
               └──────────────┬─────────────────────┘
                              │
               ┌──────────────▼────────────────────┐
               │ llm_engine.py                      │
               │ 1. OpenVINO INT4 Llama 3.2 3B      │
               │    (Intel Arc XPU — fastest)        │
               │ 2. HuggingFace Transformers (CPU)   │
               │ 3. Gemini 1.5 Flash (cloud)         │
               └────────────────────────────────────┘
```

### Key Technical Decisions & Constraints

| Component | Technology | Key Detail |
|---|---|---|
| Separation (default) | SepFormer WHAMR 16K | Native 16 kHz — no upsampling artifacts |
| Separation (alternate) | SepFormer LibriMix | 8 kHz → 16 kHz upsample; 2000+ speaker vocab |
| Post-processing | IRM on original mix | Mask applied to untouched mix STFT → no double-distortion |
| Speaker embedding | ECAPA-TDNN, 192-d | VoxCeleb pretrained; cosine similarity |
| Speaker assignment | Hungarian algorithm | Guaranteed optimal 1-to-1, prevents duplicate IDs |
| ASR | faster-Whisper large-v3 | INT8 quantized, CTranslate2 backend |
| LLM | Llama 3.2 3B INT4 | OpenVINO runtime on Intel Arc GPU |
| Backend | FastAPI + Uvicorn | Async, hot-reload, auto model caching |
| Hardware target | Intel Arc GPU (XPU) | Falls back to CUDA → CPU gracefully |

**Honest Constraint:**
Current pipeline is **batch** (processes a full audio file). Real-time streaming (WebSocket + chunked inference) is the primary planned next step.

---

## SLIDE 5 — Novelty & Innovation of your solution

### Diagram to Draw: Comparison table + innovation callout boxes

#### Comparison vs State of the Art

| Feature | Amazon Alexa / Google Home | Academic (Whisper + Pyannote) | **Our System** |
|---|---|---|---|
| Simultaneous multi-speaker input | ❌ One at a time | ⚠️ Diarization only (no active separation) | ✅ Full neural separation |
| Per-speaker personalized LLM response | ❌ | ❌ | ✅ |
| Offline / on-device LLM | ❌ Cloud only | ❌ | ✅ OpenVINO INT4 on Arc GPU |
| IRM applied to original mix | ❌ | ❌ | ✅ No double-distortion |
| User-selectable separation model (UI) | ❌ | ❌ | ✅ |
| Hungarian optimal speaker assignment | ❌ | ❌ | ✅ No duplicate IDs |
| Fully open end-to-end pipeline | ❌ Proprietary | ⚠️ Partial | ✅ |

---

#### What Makes It Unique

**1. IRM applied to the original mix — not the separated stream**
Standard Wiener / spectral subtraction filters the already-noisy separated output, causing *double-distortion*. We compute `IRM_i = |S_i|² / Σ|S_j|²` and apply it to the **untouched original mixture spectrogram**. This is the theoretically optimal soft mask — no artifact compounding.

**2. Hungarian algorithm for speaker assignment**
Greedy matching (each stream matches its best candidate independently) frequently assigns the same identity to two streams when embeddings are similar. We use the **linear assignment algorithm** to solve a global cost matrix — guaranteeing a unique 1-to-1 mapping across all speakers at once.

**3. User-selectable separation model with download status badges**
The UI exposes the model registry live. Users switch between WHAMR 16K (reverb-robust, native 16 kHz) and LibriMix (large speaker vocabulary) with one click. Models are cached independently — switching costs nothing after the first download.

**4. Tiered offline-first LLM with Arc GPU acceleration**
Priority: Arc GPU (OpenVINO INT4 Llama 3.2 3B) → CPU Transformers → Gemini API. The system degrades gracefully without internet, without GPU, or without the large model — always producing some response.

**5. End-to-end open pipeline with web UI**
No comparable open-source project connects all 7 stages (VAD → Separation → Speaker ID → ASR → Intent → LLM → UI) into a single running application. Most papers address one stage.

#### Pain Points Addressed
- **Privacy:** Commands processed on-device — nothing sent to cloud unless opted in
- **Fairness:** Every person in the room is heard and answered individually
- **Accuracy:** IRM + Hungarian together reduce the two biggest failure modes: bleed-through audio and duplicate speaker labels
- **Accessibility:** Web UI — no app install, works on any device on the local network

---

## SLIDE 6 — Open Datasets planned to be used / published

### Diagram: Table with dataset, licence, usage

| Dataset | Licence | Used For | Status |
|---|---|---|---|
| **LibriMix** (Cosentino et al. 2020) | CC-BY 4.0 | SepFormer LibriMix training corpus; our evaluation benchmark | Used |
| **WHAMR!** (Maciejewski et al. 2020) | Open research | SepFormer WHAMR 16K training (reverberant + noisy mixtures) | Used |
| **LibriSpeech** (Panayotov et al. 2015) | CC-BY 4.0 | Base corpus for ECAPA-TDNN pretraining; ASR evaluation | Used |
| **VoxCeleb 1 & 2** (Nagrani 2017/2018) | CC-BY 4.0 | ECAPA-TDNN speaker verification pretraining (7000+ speakers) | Used |
| **WHAM!** (Wichern et al. 2019) | Open research | Noise backgrounds in WHAMR! split | Used |
| **Our Demo Eval Set** | To be released | 3-speaker TTS mixtures with ground-truth JSON annotations (text, speaker, timings, overlap type) | Publishing planned |

**Note:** We do not claim to have collected or trained on private/proprietary data. All pretraining datasets are publicly available under open licences.

---

## SLIDE 7 — Open Models planned to be used / developed / trained / fine-tuned

### Diagram: Table with model, HuggingFace ID, task, status

| Model | HuggingFace ID | Task in Pipeline | Status |
|---|---|---|---|
| **SepFormer WHAMR 16K** | `speechbrain/sepformer-whamr16k` | Speech Separation (primary) | ✅ Used as-is |
| **SepFormer LibriMix** | `speechbrain/sepformer-libri2mix` | Speech Separation (alternate) | ✅ Used as-is |
| **ECAPA-TDNN** | `speechbrain/spkrec-ecapa-voxceleb` | Speaker Embedding / Verification | ✅ Used as-is; enrolled on custom speakers |
| **faster-Whisper large-v3** | `Systran/faster-whisper-large-v3` | ASR / Transcription | ✅ Used as-is (INT8) |
| **Llama 3.2 3B Instruct (OpenVINO INT4)** | `CelesteImperia/Llama-3.2-3B-Instruct-OpenVINO-INT4` | On-device LLM response | ✅ Deployed on Arc GPU |
| **Gemini 1.5 Flash** | Google AI API | Cloud LLM fallback | ✅ Used via API |
| **Silero VAD** | `snakers4/silero-vad` | Neural Voice Activity Detection | 🔜 Planned (replaces energy VAD) |
| **Whisper (fine-tuned)** | Custom | Accented / noisy speech robustness | 🔜 Planned fine-tune |
| **ECAPA-TDNN (fine-tuned)** | Custom | Meeting-room acoustics adaptation | 🔜 Planned fine-tune |

---

## SLIDE 8 — AI/GenAI/Agentic tools used/developed

### Diagram: Process map showing AI tool at each stage + best practices sidebar

#### AI Tools Per Pipeline Stage

| Pipeline Stage | Tool / Model | Why This One |
|---|---|---|
| Voice Activity Detection | Energy-based VAD (custom) | Lightweight, no extra model load; Silero planned next |
| Speech Separation | SepFormer (SpeechBrain) | Best open pretrained transformer for 2-speaker separation |
| IRM Post-Processing | SciPy STFT / ISTFT | Applied to original mix — mathematically optimal masking |
| Speaker Embedding | ECAPA-TDNN (SpeechBrain) | State-of-the-art speaker verification, 192-d compact embedding |
| Speaker Assignment | `scipy.optimize.linear_sum_assignment` | Hungarian algorithm — global optimal, not greedy |
| Transcription | faster-Whisper large-v3 INT8 | 4× faster than standard Whisper; CTranslate2 backend |
| Intent / NLU | Rule-based regex + slot parser | Zero latency, no model load; LLM-based NLU planned next |
| Response Generation | Llama 3.2 3B (OpenVINO INT4) | On-device, Arc GPU accelerated, privacy-preserving |
| Cloud LLM Fallback | Google Gemini 1.5 Flash | High quality when offline LLM unavailable |
| Demo TTS Generation | gTTS (Google TTS) | Generates distinct speaker voices for pipeline testing |
| Web Framework | FastAPI + Uvicorn | Async, typed, hot-reload; ideal for ML inference server |

#### GenAI / Agentic Tool: Claude Code (Anthropic)

Used as a **coding agent** throughout the entire build:

| What Claude Code Did | Outcome |
|---|---|
| Designed the 7-stage pipeline architecture | Full pipeline implemented and running |
| Chose IRM-on-original-mix over Wiener filter | Provably better separation quality |
| Identified Hungarian algorithm as fix for duplicate speaker IDs | Duplicate assignment eliminated |
| Selected `sepformer-whamr16k` over `libri2mix` based on native SR | No upsampling artifacts |
| Diagnosed numpy MINGW-W64 STATUS_ACCESS_VIOLATION crash | Prevented 2+ hours of debugging |
| Resolved OpenVINO INT4 conversion OOM (2.6 GB free RAM) | Switched to pre-converted HuggingFace checkpoint |
| Built multi-model UI (registry, API, frontend pills, badges) | User-selectable model live in browser |

#### Best Practices Uncovered

1. **Model caching by key** — `_sep_model_cache` dict: multiple models loaded simultaneously, no reload cost
2. **Subprocess isolation** — potentially crashing native code (Intel IPEX probe, numpy) runs in a child process; main server never crashes
3. **IRM on original mix** — never mask already-processed audio; always return to the clean original spectrogram
4. **Tiered fallback chain** — hardware-native → offline CPU → cloud: always a response, even in degraded environments
5. **Offline-first model management** — all models download once to `models/`; subsequent runs are fully air-gapped

---

## SLIDE 9 — [Optional] Additional Supporting Materials

### Sub-slide A: System Demo (add screenshots here)

Screenshots to include from the running app:
1. **Pipeline Config Card** — pill selectors for speakers / overlap / noise / sep model (WHAMR 16K vs LibriMix) / oracle vs real
2. **Demo Audio Card** — mixed audio player + ground-truth speaker chips
3. **Results Cards** — two speaker cards each with: identified name, confidence %, audio player, transcript, intent badge, LLM response
4. **Metrics Strip** — SI-SNR (dB), WER (%), xRT, Sep Model label, Process Time
5. **Speaker Enrollment Page** — enroll form + enrolled speaker list

---

### Sub-slide B: Metrics on Demo Set

*(Measured on synthetic TTS demo: 3 speakers, partial overlap, no added noise)*

| Metric | Oracle Mode | WHAMR 16K + IRM | LibriMix + IRM |
|---|---|---|---|
| WER | ~0% (TTS text known) | ~8–15% | ~12–20% |
| SI-SNR improvement | — | +6 to +11 dB | +4 to +9 dB |
| Speaker ID accuracy | 100% (ground truth used) | ~85–90% | ~80–88% |
| End-to-end time (10 s audio, CPU) | <1 s | 15–30 s | 15–30 s |
| End-to-end time (10 s, Arc GPU LLM) | <1 s | 5–10 s | 5–10 s |

---

### Sub-slide C: What Is Implemented vs What Is Planned

#### ✅ Already Implemented (Working Right Now)

| Feature | Detail |
|---|---|
| 7-stage pipeline | VAD → Sep → SpeakerID → ASR → Intent → LLM → UI |
| Two separation models (user picks) | WHAMR 16K (16 kHz native) + LibriMix (8 kHz) |
| IRM post-processing on original mix | Applied to untouched mix spectrogram — optimal masking |
| ECAPA-TDNN speaker embeddings | 192-d, cosine similarity matching |
| Hungarian algorithm assignment | Global optimal 1-to-1 across all streams |
| faster-Whisper large-v3 ASR | INT8, beam 5, VAD filter |
| Llama 3.2 3B INT4 on Arc GPU | OpenVINO runtime, fully offline |
| Tiered LLM fallback (Arc → CPU → Gemini) | Graceful degradation |
| Real speaker enrollment (web UI) | Upload voice samples → enroll identity |
| Demo audio generation | TTS + mixer: configurable overlap, noise type/level |
| Metrics dashboard | WER, SI-SNR, xRT, sep model, processing time |
| Model download status badges (UI) | Live check: downloaded / not downloaded |

#### 🔜 Plausible Next Steps (Realistic, Not Wishful)

| Feature | Effort | Why Plausible |
|---|---|---|
| **Silero VAD** (replace energy VAD) | Low | Drop-in neural VAD, HuggingFace, same interface |
| **Real-time WebSocket streaming** | Medium | FastAPI WebSocket + chunked SepFormer inference |
| **Rule → LLM-based intent NLU** | Low | Already have LLM loaded — just change the prompt |
| **3-speaker LibriMix support** | Low | `sepformer-libri3mix` model exists on HuggingFace |
| **Noise reduction pre-stage** | Low | `speechbrain/mtl-mimic-voicebank` or `noisereduce` lib |
| **Whisper fine-tune on accented speech** | Medium | Small dataset, HuggingFace Trainer, 1–2 GPU-hours |
| **ECAPA fine-tune for room acoustics** | Medium | SpeechBrain `EncoderClassifier.from_hparams` training recipe |
| **MossFormer2 SOTA separation** | Medium | Best open model; currently blocked by Windows numpy crash — solvable on Linux / Docker |
| **Mobile-friendly UI** | Low | CSS media queries — layout already responsive |

---

### Sub-slide D: Future Roadmap (1-slide visual)

*(Draw as a timeline or Gantt-style bar chart)*

```
Phase 1 (Done)       → End-to-end batch pipeline, web UI, dual-model separation
Phase 2 (Next)       → Silero VAD, WebSocket streaming, LLM-based NLU
Phase 3 (Planned)    → Fine-tuned Whisper, ECAPA room acoustics, 3-speaker UI
Phase 4 (Research)   → MossFormer2 SOTA, on-device diarization, edge deployment
```
