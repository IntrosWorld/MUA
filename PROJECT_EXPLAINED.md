# Multi-User Smart Assistant — Complete Project Explanation
### Written so that even a beginner can understand every part

---

# TABLE OF CONTENTS

1. [What Is This Project?](#1-what-is-this-project)
2. [Why Does This Problem Exist?](#2-why-does-this-problem-exist)
3. [How Do We Solve It — The Big Picture](#3-how-do-we-solve-it--the-big-picture)
4. [The Pipeline — Stage by Stage](#4-the-pipeline--stage-by-stage)
   - [Stage 1: Voice Activity Detection (VAD)](#stage-1-voice-activity-detection-vad)
   - [Stage 2: Speech Separation](#stage-2-speech-separation)
   - [Stage 3: Speaker Identification](#stage-3-speaker-identification)
   - [Stage 4: Automatic Speech Recognition (ASR)](#stage-4-automatic-speech-recognition-asr)
   - [Stage 5: Intent Detection](#stage-5-intent-detection)
   - [Stage 6: Response Generation (LLM)](#stage-6-response-generation-llm)
   - [Stage 7: Web UI](#stage-7-web-ui)
5. [The Three Big Technical Innovations](#5-the-three-big-technical-innovations)
6. [The Tech Stack — Every Tool Explained](#6-the-tech-stack--every-tool-explained)
7. [Project File Structure](#7-project-file-structure)
8. [How to Run the Project](#8-how-to-run-the-project)
9. [What We Built vs What Exists](#9-what-we-built-vs-what-exists)
10. [Honest Limitations & What Comes Next](#10-honest-limitations--what-comes-next)

---

# 1. What Is This Project?

Imagine you are at home. You say **"Hey assistant, turn on the bedroom light"** and at the same moment your roommate says **"Hey assistant, play my study playlist."**

A normal voice assistant — Alexa, Google Assistant, Siri — will hear a mess of two voices, get confused, and probably respond to neither (or get one wrong).

**This project builds a voice assistant that can handle both of you at the same time.**

It does five things that no consumer assistant currently does together:
1. **Separates** the two voices from each other (like removing two colors mixed in paint)
2. **Identifies** whose voice is whose (matches to enrolled user profiles)
3. **Transcribes** what each person said (speech to text)
4. **Understands** what each person wants (intent + command parsing)
5. **Responds** individually to each person with a personalized reply

The output is not one answer to the room. It is **two separate, personalized answers** — one for each speaker.

---

# 2. Why Does This Problem Exist?

## 2.1 The "Cocktail Party Problem"

Scientists have a name for this challenge: the **cocktail party problem**.

Picture a noisy party. There are 20 conversations happening at once. Yet somehow, your brain can focus on just one person and filter out everyone else. How? Your brain does incredibly complex signal processing — in real time, without you thinking about it.

Microphones do not have this ability. A microphone records everything and mixes it all together into one signal. Getting individual voices back out of that mix is a hard mathematical and machine-learning problem.

## 2.2 Why Current Assistants Fail

Current voice assistants are built on one core assumption: **one person is speaking at a time.**

The entire pipeline — from the microphone hardware to the cloud server to the response — is designed for a single input voice. When two people speak simultaneously:

- The automatic speech recognition gets confused and transcribes a garbled mix
- There is no mechanism to figure out who said what
- Even if it catches one phrase, it has no idea which user account to respond to
- It gives one generic response that may be wrong for both users

## 2.3 Why This Matters in the Real World

- **Smart homes:** Family members issue commands at the same time
- **Classrooms:** Students ask questions while the teacher is giving instructions
- **Meeting rooms:** Multiple participants speak in overlapping bursts
- **Elderly care:** Multiple residents in a care home share one assistant device
- **Accessibility:** Users with motor impairments rely on voice — they should not have to wait in turn

---

# 3. How Do We Solve It — The Big Picture

We solve it as a **pipeline** — a sequence of processing stages, one after another, each one transforming the audio into something more useful.

Think of it like a factory assembly line:

```
Raw mixed audio goes in one end
    ↓
Each stage refines it further
    ↓
Personalized text responses come out the other end
```

The pipeline has 7 stages:

```
[AUDIO IN: 2 voices mixed together]
        │
        ▼
  ┌──────────┐
  │  Stage 1 │  VAD — Is anyone even speaking right now?
  └────┬─────┘
       ▼
  ┌──────────┐
  │  Stage 2 │  SEPARATION — Split the mixed audio into 2 separate voice tracks
  └────┬─────┘
       ▼
  ┌──────────┐
  │  Stage 3 │  SPEAKER ID — Who is Track 1? Who is Track 2?
  └────┬─────┘
       ▼
  ┌──────────┐
  │  Stage 4 │  ASR — What did each person say? (speech → text)
  └────┬─────┘
       ▼
  ┌──────────┐
  │  Stage 5 │  INTENT — What do they want? (text → command)
  └────┬─────┘
       ▼
  ┌──────────┐
  │  Stage 6 │  LLM — Generate a personalized reply for each person
  └────┬─────┘
       ▼
  ┌──────────┐
  │  Stage 7 │  WEB UI — Show results in the browser
  └──────────┘
        │
        ▼
[OUTPUT: 2 separate responses, one per speaker, with names]
```

---

# 4. The Pipeline — Stage by Stage

## Stage 1: Voice Activity Detection (VAD)

### What it does
Before doing any heavy processing, we first check: **is anyone even speaking in this audio?**

VAD stands for Voice Activity Detection. It scans the audio and marks which parts contain speech and which parts are silence or background noise.

### Why it matters
Processing silence is wasteful. If you have a 30-second recording where only 10 seconds contain speech, you want to process just those 10 seconds. This saves time and prevents the separation model from trying to "separate silence" which produces garbage.

### How it works (our implementation)
We use an **energy-based method**:

```
1. Split the audio into small chunks (30 milliseconds each)
2. Calculate the energy (loudness) of each chunk
3. Chunks above a threshold → speech
4. Chunks below → silence
5. Return the timestamps of speech segments
```

Think of it as a volume meter. If the audio is loud enough, someone is probably speaking.

```python
# src/vad.py (simplified)
def _energy_vad(audio, frame_ms=30, threshold_ratio=0.05):
    frame_size = int(sample_rate * frame_ms / 1000)
    rms_values = [np.sqrt(np.mean(chunk**2)) for chunk in frames]
    threshold = max(rms_values) * threshold_ratio
    # frames above threshold = speech
```

**File:** `src/vad.py`

---

## Stage 2: Speech Separation

This is the hardest and most important stage. This is where we take the mixed audio (two voices blended together) and try to pull them apart into two clean individual tracks.

### The core problem visualized

Imagine two audio signals as waves:

```
Speaker 1:  ▁▃▅▇▅▃▁▃▅▇   (wave pattern of voice 1)
Speaker 2:  ▂▄▆▄▂▄▆▄▂▄   (wave pattern of voice 2)

When mixed:
Mixed:      ▃▇█▇▄▇█▇▃▇   (the waves add up — now inseparable by simple math)
```

Once mixed, you cannot just subtract one wave from the other because you do not know what either wave looked like before mixing. You have to **learn** to separate them.

### What is SepFormer?

SepFormer is a **deep neural network** specifically trained to separate mixed speech. It was created by the SpeechBrain team and is considered one of the best open models for this task.

It was trained on **millions of pairs** of:
- Input: two voices mixed together (with various levels of overlap, noise, and room reverb)
- Output: the two original clean voices separately

After training on so many examples, the model learned patterns that distinguish one voice from another — even when they overlap.

### Which models we support (and why user can choose)

We built a **model registry** — a menu of separation models. The user picks from the web UI:

**Option A — SepFormer WHAMR 16K (default, recommended)**
- Trained on WHAMR! dataset: realistic room acoustics, noise, reverb
- Works at **16,000 Hz (16 kHz)** — this is important
- 16 kHz is the standard quality for voice AI. Since this model works natively at 16 kHz, there is no quality loss from resampling
- Better for real-world audio with background noise or echo

**Option B — SepFormer LibriMix**
- Trained on LibriMix dataset: clean speech from 2000+ LibriSpeech speakers
- Works at **8,000 Hz** internally → we upsample to 16 kHz afterwards
- Upsampling adds a small quality penalty (you are "guessing" the high frequencies)
- Better vocabulary of speaker voice types due to 2000+ speaker training

```python
# src/separation.py — model registry
SEPARATION_MODELS = {
    "sepformer_whamr16k": {
        "repo_id": "speechbrain/sepformer-whamr16k",
        "native_sr": 16000,   # no resampling needed
        "label": "SepFormer WHAMR 16K",
    },
    "sepformer_libri2mix": {
        "repo_id": "speechbrain/sepformer-libri2mix",
        "native_sr": 8000,    # will upsample to 16k after
        "label": "SepFormer LibriMix",
    },
}
```

### Post-processing: IRM (Ideal Ratio Mask) — our key innovation

Even the best separation model is not perfect. Some "bleed" remains — a little bit of Speaker 2's voice leaks into Speaker 1's track and vice versa.

To clean this up, we apply a post-processing step called the **Ideal Ratio Mask (IRM)**.

**The basic idea:**
- For every tiny slice of time and every frequency (think of it like a grid of pixels in a spectrogram image)
- We look at how loud Speaker 1 is vs Speaker 2 in that slice
- We create a "mask" value between 0 and 1 for each grid cell
- Where Speaker 1 dominates → mask = close to 1 (keep this)
- Where Speaker 2 dominates → mask = close to 0 (throw this away)

**The critical insight that makes ours better:**
Most systems apply this mask to the already-separated (and therefore already slightly distorted) audio. This causes **double-distortion** — you are adding a second round of errors on top of the first.

We apply the mask to the **original mixed audio** instead:

```
Standard approach (bad):
   Mixed Audio → SepFormer → Noisy Stream 1 → Mask → Cleaned Stream 1
                                             ↑ mask applied here
                                             (already has separation errors)

Our approach (better):
   Mixed Audio → SepFormer → Noisy Stream 1 ─┐
                                              ├─ Compute mask from streams
   Mixed Audio ──────────────────────────────→ Apply mask to ORIGINAL mix → Cleaned Stream 1
                                             ↑ mask applied to untouched audio
```

The original mixed audio has no distortion. Applying the mask to it gives the cleanest possible result.

```python
# src/separation.py — IRM implementation (simplified)
def _irm_refine(streams, mixed_audio, sr):
    # Get spectrogram of original mix
    S_mix = stft(mixed_audio)

    # Get spectrogram of each separated stream
    stream_stfts = [stft(s) for s in streams]

    # For each speaker, compute mask
    total_power = sum(|S|^2 for S in stream_stfts)

    for each speaker i:
        mask = |S_i|^2 / total_power   # how dominant is speaker i here?
        S_clean = mask * S_mix          # apply to original mix
        audio_clean = istft(S_clean)    # convert back to audio
```

**File:** `src/separation.py`

---

## Stage 3: Speaker Identification

After separation, we have two audio tracks. But we do not know which is which. This stage answers: **who is speaking in each track?**

### The concept: speaker embeddings

Every person's voice has a unique "fingerprint" — made up of the pitch, tone, rhythm, and shape of how they speak. We can mathematically capture this fingerprint as a list of numbers called an **embedding**.

Think of it like a GPS coordinate for a voice. Every speaker has their own coordinate in a 192-dimensional space (192 numbers that describe their voice). When we hear an unknown voice, we compute its coordinates and find the closest known speaker.

### The model: ECAPA-TDNN

We use a neural network called **ECAPA-TDNN** (Enhanced Channels Attention, Propagation and Aggregation TDNN). It was trained on VoxCeleb — a dataset of 7000+ real celebrity voices — to learn what makes each voice unique.

When given a voice clip, ECAPA-TDNN outputs a vector of 192 numbers — that voice's embedding.

```python
# src/speaker_id.py — simplified
embedding = ecapa_model(voice_clip)  # → 192 numbers
```

### Enrollment

Before the system can recognize a speaker, you must **enroll** them — give the system a few voice samples with their name so it can learn their "coordinates."

```
Enrollment:
  Upload 3 voice clips of "Mohit" speaking
  → Model computes 3 embeddings
  → Average them → 1 reference embedding
  → Save to database: {"mohit": [0.23, -0.41, 0.87, ...]}  (192 numbers)
```

### Recognition (cosine similarity)

When we get a new separated stream, we:
1. Compute its embedding (192 numbers)
2. Compare it to every enrolled speaker's embedding using **cosine similarity** — a measurement of how similar two vectors are (1.0 = identical, 0.0 = completely different)
3. The enrolled speaker with the highest similarity wins

```
New stream embedding:    [0.21, -0.39, 0.85, ...]
Mohit's stored embedding:[0.23, -0.41, 0.87, ...]
Rahul's stored embedding:[0.10,  0.55, -0.20, ...]

Similarity to Mohit: 0.92  ← high match
Similarity to Rahul: 0.11  ← low match
→ This stream is probably Mohit
```

### The Hungarian Algorithm — solving a subtle problem

Here is a problem most systems get wrong:

If there are 2 speakers and we identify them independently (one at a time), we might end up saying:
- Stream 1 → "Mohit" (similarity 0.85)
- Stream 2 → "Mohit" (similarity 0.78)

Both streams match Mohit best! Now we have two Mohits and no Rahul — clearly wrong.

The fix is to solve this as a **global assignment problem**: find the assignment of streams to speakers that maximizes total similarity across all pairs, while ensuring each speaker is assigned to exactly one stream.

This is solved by the **Hungarian algorithm** (also called the linear assignment algorithm). It considers all possible pairings simultaneously and finds the globally optimal matching.

```python
# src/speaker_id.py — using scipy's Hungarian implementation
from scipy.optimize import linear_sum_assignment

cost_matrix = build_similarity_matrix(streams, enrolled_speakers)
row_indices, col_indices = linear_sum_assignment(-cost_matrix)  # negate = maximize
# result: optimal 1-to-1 assignment, no duplicates
```

**File:** `src/speaker_id.py`

---

## Stage 4: Automatic Speech Recognition (ASR)

Now we know who each track belongs to. Next: **what did they say?**

ASR (Automatic Speech Recognition) is the task of converting audio to text. This is speech-to-text.

### The model: faster-Whisper large-v3

We use **Whisper**, OpenAI's state-of-the-art speech recognition model. Specifically the `large-v3` version — the most accurate one.

However, the original Whisper is slow. We use **faster-Whisper**, a reimplementation using a library called CTranslate2 that runs inference 4× faster with the same accuracy.

We also use **INT8 quantization** — a technique that compresses the model's internal numbers from 32-bit floats to 8-bit integers. The model goes from ~6 GB to ~1.6 GB and runs faster, with only a tiny accuracy drop.

```python
# src/asr.py — simplified
from faster_whisper import WhisperModel

model = WhisperModel("large-v3",
                     device="cpu",
                     compute_type="int8")  # INT8 quantization

segments, info = model.transcribe(audio_path,
                                  beam_size=5,
                                  vad_filter=True)

transcript = " ".join(seg.text for seg in segments)
```

The `beam_size=5` means the model considers 5 possible transcriptions at once and picks the best — more accurate than greedy (beam=1) at a small speed cost.

**File:** `src/asr.py`

---

## Stage 5: Intent Detection

Now we have text — for example: `"Hey assistant, turn on my bedroom light."`

This stage figures out **what the person wants**. This is called NLU (Natural Language Understanding) or intent detection.

### What we extract

1. **Wake word** — was the trigger word "hey assistant" present? If not, we treat it as background conversation
2. **Intent** — what category of command is this?
   - `light_on`, `light_off`, `play_music`, `set_alarm`, `remind_me`, `check_schedule`, etc.
3. **Slots** — the specific details
   - Which device? ("bedroom light", "fan", "AC")
   - What time? ("7 AM")
   - What content? ("study playlist")

### How we implement it

We use **rule-based matching** — a set of if/then rules with keyword matching:

```python
# src/intent.py — simplified
INTENT_RULES = [
    ("light_on",   r"turn on.*light"),
    ("light_off",  r"turn off.*light"),
    ("play_music", r"play.*music|play.*playlist"),
    ("set_alarm",  r"set.*alarm|alarm.*for"),
    ("remind_me",  r"remind me"),
]

def detect_intent(text):
    if not has_wake_word(text):        # "hey assistant" check
        return {"intent": "no_command"}

    for intent_name, pattern in INTENT_RULES:
        if re.search(pattern, text.lower()):
            return {"intent": intent_name, "slots": extract_slots(text)}
```

**Why rule-based?** It is instant (zero latency), needs no model loading, and works perfectly for a defined set of commands. For open-ended questions, we pass directly to the LLM in Stage 6.

**File:** `src/intent.py`

---

## Stage 6: Response Generation (LLM)

This is where the assistant actually "thinks" and generates a natural language response.

**LLM** = Large Language Model. Think ChatGPT — a model that has read enormous amounts of text and learned to generate human-like language.

### What we give the LLM

We construct a **prompt** that includes:
- The speaker's name
- What they said (transcript)
- What they want (intent + slots)
- Their enrolled profile (preferences, home setup)

The LLM then generates a natural, personalized reply.

### Our three-tier LLM system

We built a fallback chain — the system tries the best option first, falls back if unavailable:

```
Tier 1: OpenVINO INT4 Llama 3.2 3B on Intel Arc GPU
         └── Fully offline. Private. Fast on Arc GPU hardware.
             ~1.77 GB model file.
                │
                ↓ (if Arc GPU not available or model not loaded)

Tier 2: HuggingFace Transformers CPU
         └── Same model, runs on CPU. Slower but still offline.
             Works on any computer.
                │
                ↓ (if offline LLM fails)

Tier 3: Google Gemini 1.5 Flash API
         └── Cloud. Requires internet + API key.
             Highest quality responses. Used as last resort.
```

### Why Llama 3.2 3B with OpenVINO INT4?

**Llama 3.2 3B** is Meta's open-source language model with 3 billion parameters. It is small enough to run on consumer hardware but smart enough to generate coherent responses.

**INT4 quantization** compresses the model further — from 32-bit floats to just 4 bits per weight. The model shrinks from ~12 GB to ~1.77 GB. Quality loss is minimal for short-form responses.

**OpenVINO** is Intel's runtime for running AI models efficiently on Intel hardware — CPUs, integrated graphics, and Intel Arc GPUs (the discrete GPUs Intel makes). Our project uses OpenVINO to run Llama on the Intel Arc GPU at 5–20× the speed of plain CPU execution.

```python
# src/llm_engine.py — simplified
from optimum.intel.openvino import OVModelForCausalLM

model = OVModelForCausalLM.from_pretrained("models/llama-3.2-3b-ov",
                                            device="GPU")  # Intel Arc
response = model.generate(prompt_tokens, max_new_tokens=200)
```

**File:** `src/llm_engine.py`

---

## Stage 7: Web UI

The entire pipeline is wrapped in a web application so you can use it without touching code.

### Backend: FastAPI

**FastAPI** is a Python web framework. It handles HTTP requests from the browser and runs the pipeline.

Key endpoints:
- `POST /generate_demo` — generate a test audio file with multiple TTS speakers mixed
- `POST /process_demo` — run the full pipeline on the demo audio
- `POST /process_audio` — run the pipeline on a user-uploaded audio file
- `GET /api/separation_models` — return which models are downloaded
- `POST /enroll_speaker` — enroll a new speaker with uploaded voice samples
- `POST /chat` — standalone LLM chat

### Frontend: HTML + CSS + JavaScript

A single-page web app with:

**Config card:** pill-style buttons to choose —
- Number of speakers (2 or 3)
- Overlap type (full / partial / sequential)
- Noise level (none / low / medium / high)
- Noise type (white / fan)
- Separation mode (real model vs oracle cheat)
- **Separation model (WHAMR 16K or LibriMix)** ← added by us

**Results card:** after processing, each speaker gets their own card showing —
- Identified name + confidence %
- Audio player (listen to their separated stream)
- Transcript (what they said)
- Intent badge (what command was detected)
- Response (the LLM's personalized reply)

**Metrics strip:** a dashboard row showing —
- Audio duration
- Processing time
- xRT (how many times faster/slower than real-time)
- WER (Word Error Rate — how accurate was the transcription)
- SI-SNR (Signal-to-Noise Ratio improvement — how clean is the separation)
- Which separation model was used

**Files:** `frontend/index.html`, `frontend/style.css`, `frontend/script.js`

---

# 5. The Three Big Technical Innovations

## Innovation 1: IRM on the Original Mix (not the separated stream)

**The problem with standard post-processing:**
Every separation model produces imperfect output. The standard fix is to apply a mask to the separated output. But that separated output already has some distortion (errors introduced by the model). Masking distorted audio introduces a second round of errors — "double-distortion."

**Our fix:**
We always mask the **original mixed audio** — the untouched recording with no model-introduced artifacts. The mask values come from the separated streams (telling us who dominates where), but the audio we actually output comes from the original.

Mathematically, this is the **Ideal Ratio Mask** — it has been proven to be the optimal soft mask when you have estimates of each speaker's signal.

---

## Innovation 2: Hungarian Algorithm for Speaker Assignment

**The problem with greedy matching:**
If you identify speakers one stream at a time (greedy), you can end up assigning the same identity to two different streams. With 3 streams and 2 of them sounding similar to "Mohit," greedy matching assigns Mohit to both.

**Our fix:**
We build a **cost matrix** — a table where each cell is the similarity between stream X and speaker Y. Then we use the **Hungarian algorithm** to find the assignment that maximizes total similarity across all pairs while guaranteeing that no speaker is assigned twice.

This is a classical combinatorial optimization problem solved in O(n³) time — fast enough to not even register in processing time.

---

## Innovation 3: User-Selectable Separation Model with Live Registry

**What most systems do:**
Hard-code one model. If you want a different model, you edit the code and restart.

**What we built:**
A model registry — a Python dictionary that defines all available models (ID, HuggingFace repo, native sample rate, description). The user picks from the web UI. The system caches each model independently in memory — switching costs nothing after first load. The UI fetches download status live from the API and shows ✓ Downloaded badges on each option.

This makes adding a new separation model a **configuration change**, not a code change.

---

# 6. The Tech Stack — Every Tool Explained

## Python (the programming language)
All backend logic is written in Python 3.10+. Python is the dominant language for AI/ML because nearly every AI library is built for it.

## SpeechBrain
An open-source speech AI toolkit from the University of Montreal. We use it for two things:
- **SepFormer** — the separation neural network
- **ECAPA-TDNN** — the speaker embedding model

SpeechBrain makes loading these pretrained models extremely simple:
```python
model = SepformerSeparation.from_hparams(source="speechbrain/sepformer-whamr16k")
```

## HuggingFace Hub
A platform hosting thousands of pretrained AI models. When our code calls `snapshot_download(repo_id="speechbrain/sepformer-whamr16k")`, it automatically downloads all model files to our local `models/` folder. This happens once — subsequent runs use the cached files.

## faster-Whisper
A fast implementation of OpenAI's Whisper ASR model. Uses CTranslate2 (a C++ inference engine) to run 4× faster than the original. We use INT8 quantization to reduce memory from ~6 GB to ~1.6 GB.

## Intel OpenVINO
Intel's AI runtime toolkit. Converts and runs neural networks efficiently on Intel hardware — including their Arc discrete GPUs. We run Llama 3.2 3B in INT4 format through OpenVINO for fast, private, on-device LLM responses.

## SciPy
A scientific computing library. We use `scipy.signal.stft` and `scipy.signal.istft` for computing spectrograms (STFT = Short-Time Fourier Transform — converts audio to a time-frequency image). This is the mathematical foundation of our IRM post-processing.

## NumPy
The fundamental array computing library for Python. All audio data is stored and manipulated as NumPy arrays of floating-point numbers.

## SoundFile
A library for reading and writing WAV audio files. Every separated stream is saved as a WAV file.

## FastAPI
A modern Python web framework for building APIs. Much faster than Flask because it is asynchronous. We use it for all backend endpoints.

## Uvicorn
The server that runs FastAPI. Supports hot-reload — when you change a Python file during development, the server automatically restarts.

## gTTS (Google Text-to-Speech)
For the demo mode, we generate fake "speaker" voices using gTTS — it takes a text string and produces a WAV file of a voice reading it. We use different voices/speeds per speaker to simulate distinct people.

## Google Gemini API
Google's cloud LLM API. Used as the last-resort fallback when the local LLM is unavailable. Requires a free API key and internet connection.

## scipy.optimize.linear_sum_assignment
The Hungarian algorithm implementation in SciPy. One function call — does the entire global speaker assignment in milliseconds.

---

# 7. Project File Structure

```
multi_user_smart_assistant/
│
├── app.py                    ← Web server (FastAPI). Entry point.
│                               Handles all HTTP routes.
│
├── src/                      ← All pipeline logic lives here
│   ├── vad.py                ← Stage 1: Voice Activity Detection
│   ├── separation.py         ← Stage 2: Speech Separation (SepFormer + IRM)
│   ├── speaker_id.py         ← Stage 3: Speaker Identification (ECAPA-TDNN + Hungarian)
│   ├── asr.py                ← Stage 4: Automatic Speech Recognition (faster-Whisper)
│   ├── intent.py             ← Stage 5: Intent Detection (rule-based NLU)
│   ├── response_engine.py    ← Stage 6: Decides whether to use LLM or rule response
│   ├── llm_engine.py         ← Stage 6: LLM (OpenVINO → CPU → Gemini chain)
│   ├── pipeline.py           ← Orchestrates all 7 stages in sequence
│   ├── audio_mixer.py        ← Demo tool: mix multiple voices into one file
│   ├── tts_generate.py       ← Demo tool: generate TTS voice files
│   ├── metrics.py            ← Computes WER, SI-SNR, xRT after pipeline
│   ├── user_profiles.py      ← User profile storage and retrieval
│   └── utils.py              ← Shared helpers (file loading, duration, etc.)
│
├── frontend/                 ← Browser-side code
│   ├── index.html            ← Main web page
│   ├── style.css             ← All styling
│   ├── script.js             ← All browser logic (fetch API calls, render results)
│   ├── profiles.html         ← Speaker profiles page
│   ├── profiles.css
│   └── profiles.js
│
├── models/                   ← Downloaded AI models (auto-created, git-ignored)
│   ├── sepformer-whamr16k/   ← SepFormer WHAMR 16K weights
│   ├── sepformer-libri2mix/  ← SepFormer LibriMix weights
│   ├── whisper/              ← faster-Whisper large-v3 weights
│   ├── speaker_encoder/      ← ECAPA-TDNN weights
│   └── llama-3.2-3b-ov/     ← Llama 3.2 3B INT4 OpenVINO weights
│
├── data/
│   ├── mixed/                ← Generated demo mixed audio files
│   ├── generated/            ← TTS output files per speaker
│   ├── enrolled_speakers/    ← Voice samples for enrolled real users
│   └── profiles/             ← JSON profile files per user
│
├── outputs/
│   ├── separated/            ← Per-speaker WAV files after separation
│   └── reports/              ← Pipeline result JSON reports
│
├── mossformer2_run.py        ← Standalone script for MossFormer2 (blocked on Windows)
├── sep_test.py               ← Quick test script for separation pipeline
├── PPT_CONTENT.md            ← Slide content for presentation
├── PROJECT_EXPLAINED.md      ← This document
├── requirements.txt          ← Python package dependencies
└── .gitignore                ← Files not tracked by git
```

---

# 8. How to Run the Project

## Prerequisites
- Python 3.10 or higher
- ~8 GB free disk space (for all models)
- At least 4 GB RAM (8 GB recommended)
- Intel Arc GPU optional — everything works on CPU, just slower

## Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

## Step 2: Start the server
```bash
python app.py
```

On first start:
- The server starts on `http://localhost:8000`
- Models download automatically on first use (can take a few minutes each)

## Step 3: Open the browser
Go to: `http://localhost:8000`

## Step 4: Try the demo

**Option A — Use the built-in demo**
1. Click **"Generate Demo Audio"** — this creates a fake 3-speaker conversation using TTS
2. Click **"Process Demo Audio"** — runs the full pipeline
3. Wait ~15–30 seconds (CPU) or ~5–10 seconds (Arc GPU)
4. See the results cards — one per detected speaker

**Option B — Upload your own audio**
1. Record yourself and a friend speaking (can overlap)
2. Use the Upload section to submit the file
3. Make sure both voices are enrolled first (Speaker Management section)

## What happens during first run

The first time you run each stage, the model downloads automatically:

| Model | Size | When it downloads |
|---|---|---|
| SepFormer WHAMR 16K | ~300 MB | First time you process audio (real mode) |
| ECAPA-TDNN | ~100 MB | First time any audio is processed |
| faster-Whisper large-v3 | ~1.6 GB | First time ASR runs |
| Llama 3.2 3B OpenVINO INT4 | ~1.77 GB | First time chat or LLM response is requested |

---

# 9. What We Built vs What Exists

## What existed before (state of the art)

| System | What it does | What it cannot do |
|---|---|---|
| **Amazon Alexa** | One voice, one command, cloud LLM, smart home | Multiple speakers, offline, separation |
| **Google Assistant** | Same as Alexa | Same |
| **OpenAI Whisper** | Transcription of a single speaker | Separation, speaker ID, response |
| **Pyannote.audio** | Diarization — "who spoke when" (labels only) | Does not separate or transcribe |
| **SpeechBrain SepFormer** | Separates two voices from a mix | Just audio separation — no ID, ASR, or response |
| **Research papers** | Address one stage at a time in isolation | No complete working system, no UI |

## What we built

**A complete working system that chains all stages into one application with a web UI.**

No open-source project connects: separation + speaker identification + ASR + intent detection + LLM response generation + web interface — in a single runnable application.

We also added three things not present in any existing system:
1. IRM applied to the original mix (not standard)
2. Hungarian assignment across all speakers (not standard in speech pipelines)
3. User-selectable separation model with live download status in the UI

---

# 10. Honest Limitations & What Comes Next

## Current Limitations (being honest)

**1. Batch processing only — not real-time**
The pipeline processes a complete audio file. You cannot speak into the microphone and get live responses. This requires WebSocket streaming and chunked model inference — planned next.

**2. 2–3 speakers only**
SepFormer is designed for 2-speaker separation (our primary model). The LibriMix variant can handle 3. Beyond 3, no pretrained model in our stack handles this cleanly.

**3. Separation is imperfect on very overlapping audio**
When both speakers talk at the exact same time (full overlap) for extended periods, current separation models struggle — bleed-through increases. Partial overlap works much better.

**4. Speaker ID requires enrollment first**
A speaker must be enrolled (submit voice samples) before they can be identified. A complete stranger walking into the room gets labeled "Unknown."

**5. Windows-specific limitation: MossFormer2 crashes**
MossFormer2 (the current SOTA separation model from Alibaba Research) crashes on Windows with a memory access violation in the numpy C extension. It works on Linux. Since development was on Windows, we use SepFormer instead.

**6. Intent detection is rule-based**
The current intent system uses keyword matching. It does not handle paraphrasing well. "Switch on the lights" works. "Could you maybe brighten things up in here?" does not. LLM-based NLU is the planned replacement.

## Realistic Next Steps (what we would implement if we had more time)

**1. Silero VAD** (1 day of work)
Replace energy-based VAD with Silero — a small neural model from the snakers4 team. It is much more accurate at detecting speech boundaries, especially with background noise. Drop-in replacement for our current VAD.

**2. WebSocket real-time streaming** (1 week)
Stream audio from the browser microphone → server → process in chunks → send responses back. FastAPI supports WebSockets natively. This is the biggest UX improvement possible.

**3. LLM-based intent NLU** (2 days)
Llama is already loaded. Change the intent module to send a structured prompt to the LLM asking it to extract intent and slots. Handles paraphrasing, complex sentences, and multi-part commands automatically.

**4. 3-speaker support in the UI** (1 day)
`speechbrain/sepformer-libri3mix` exists on HuggingFace. Add it to the model registry. Update the UI to allow "3 speakers" selection. The rest of the pipeline handles N speakers already.

**5. MossFormer2** (1 day on Linux)
The best open-source separation model by objective metrics. Our `mossformer2_run.py` script is already written. Would work immediately on Linux or in a Docker container.

**6. Fine-tuning Whisper** (1 GPU-week)
Take a dataset of accented English speech, fine-tune Whisper large-v3 on it using HuggingFace Trainer. Expect WER to drop by 30–50% on accented speech.

---

# Summary: The Project in 5 Sentences

We built a **multi-user smart voice assistant** that solves a problem no consumer product has solved: understanding and responding to two people speaking at the same time.

The system works as a **7-stage pipeline**: detect speech → separate voices → identify speakers → transcribe → parse intent → generate personalized LLM response → display in web UI.

The three key technical contributions are: **IRM masking applied to the original mix** (better separation quality), **Hungarian algorithm for speaker assignment** (no duplicate identities), and **a model registry with user-selectable separation** (WHAMR 16K or LibriMix from the UI).

We use **only open-source models and datasets**: SepFormer, ECAPA-TDNN, faster-Whisper, Llama 3.2 3B — all downloaded automatically from HuggingFace — plus an offline-first LLM running on Intel Arc GPU via OpenVINO.

The result is a **fully working local web application** with demo generation, real audio upload, speaker enrollment, live metrics, and a conversational AI chat — all running on a laptop, with no cloud dependency.

---

*Document written as part of the Multi-User Smart Assistant project, May 2026.*
