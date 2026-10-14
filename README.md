# MUA: Multi-User Smart Assistant

MUA separates overlapping voices, identifies enrolled speakers, transcribes their speech, and responds using personal workspace context. This research prototype runs in a browser or an Electron desktop window.

The `ui-ux` branch contains the redesigned interface, local LLM adapters, and desktop packaging. The product website has its own repository: https://github.com/IntrosWorld/mua-website.

## Run the application

Use Python 3.11+ and Node.js 22+.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
npm ci
npm start
```

On macOS/Linux, activate the environment with `source .venv/bin/activate`. Electron starts Python on an available loopback port and stops it when the app closes. Set `MUA_PYTHON` to a Python executable if it is not on PATH. The launcher automatically detects a project `.venv`.

For the browser interface only:

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000.

## Local models

Install and start Ollama, then install a model:

```powershell
ollama pull llama3.2:3b
```

In **Model settings**, select **Ollama**, use `http://127.0.0.1:11434`, refresh the installed models, and save a choice. MUA uses installed models and does not download LLM weights automatically. If Ollama uses a custom model directory, start it with the same `OLLAMA_MODELS` setting as your installation.

For LM Studio or llama.cpp, start an OpenAI-compatible server, choose **Local API**, and enter its address including `/v1`, for example `http://127.0.0.1:1234/v1`. These providers accept loopback HTTP addresses and require no API key.

Legacy Transformers/OpenVINO weights remain available through **Local model weights**, with optional runtime dependencies and existing models under `models/llama-3.2-3b` or `models/llama-3.2-3b-ov`. Gemini remains an optional cloud provider.

## Using the workspace

- **Voice studio:** choose a demo conversation or upload a recording; configure overlap, noise, and separation; run the pipeline and export a report.
- **People:** enroll 1 to 5 recordings per person and check a single-speaker voice match.
- **Workspaces:** manage schedules, reminders, notes, playlists, watchlists, and simulated room controls. Assistant chat keeps separate history per person during the browser session.
- **Model settings:** discover local models, check the connection, and save the response engine.

Reference demo mode uses known clean sources and does not evaluate real separation. Results show identity, transcripts, response sources, and fallback warnings.

## Windows desktop build

```powershell
npm run desktop:dist
```

This creates a portable executable in `desktop-dist/`. It bundles Electron, the interface, and Python source. **Python and the packages in requirements.txt must be installed on the target machine.** Models and personal data are excluded. This is an unsigned development build, not a self-contained Python installer.

Packaged builds use the application's user-data directory. Set `MUA_DATA_DIR` to an absolute directory to keep profiles, uploads, outputs, and model caches on another drive. Development uses the repository directory by default.

To package from an installed Electron distribution:

```powershell
npm run desktop:dist -- --config.electronDist=node_modules/electron/dist
```

## Checks

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
npm run test:desktop
```

With the backend at http://127.0.0.1:8765, run `npm run test:ui`. Set `MUA_TEST_URL` for another address. Browser checks cover inputs, report export, identification, workspace edits, chat history, model settings, escaping, both themes, and three viewport sizes. Install a browser with `npx playwright install chromium` if needed, or set `MUA_CHROMIUM`.

## Architecture and data

Mixed audio → voice activity detection → separation → speaker identification → recognition → intent detection → personal response.

The backend is FastAPI with SpeechBrain, faster-whisper, and soundfile. Local LLM adapters use Python's standard HTTP client. The buildless interface uses self-hosted Geist fonts and licensed Phosphor icons.

Profiles and LLM configuration live under `data/`; outputs and reports under `outputs/`. Recordings, runtime configuration, models, and build outputs are ignored by Git.

This is a research prototype: voice matching is not authentication, room devices are simulated, and model answers can be inaccurate. Demo TTS is online and speech models may need initial downloads. Selecting Gemini sends profile context to a cloud provider.
