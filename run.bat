@echo off
echo Setting up environment...
set HF_HOME=%~dp0models
set TRANSFORMERS_CACHE=%~dp0models\transformers
set TORCH_HOME=%~dp0models\torch

echo Starting Multi-User Smart Assistant...
cd /d "%~dp0"
uvicorn app:app --reload --host 0.0.0.0 --port 8000
pause
