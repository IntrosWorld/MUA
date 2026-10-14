"""
LLM Engine
----------
Backends (in priority order):

  openvino — Llama 3.2 3B INT4 via Intel OpenVINO on Arc GPU / NPU / CPU
              Converted once with: optimum-cli export openvino --weight-format int4
              Runs on: Intel Arc GPU, Intel AI Boost NPU, or CPU fallback
              Speed vs CPU transformers: 5-20x faster on Arc/NPU

  offline  — Llama 3.2 3B via HuggingFace transformers (CPU, float32)
              Fallback when OpenVINO model not yet converted

  gemini   — Google Gemini 1.5 Flash (online, needs API key)

Default: offline (auto-upgrades to openvino if converted model exists)
"""

import os, sys
from .local_llm import discover, chat as local_chat, validate_url

BASE_DIR        = os.environ.get("MUA_DATA_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR      = os.path.join(BASE_DIR, "models", "llama-3.2-3b")
OV_MODELS_DIR   = os.path.join(BASE_DIR, "models", "llama-3.2-3b-ov")
CONFIG_FILE     = os.path.join(BASE_DIR, "data", "llm_config.json")

_mode           = "ollama"
_local_config = {"ollama_url": "http://127.0.0.1:11434", "ollama_model": "llama3.2:3b",
                 "local_api_url": "http://127.0.0.1:1234/v1", "local_api_model": ""}
_llm_model      = None
_llm_tok        = None
_ov_pipe        = None          # OpenVINO pipeline
_gemini_client  = None
_llm_ready      = False
_llm_error      = None
_active_backend = None          # "openvino" | "transformers"

SYSTEM_PROMPT = """You are a smart personal assistant embedded in a multi-user smart home system.
You have access to the user's personal profile including their schedule, fitness data, music playlists, movie watchlist, notes, and smart home controls.
Always address the user by their first name. Be concise, warm, and helpful.
When answering questions about their data, use the profile context provided.
Chat is read-only. Do not claim to have changed devices, created events, or saved data.
Room controls are simulated. If asked to change something, direct the user to the workspace controls.
Keep responses under 3 sentences unless asked for more detail."""


# ── Config persistence ──────────────────────────────────────────────────────

def _load_config():
    global _mode
    try:
        import json
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
            _mode = cfg.get("mode", "ollama")
            if _mode not in ("ollama", "local-api", "offline", "gemini"):
                _mode = "ollama"
            _local_config.update({k: cfg[k] for k in _local_config if k in cfg})
            if cfg.get("gemini_key") and not os.environ.get("GEMINI_API_KEY"):
                os.environ["GEMINI_API_KEY"] = cfg["gemini_key"]
    except Exception:
        pass

def _save_config():
    import json
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump({"mode": _mode, **_local_config, "gemini_key": os.environ.get("GEMINI_API_KEY", "")}, f)

_load_config()


# ── Mode control ────────────────────────────────────────────────────────────

def set_llm_mode(mode: str):
    global _mode
    if mode not in ("ollama", "local-api", "offline", "gemini"):
        raise ValueError("Unknown inference provider")
    _mode = mode
    _save_config()
    print(f"[LLM] Mode set to: {mode}")

def get_llm_mode() -> str:
    return _mode

def set_gemini_key(key: str):
    global _gemini_client
    _gemini_client = None
    os.environ["GEMINI_API_KEY"] = key
    _save_config()


def configure_local(body: dict):
    updates = {}
    for key in _local_config:
        if key in body:
            value = body[key]
            if not isinstance(value, str) or len(value) > 250:
                raise ValueError('Invalid runtime configuration')
            updates[key] = validate_url(value) if key.endswith('_url') else value.strip()
    _local_config.update(updates)
    _save_config()
    return dict(_local_config)


def local_models(provider=None, base=None):
    provider = provider or (_mode if _mode in ('ollama', 'local-api') else 'ollama')
    if provider not in ('ollama', 'local-api'):
        raise ValueError('Select Ollama or a local API runtime')
    url = base or _local_config['ollama_url' if provider == 'ollama' else 'local_api_url']
    return discover(provider, url)


# ── OpenVINO backend (Arc GPU / NPU) ────────────────────────────────────────

def _ov_model_ready() -> bool:
    """Check if the OpenVINO INT4 model has been converted and is ready."""
    return os.path.isdir(OV_MODELS_DIR) and any(
        f.endswith(".xml") for f in os.listdir(OV_MODELS_DIR)
    )


def _load_openvino():
    global _ov_pipe, _llm_tok, _llm_ready, _llm_error, _active_backend
    if _ov_pipe is not None:
        return True
    if not _ov_model_ready():
        _llm_error = (
            f"OpenVINO model not found at {OV_MODELS_DIR}. "
            "Run: optimum-cli export openvino --model models/llama-3.2-3b "
            "--weight-format int4 --task text-generation-with-past models/llama-3.2-3b-ov"
        )
        return False
    try:
        import openvino as ov
        from optimum.intel.openvino import OVModelForCausalLM
        from transformers import AutoTokenizer

        # Detect best device: Arc GPU > NPU > CPU
        core = ov.Core()
        available = core.available_devices
        if "GPU" in available:
            ov_device = "GPU"
        elif "NPU" in available:
            ov_device = "NPU"
        else:
            ov_device = "CPU"

        device_name = core.get_property(ov_device, "FULL_DEVICE_NAME")
        print(f"[LLM-OV] Loading Llama 3.2 3B INT4 on {ov_device} ({device_name})...")

        _ov_pipe = OVModelForCausalLM.from_pretrained(
            OV_MODELS_DIR,
            device=ov_device,
            compile=True,
        )
        _llm_tok = _load_tokenizer()
        _llm_ready = True
        _llm_error = None
        _active_backend = "openvino"
        print(f"[LLM-OV] Llama 3.2 3B loaded on {ov_device}. Ready.")
        return True
    except Exception as e:
        _llm_error = str(e)
        print(f"[LLM-OV] OpenVINO load failed: {e}")
        return False


def _load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        OV_MODELS_DIR if _ov_model_ready() else MODELS_DIR
    )


def _generate_openvino(messages: list, max_new_tokens: int = 200) -> str:
    global _llm_tok
    if not _load_openvino():
        return f"[OpenVINO unavailable: {_llm_error}]"
    if _llm_tok is None:
        _llm_tok = _load_tokenizer()
    import torch
    try:
        text = _llm_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _llm_tok(text, return_tensors="pt")
        with torch.no_grad():
            output = _ov_pipe.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=_llm_tok.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return _llm_tok.decode(new_tokens, skip_special_tokens=True).strip()
    except Exception as e:
        return f"[OpenVINO generation error: {e}]"


# ── Offline: Llama 3.2 3B via transformers (CPU fallback) ───────────────────

def _load_llama():
    global _llm_model, _llm_tok, _llm_ready, _llm_error, _active_backend
    if _llm_ready and _active_backend == "transformers":
        return True
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        local_dir = MODELS_DIR
        if not os.path.isfile(os.path.join(local_dir, "config.json")):
            _llm_error = f"Model not found at {local_dir}"
            return False

        dtype  = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[LLM] Loading Llama 3.2 3B on {device} ({dtype}) — CPU fallback...")
        _llm_tok   = AutoTokenizer.from_pretrained(local_dir)
        _llm_model = AutoModelForCausalLM.from_pretrained(
            local_dir, torch_dtype=dtype, device_map="auto", low_cpu_mem_usage=True
        )
        _llm_model.eval()
        _llm_ready      = True
        _llm_error      = None
        _active_backend = "transformers"
        print("[LLM] Llama 3.2 3B loaded (CPU transformers).")
        return True
    except Exception as e:
        _llm_error = str(e)
        print(f"[LLM] Llama load failed: {e}")
        return False


def _generate_llama(messages: list, max_new_tokens: int = 200) -> str:
    if not _load_llama():
        return f"[Offline LLM unavailable: {_llm_error}]"
    import torch
    try:
        text = _llm_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _llm_tok(text, return_tensors="pt").to(_llm_model.device)
        with torch.no_grad():
            output = _llm_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=_llm_tok.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return _llm_tok.decode(new_tokens, skip_special_tokens=True).strip()
    except Exception as e:
        return f"[LLM generation error: {e}]"


# ── Online: Gemini ───────────────────────────────────────────────────────────

def _load_gemini():
    global _gemini_client
    if _gemini_client is not None:
        return True
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        try:
            import json
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE) as f:
                    cfg = json.load(f)
                key = cfg.get("gemini_key", "")
                if key:
                    os.environ["GEMINI_API_KEY"] = key
        except Exception:
            pass
    if not key:
        return False
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        _gemini_client = genai.GenerativeModel("gemini-1.5-flash")
        return True
    except Exception as e:
        print(f"[LLM] Gemini load failed: {e}")
        return False


def _generate_gemini(messages: list) -> str:
    if not _load_gemini():
        return "[Gemini unavailable — check GEMINI_API_KEY]"
    try:
        import google.generativeai as genai
        system_text, user_msg = "", ""
        history = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            elif m["role"] == "user":
                user_msg = m["content"]
            elif m["role"] == "assistant":
                history.append({"role": "model", "parts": [m["content"]]})
        full_user = f"{system_text}\n\n{user_msg}" if system_text else user_msg
        chat     = _gemini_client.start_chat(history=history)
        return chat.send_message(full_user).text.strip()
    except Exception as e:
        return f"[Gemini error: {e}]"


# ── Public API ───────────────────────────────────────────────────────────────

def generate(
    user_name: str,
    user_context: str,
    transcript: str,
    intent: str,
    mode: str = None,
    history: list = None,
) -> dict:
    effective_mode = mode or _mode
    if effective_mode not in ('ollama', 'local-api', 'offline', 'gemini'):
        raise ValueError('Unknown inference provider')

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": (
            f"User profile:\n{user_context}\n\n"
            f"Detected intent: {intent}\n"
            f"User said: \"{transcript}\"\n\n"
            f"Please respond to {user_name} personally based on their profile data above."
        )},
    ]
    if history:
        messages[1:1] = [{'role': m['role'], 'content': m['content']} for m in history[-20:]
                         if m.get('role') in ('user', 'assistant') and isinstance(m.get('content'), str)]

    if effective_mode in ('ollama', 'local-api'):
        prefix = 'ollama' if effective_mode == 'ollama' else 'local_api'
        model_name = _local_config[prefix + '_model']
        try:
            response = local_chat(effective_mode, _local_config[prefix + '_url'], model_name, messages)
        except (RuntimeError, ValueError) as exc:
            return {'response': '', 'mode': effective_mode, 'model': model_name, 'error': str(exc)}
        return {'response': response, 'mode': effective_mode, 'model': model_name, 'error': None}
    elif effective_mode == "gemini":
        response = _generate_gemini(messages)
        model_name = "gemini-1.5-flash"
        if response.startswith("[Gemini"):
            print("[LLM] Gemini failed, falling back to offline.")
            effective_mode = "offline"
            response, model_name = _offline_generate(messages)
    else:
        response, model_name = _offline_generate(messages)

    return {
        "response": response,
        "mode": effective_mode,
        "model": model_name,
        "error": None if not response.startswith("[") else response,
    }


def _offline_generate(messages):
    """Try OpenVINO first, fall back to CPU transformers."""
    # Prefer OpenVINO (Arc GPU / NPU) if model is ready
    if _ov_model_ready():
        response = _generate_openvino(messages)
        if not response.startswith("[OpenVINO"):
            return response, f"Llama-3.2-3B-INT4 (OpenVINO/{_active_backend_device()})"

    # Fall back to CPU transformers
    response = _generate_llama(messages)
    return response, "Llama-3.2-3B (CPU)"


def _active_backend_device() -> str:
    try:
        import openvino as ov
        core = ov.Core()
        avail = core.available_devices
        if "GPU" in avail: return "Arc GPU"
        if "NPU" in avail: return "NPU"
    except Exception:
        pass
    return "CPU"


def get_status() -> dict:
    ov_ready = _ov_model_ready()
    return {
        "mode": _mode,
        "llama_ready": _llm_ready,
        "llama_error": _llm_error,
        "gemini_ready": _gemini_client is not None,
        "gemini_key_set": bool(os.environ.get("GEMINI_API_KEY")),
        "openvino_model_ready": ov_ready,
        "openvino_model_path": OV_MODELS_DIR if ov_ready else None,
        "active_backend": _active_backend,
        "local_config": dict(_local_config),
    }
