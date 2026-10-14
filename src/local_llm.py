"""Loopback-only adapters for Ollama and OpenAI-compatible local runtimes."""
import json
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def validate_url(url: str) -> str:
    url = url.strip().rstrip('/')
    parts = urlsplit(url)
    if parts.scheme != 'http' or parts.hostname not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('Use an HTTP localhost address for local inference.')
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError('The runtime address cannot contain credentials, a query, or a fragment.')
    return url


def request_json(base: str, path: str, body=None, timeout=4):
    url = validate_url(base) + path
    data = json.dumps(body).encode('utf-8') if body is not None else None
    request = Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode()).get('error', 'Runtime request failed')
            if isinstance(detail, dict):
                detail = detail.get('message', 'Runtime request failed')
        except (ValueError, UnicodeDecodeError):
            detail = 'Runtime request failed'
        raise RuntimeError(str(detail)) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError('Local runtime is unavailable. Start Ollama or your local API server and retry.') from exc


def discover(provider, base):
    payload = request_json(base, '/api/tags' if provider == 'ollama' else '/models')
    rows = payload.get('models' if provider == 'ollama' else 'data', [])
    return [{'name': row.get('name') or row.get('id'), 'size': row.get('size'),
             'parameters': row.get('details', {}).get('parameter_size')} for row in rows
            if row.get('name') or row.get('id')]


def chat(provider, base, model, messages):
    if not model:
        raise ValueError('Select an installed local model in Model settings.')
    if provider == 'ollama':
        payload = request_json(base, '/api/chat', {'model': model, 'messages': messages,
            'stream': False, 'options': {'num_predict': 250, 'temperature': 0.6}}, timeout=180)
        text = payload.get('message', {}).get('content', '')
    else:
        payload = request_json(base, '/chat/completions', {'model': model, 'messages': messages,
            'stream': False, 'max_tokens': 250, 'temperature': 0.6}, timeout=180)
        text = (payload.get('choices') or [{}])[0].get('message', {}).get('content', '')
    if not text.strip():
        raise RuntimeError('The local model returned an empty response. Try another model.')
    return text.strip()
