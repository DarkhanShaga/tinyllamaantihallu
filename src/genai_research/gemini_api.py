from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request


class GeminiApiError(RuntimeError):
    pass


def generate_json_text(
    *,
    prompt: str,
    model: str,
    api_key: str,
    timeout_sec: int = 60,
) -> str:
    """
    Call Gemini generateContent REST API and return first text part.

    Uses JSON mode via responseMimeType=application/json.
    """
    api_key = api_key.strip()
    if not api_key:
        raise GeminiApiError("Missing Gemini API key.")
    if not model:
        raise GeminiApiError("Missing Gemini model id.")

    base = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    url = f"{base}?{urllib.parse.urlencode({'key': api_key})}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise GeminiApiError(f"Gemini HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise GeminiApiError(f"Gemini request failed: {e}") from e

    candidates = data.get("candidates") or []
    if not candidates:
        raise GeminiApiError(f"No candidates in Gemini response: {data}")
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    if not parts:
        raise GeminiApiError(f"No content parts in Gemini response: {data}")
    text = parts[0].get("text")
    if not isinstance(text, str):
        raise GeminiApiError(f"Gemini response missing text field: {data}")
    return text

