"""core/llm_client.py
Single boundary for Ollama communication.
"""
from __future__ import annotations

from typing import Any

import requests

from core.config import settings


def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    *,
    format: str | dict[str, Any] | None = None,
) -> dict:
    """Call Ollama chat.

    `format="json"` is used by the semantic parser so classifier output is
    machine-readable instead of relaxed text such as `manager, self, ...`.
    Normal conversational calls leave format unset.
    """
    payload: dict[str, Any] = {
        "model": settings.ollama_model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": settings.ollama_temperature},
    }
    if tools:
        payload["tools"] = tools
    if format is not None:
        payload["format"] = format

    url = f"{settings.ollama_base_url}/api/chat"

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=settings.ollama_timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("Ollama returned an invalid response shape.")
        return data
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Could not connect to Ollama at {settings.ollama_base_url}. "
            f"Make sure Ollama is running and model '{settings.ollama_model}' is pulled."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("Ollama request timed out.") from exc
    except requests.exceptions.HTTPError as exc:
        status = getattr(exc.response, "status_code", "unknown")
        raise RuntimeError(f"Ollama API returned HTTP {status}.") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Ollama request failed: {type(exc).__name__}") from exc


def is_available() -> bool:
    try:
        response = requests.get(
            f"{settings.ollama_base_url}/api/tags",
            timeout=3,
        )
        return response.status_code == 200
    except Exception:
        return False
