"""Small, classified boundary around the Gemini SDK."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from google import genai


RATE_LIMITED = "rate_limited"
PROVIDER_UNAVAILABLE = "provider_unavailable"
PROVIDER_TIMEOUT = "provider_timeout"
PROVIDER_AUTH_ERROR = "provider_auth_error"
PROVIDER_BAD_RESPONSE = "provider_bad_response"
MODEL_UNAVAILABLE = "model_unavailable"
UNKNOWN_PROVIDER_ERROR = "unknown_provider_error"
GEMINI_MODEL = "models/gemini-3.6-flash"


@dataclass(frozen=True)
class LLMProviderFailure(Exception):
    failure_type: str
    status_code: int | None = None


def _status_code(exc: Exception) -> int | None:
    for name in ("code", "status_code", "status"):
        value = getattr(exc, name, None)
        if isinstance(value, int):
            return value
    return None


def classify_provider_exception(exc: Exception) -> LLMProviderFailure:
    code = _status_code(exc)
    name = type(exc).__name__.lower()
    if code == 429:
        failure_type = RATE_LIMITED
    elif code == 503:
        failure_type = PROVIDER_UNAVAILABLE
    elif code == 404:
        # This boundary only calls the configured model-generation endpoint, so
        # a provider NOT_FOUND response means that model is unavailable.
        failure_type = MODEL_UNAVAILABLE
    elif code in {401, 403}:
        failure_type = PROVIDER_AUTH_ERROR
    elif isinstance(exc, TimeoutError) or "timeout" in name:
        failure_type = PROVIDER_TIMEOUT
    else:
        failure_type = UNKNOWN_PROVIDER_ERROR
    return LLMProviderFailure(failure_type, code)


def _client() -> Any:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise LLMProviderFailure(PROVIDER_AUTH_ERROR)
    return genai.Client(api_key=api_key)


def ask_gemini(prompt: str) -> str:
    """Generate once; classify provider failures without leaking SDK details."""
    try:
        client = _client()
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
        finally:
            client.close()
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise LLMProviderFailure(PROVIDER_BAD_RESPONSE)
        return text.strip()
    except LLMProviderFailure:
        raise
    except Exception as exc:
        raise classify_provider_exception(exc) from None
