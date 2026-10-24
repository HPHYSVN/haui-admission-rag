"""Environment configuration for LLM generation, and nothing else.

Kept deliberately separate from :mod:`rag.config` (corpus paths, retrieval weights,
embeddings). Switching the generation model must never invalidate the vector index, and
rebuilding the index must never depend on which model happens to be serving chat right now.
The two are joined in exactly one place: :mod:`rag.llm` builds the provider, and
:mod:`rag.pipeline` builds the pipeline, so neither can change the other's configuration.

``LLM_*`` names are primary. The original ``VLLM_*`` names from the first version of this
project remain supported as aliases, so an existing ``.env`` keeps working untouched.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

# Reused so the generation token budget has a single definition in the project.
from rag.config import DEFAULT_MAX_OUTPUT_TOKENS

DEFAULT_PROVIDER = "openai_compatible"
#: Model served by the local vLLM instance this project was built against.
DEFAULT_MODEL = "JunHowie/Qwen3-4B-Instruct-2507-GPTQ-Int4"
DEFAULT_TIMEOUT = 120.0
#: Matches the value the pre-abstraction client hard-coded.
DEFAULT_TEMPERATURE = 0.1

#: Accepted ``LLM_PROVIDER`` names. All of them speak the OpenAI-compatible
#: ``/chat/completions`` API and share one adapter; the name only selects a default base URL.
PROVIDER_BASE_URLS: dict[str, str] = {
    "openai_compatible": "http://localhost:8000/v1",
    "vllm": "http://localhost:8000/v1",
    "ollama": "http://localhost:11434/v1",
    "openai": "https://api.openai.com/v1",
}


def _read_text(primary: str, alias: str | None, default: str) -> str:
    """Read ``primary``, then the legacy ``alias``; an empty value counts as unset."""
    for name in (primary, alias):
        if name is None:
            continue
        raw = os.getenv(name)
        if raw is not None and raw.strip():
            return raw.strip()
    return default


def _parse_float(
    name: str,
    raw: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {raw!r}")
    if minimum is not None:
        if exclusive_minimum and value <= minimum:
            raise ValueError(f"{name} must be greater than {minimum}, got {value}")
        if not exclusive_minimum and value < minimum:
            raise ValueError(f"{name} must be at least {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}, got {value}")
    return value


def _read_float(
    name: str,
    alias: str | None,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    raw = _read_text(name, alias, "")
    if not raw:
        return default
    return _parse_float(
        name,
        raw,
        minimum=minimum,
        maximum=maximum,
        exclusive_minimum=exclusive_minimum,
    )


def _optional_float(
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float | None:
    """Read an optional numeric setting; unset stays ``None`` so it is omitted from requests."""
    raw = _read_text(name, None, "")
    if not raw:
        return None
    return _parse_float(
        name,
        raw,
        minimum=minimum,
        maximum=maximum,
        exclusive_minimum=exclusive_minimum,
    )


def _read_int(name: str, alias: str | None, default: int, *, minimum: int) -> int:
    raw = _read_text(name, alias, "")
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def _read_json_object(name: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = _read_text(name, None, "")
    if not raw:
        return dict(default or {})
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a JSON object, got {raw!r}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class LLMConfig:
    """Validated generation settings.

    Attributes:
        provider: One of :data:`PROVIDER_BASE_URLS`.
        base_url: OpenAI-compatible root, e.g. ``http://localhost:8000/v1``. Never reported
            in API responses, so it stays internal to the deployment.
        model: Model identifier sent in the request body.
        api_key: Bearer token, or ``None`` for endpoints that need none. Never logged.
        timeout: Per-request timeout in seconds.
        temperature: Sampling temperature.
        max_tokens: Output token budget.
        top_p: Optional nucleus sampling parameter.
        extra_body: Additional request-body fields for provider-specific parameters
            (``repetition_penalty``, ``seed``, ``stop`` ...). ``model`` and ``messages``
            are ignored here and can never be overridden by this mapping.
    """

    provider: str = DEFAULT_PROVIDER
    base_url: str = PROVIDER_BASE_URLS[DEFAULT_PROVIDER]
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    top_p: float | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> LLMConfig:
        provider = _read_text("LLM_PROVIDER", None, DEFAULT_PROVIDER).casefold()
        if provider not in PROVIDER_BASE_URLS:
            raise ValueError(
                f"LLM_PROVIDER must be one of {sorted(PROVIDER_BASE_URLS)}, got {provider!r}"
            )
        return cls(
            provider=provider,
            base_url=_read_text(
                "LLM_BASE_URL", "VLLM_BASE_URL", PROVIDER_BASE_URLS[provider]
            ).rstrip("/"),
            model=_read_text("LLM_MODEL", "VLLM_MODEL", DEFAULT_MODEL),
            api_key=_read_text("LLM_API_KEY", "VLLM_API_KEY", "") or None,
            timeout=_read_float(
                "LLM_TIMEOUT", "VLLM_TIMEOUT", DEFAULT_TIMEOUT, minimum=0.0, exclusive_minimum=True
            ),
            temperature=_read_float(
                "LLM_TEMPERATURE", None, DEFAULT_TEMPERATURE, minimum=0.0, maximum=2.0
            ),
            max_tokens=_read_int("LLM_MAX_TOKENS", None, DEFAULT_MAX_OUTPUT_TOKENS, minimum=1),
            top_p=_optional_float(
                "LLM_TOP_P", minimum=0.0, maximum=1.0, exclusive_minimum=True
            ),
            extra_body=_read_json_object("LLM_EXTRA_BODY"),
        )
