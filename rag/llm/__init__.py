"""LLM generation package: one provider interface, one adapter family.

Public entry point:

    from rag.llm import build_provider
    provider = build_provider()          # from LLM_* / VLLM_* environment
    answer = provider.complete(messages)

Nothing here imports the RAG pipeline, so generation can be swapped without
touching retrieval, and retrieval can be reused with a different model.
"""

from __future__ import annotations

from rag.llm.base import ChatProvider
from rag.llm.config import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    PROVIDER_BASE_URLS,
    LLMConfig,
)
from rag.llm.errors import (
    ProviderBadResponse,
    ProviderConfigurationError,
    ProviderEmptyAnswer,
    ProviderError,
    ProviderHTTPError,
    ProviderTimeout,
    ProviderUnreachable,
)
from rag.llm.openai_compatible import OpenAICompatibleProvider

__all__ = [
    "ChatProvider",
    "DEFAULT_MODEL",
    "DEFAULT_PROVIDER",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT",
    "PROVIDER_BASE_URLS",
    "LLMConfig",
    "OpenAICompatibleProvider",
    "ProviderBadResponse",
    "ProviderConfigurationError",
    "ProviderEmptyAnswer",
    "ProviderError",
    "ProviderHTTPError",
    "ProviderTimeout",
    "ProviderUnreachable",
    "build_provider",
]


def build_provider(config: LLMConfig | None = None) -> ChatProvider:
    """Build the configured provider.

    Every supported ``LLM_PROVIDER`` name currently maps to the same
    OpenAI-compatible adapter; the name selects the default base URL and is
    reported by ``/api/v1/health``.
    """
    resolved = config or LLMConfig.from_env()
    return OpenAICompatibleProvider(
        provider=resolved.provider,
        base_url=resolved.base_url,
        model=resolved.model,
        api_key=resolved.api_key,
        timeout=resolved.timeout,
        temperature=resolved.temperature,
        max_tokens=resolved.max_tokens,
        top_p=resolved.top_p,
        extra_body=resolved.extra_body,
    )
