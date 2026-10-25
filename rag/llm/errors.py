"""Typed LLM provider failures.

Every adapter raises one of these instead of leaking ``requests`` exceptions, so callers
(the RAG pipeline, the API layer, the CLI) can react to a failure class without parsing
adapter-specific HTTP text.

They subclass :class:`RuntimeError` on purpose: ``app/main.py`` already turns
``RuntimeError``/``ValueError`` from the pipeline into a controlled HTTP 502, so introducing
these types changes no response contract.
"""

from __future__ import annotations


class ProviderError(RuntimeError):
    """Base class for all LLM provider failures.

    ``code`` is a stable, machine-readable identifier that also documents itself in the API
    error responses. Messages never contain credentials, and never contain the configured
    endpoint URL, so they are safe to surface.
    """

    code = "provider_error"

    def __init__(self, message: str, *, provider: str = "", model: str = "") -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model


class ProviderConfigurationError(ProviderError):
    """The provider is not usable as configured (missing model, bad base URL, bad request)."""

    code = "provider_misconfigured"


class ProviderUnreachable(ProviderError):
    """The endpoint could not be contacted at all (connection refused, DNS failure)."""

    code = "provider_unreachable"


class ProviderTimeout(ProviderError):
    """The endpoint accepted the connection but did not answer within the configured timeout."""

    code = "provider_timeout"


class ProviderHTTPError(ProviderError):
    """The endpoint answered with a non-2xx status."""

    code = "provider_http_error"

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider: str = "",
        model: str = "",
    ) -> None:
        super().__init__(message, provider=provider, model=model)
        self.status_code = status_code


class ProviderBadResponse(ProviderError):
    """The endpoint answered 2xx but the body was not the expected chat-completion shape."""

    code = "provider_bad_response"


class ProviderEmptyAnswer(ProviderError):
    """The endpoint answered 2xx with an empty or whitespace-only message."""

    code = "provider_empty_answer"
