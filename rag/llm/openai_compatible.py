"""One adapter for every OpenAI-compatible ``/chat/completions`` endpoint.

vLLM, Ollama, llama.cpp server, TGI, OpenAI, OpenRouter, DeepSeek and friends all
speak the same request/response shape, so they share this implementation. The
``LLM_PROVIDER`` value only selects a default base URL; it never changes the wire
protocol.

This module owns HTTP and error translation, and nothing else. It does not build
prompts, does not know the corpus, and does not read environment variables
(:class:`rag.llm.config.LLMConfig` does that).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import requests

from rag.llm.errors import (
    ProviderBadResponse,
    ProviderConfigurationError,
    ProviderEmptyAnswer,
    ProviderHTTPError,
    ProviderTimeout,
    ProviderUnreachable,
)

#: Body text included in an HTTP error is truncated to this many characters.
MAX_ERROR_BODY_CHARS = 1000

#: Keys that belong to the adapter and must not be overridden through ``extra_body``.
RESERVED_BODY_KEYS = frozenset({"model", "messages", "stream"})


class OpenAICompatibleProvider:
    """Single-turn completion over an OpenAI-compatible HTTP endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        provider: str = "openai_compatible",
        api_key: str | None = None,
        timeout: float = 120.0,
        temperature: float = 0.1,
        max_tokens: int = 800,
        top_p: float | None = None,
        extra_body: dict[str, Any] | None = None,
        session: requests.Session | None = None,
    ) -> None:
        if not base_url or not base_url.startswith(("http://", "https://")):
            raise ProviderConfigurationError(
                "LLM_BASE_URL must be an http(s) URL",
                provider=provider,
                model=model,
            )
        if not model:
            raise ProviderConfigurationError(
                "LLM_MODEL must be set to the model name served by the endpoint",
                provider=provider,
                model="",
            )
        if timeout <= 0:
            raise ProviderConfigurationError(
                "LLM_TIMEOUT must be greater than 0",
                provider=provider,
                model=model,
            )
        self.name = provider
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.extra_body = dict(extra_body or {})
        self._session = session or requests.Session()

    # -- interface ---------------------------------------------------------

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return the assistant text for ``messages``.

        Raises:
            ProviderError: one of the typed subclasses from :mod:`rag.llm.errors`.
        """
        if not messages:
            raise ProviderConfigurationError(
                "messages must not be empty",
                provider=self.name,
                model=self.model,
            )

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key

        try:
            response = self._session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=self._request_body(messages),
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise ProviderTimeout(
                f"LLM endpoint did not answer within {self.timeout:g}s",
                provider=self.name,
                model=self.model,
            ) from exc
        except requests.RequestException as exc:
            # str(exc) may contain the endpoint URL; report only the failure class.
            raise ProviderUnreachable(
                f"Could not reach the LLM endpoint ({type(exc).__name__})",
                provider=self.name,
                model=self.model,
            ) from exc

        status = int(getattr(response, "status_code", 0) or 0)
        if status >= 400:
            raise ProviderHTTPError(
                f"LLM endpoint answered HTTP {status}: {_truncate(_response_text(response))}",
                status_code=status,
                provider=self.name,
                model=self.model,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderBadResponse(
                "LLM endpoint returned a response that is not valid JSON",
                provider=self.name,
                model=self.model,
            ) from exc

        try:
            message = payload["choices"][0]["message"]
            answer = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderBadResponse(
                "LLM response did not contain choices[0].message.content",
                provider=self.name,
                model=self.model,
            ) from exc

        if not isinstance(answer, str) or not answer.strip():
            reasoning = ""
            if isinstance(message, dict):
                reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            hint = (
                "; the model produced reasoning but no final content, so raise "
                "LLM_MAX_TOKENS or disable thinking"
                if isinstance(reasoning, str) and reasoning.strip()
                else ""
            )
            raise ProviderEmptyAnswer(
                f"LLM endpoint returned an empty answer{hint}",
                provider=self.name,
                model=self.model,
            )
        return answer.strip()

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Yield assistant deltas from ``POST /chat/completions`` with ``stream=true``.

        The response is consumed line by line and never accumulated into a full
        answer here. Each yielded fragment is forwarded by the caller. The HTTP
        response is closed when this generator exits, including on cancellation
        or a provider error.
        """
        if not messages:
            raise ProviderConfigurationError(
                "messages must not be empty",
                provider=self.name,
                model=self.model,
            )

        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key

        try:
            response = self._session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=self._request_body(messages, stream=True),
                timeout=self.timeout,
                stream=True,
            )
        except requests.Timeout as exc:
            raise ProviderTimeout(
                f"LLM endpoint did not answer within {self.timeout:g}s",
                provider=self.name,
                model=self.model,
            ) from exc
        except requests.RequestException as exc:
            raise ProviderUnreachable(
                f"Could not reach the LLM endpoint ({type(exc).__name__})",
                provider=self.name,
                model=self.model,
            ) from exc

        try:
            status = int(getattr(response, "status_code", 0) or 0)
            if status >= 400:
                raise ProviderHTTPError(
                    f"LLM endpoint answered HTTP {status}: "
                    f"{_truncate(_response_text(response))}",
                    status_code=status,
                    provider=self.name,
                    model=self.model,
                )

            yielded = False
            for raw_line in response.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue
                line = (
                    raw_line.decode("utf-8", errors="replace")
                    if isinstance(raw_line, bytes)
                    else str(raw_line)
                ).strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise ProviderBadResponse(
                        "LLM stream contained a non-JSON data frame",
                        provider=self.name,
                        model=self.model,
                    ) from exc
                if not isinstance(payload, dict):
                    raise ProviderBadResponse(
                        "LLM stream contained a non-object data frame",
                        provider=self.name,
                        model=self.model,
                    )
                if payload.get("error"):
                    # Do not echo arbitrary provider text; classify it as a bad
                    # response and leave the details in the server-side log.
                    raise ProviderBadResponse(
                        "LLM stream reported an error",
                        provider=self.name,
                        model=self.model,
                    )
                choices = payload.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    delta = choice.get("message")
                if not isinstance(delta, dict):
                    continue
                content = delta.get("content")
                if isinstance(content, str) and content:
                    yielded = True
                    yield content
            if not yielded:
                raise ProviderEmptyAnswer(
                    "LLM endpoint returned an empty stream; the model may have spent "
                    "the token budget on reasoning (raise LLM_MAX_TOKENS or disable thinking)",
                    provider=self.name,
                    model=self.model,
                )
        finally:
            try:
                response.close()
            except Exception:  # pragma: no cover - defensive cleanup
                pass

    def probe(self, timeout: float = 2.0) -> bool:
        """Cheap reachability check for ``/ready``; never raises."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        try:
            response = self._session.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=timeout,
            )
        except requests.RequestException:
            return False
        return 200 <= int(getattr(response, "status_code", 0) or 0) < 300

    def close(self) -> None:
        self._session.close()

    # -- internals ---------------------------------------------------------

    def _request_body(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if stream:
            body["stream"] = True
        if self.top_p is not None:
            body["top_p"] = self.top_p
        for key, value in self.extra_body.items():
            if key in RESERVED_BODY_KEYS or value is None:
                continue
            body[key] = value
        return body


def _response_text(response: Any) -> str:
    try:
        return str(response.text).strip()
    except Exception:  # pragma: no cover - defensive against odd transports
        return ""


def _truncate(text: str, limit: int = MAX_ERROR_BODY_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."
