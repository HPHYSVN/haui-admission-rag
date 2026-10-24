"""The generation-side interface the RAG core depends on.

Adapters receive an already-built ``messages`` list rather than ``(question, chunks)``. That
is what keeps prompt construction in the pipeline and retrieval untouched: this layer holds
nothing but inference, and no adapter knows the corpus, the retriever, or the prompt wording.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class ChatProvider(Protocol):
    """Single-turn text generation over a message list.

    Attributes:
        name: Adapter family, e.g. ``vllm`` or ``ollama``. Not a credential, safe to report.
        model: The model identifier requested from the endpoint.
    """

    name: str
    model: str

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return the assistant text for ``messages``.

        Raises:
            ProviderError: a subclass from :mod:`rag.llm.errors` describing the failure class.
        """
        ...

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Yield assistant text fragments as the endpoint produces them.

        Implementations must raise the same typed errors as :meth:`complete`.
        The returned iterator owns the HTTP response and closes it when exhausted
        or when the caller calls ``close()``. Callers must not join all fragments
        before forwarding them; the RAG streaming endpoint forwards each one.
        """
        ...
