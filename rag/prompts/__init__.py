"""Prompt profiles, selected independently from the LLM provider and retrieval.

``RAG_PROMPT_PROFILE`` chooses the policy:
  * ``haui_admission`` (default) keeps the existing HaUI scope/year rules.
  * ``generic`` is a neutral builder for reusing the RAG core elsewhere.

The pipeline depends on the :class:`PromptBuilder` protocol, not on either
concrete module, so adding a profile does not touch retrieval or generation.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

from dotenv import load_dotenv

from rag.config import ROOT_DIR

DEFAULT_PROMPT_PROFILE = "haui_admission"
PROMPT_PROFILES = ("haui_admission", "generic")

load_dotenv(ROOT_DIR / ".env", override=False)


@runtime_checkable
class PromptBuilder(Protocol):
    """Turns a question plus retrieved chunks into a chat ``messages`` list."""

    name: str

    def build(
        self,
        question: str,
        results: list[tuple[float, dict[str, Any]]],
        history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        ...


def prompt_profile_name() -> str:
    return (
        os.getenv("RAG_PROMPT_PROFILE", DEFAULT_PROMPT_PROFILE).strip().casefold()
        or DEFAULT_PROMPT_PROFILE
    )


def build_prompt_builder(name: str | None = None) -> PromptBuilder:
    """Build the prompt profile named by ``name`` or ``RAG_PROMPT_PROFILE``."""
    profile = (name or prompt_profile_name()).strip().casefold()
    if profile in {"haui", "haui_admission"}:
        from rag.prompts.haui_admission import HauiAdmissionPromptBuilder

        return HauiAdmissionPromptBuilder()
    if profile == "generic":
        from rag.prompts.generic import GenericPromptBuilder

        return GenericPromptBuilder()
    raise ValueError(
        f"RAG_PROMPT_PROFILE must be one of {list(PROMPT_PROFILES)}, got {profile!r}"
    )


__all__ = [
    "DEFAULT_PROMPT_PROFILE",
    "PROMPT_PROFILES",
    "PromptBuilder",
    "build_prompt_builder",
    "prompt_profile_name",
]
