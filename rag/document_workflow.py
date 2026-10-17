"""Independent review state for documents, kept outside the source JSONL schema."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag.config import ROOT_DIR

DEFAULT_STATE_FILE = ROOT_DIR / "data" / "workflow" / "documents.json"
DOCUMENT_STATES = {
    "raw",
    "cleaned",
    "review_pending",
    "approved",
    "rejected",
    "chunk_proposed",
    "chunk_review",
    "final",
    "indexed",
}


def load_states(state_file: Path = DEFAULT_STATE_FILE) -> dict[str, dict[str, Any]]:
    if not state_file.exists():
        return {}
    try:
        payload = json.loads(state_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid document workflow state in {state_file}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Document workflow state must be a JSON object: {state_file}")
    return payload


def set_state(
    document_id: str,
    status: str,
    *,
    note: str = "",
    state_file: Path = DEFAULT_STATE_FILE,
) -> None:
    if status not in DOCUMENT_STATES:
        raise ValueError(f"Unsupported document workflow status: {status!r}")
    states = load_states(state_file)
    states[document_id] = {
        "status": status,
        "note": note,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    state_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = state_file.with_suffix(".json.tmp")
    temp_file.write_text(
        json.dumps(states, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_file.replace(state_file)


def review_document(
    document_id: str,
    action: str,
    *,
    note: str = "",
    state_file: Path = DEFAULT_STATE_FILE,
) -> str:
    transitions = {
        "pending": "review_pending",
        "approve": "approved",
        "reject": "rejected",
    }
    if action not in transitions:
        raise ValueError(f"Unsupported document review action: {action!r}")
    status = transitions[action]
    set_state(document_id, status, note=note, state_file=state_file)
    return status


def get_state(document_id: str, state_file: Path = DEFAULT_STATE_FILE) -> str:
    return load_states(state_file).get(document_id, {}).get("status", "cleaned")


def require_approved(document_id: str, state_file: Path = DEFAULT_STATE_FILE) -> None:
    status = get_state(document_id, state_file)
    if status != "approved":
        raise ValueError(
            f"Document {document_id!r} is {status!r}; mark it approved before generating chunks."
        )
