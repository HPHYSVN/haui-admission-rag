from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from crawler.common.models import Document


def make_id(category: str, index: int) -> str:
    """Generates unique ID following {category}_{number:03d} convention.

    Example: 01_admission_001, 02_ministry_regulations_002
    """
    return f"{category}_{index:03d}"


def slugify(text: str, max_len: int = 80) -> str:
    """Safe ascii slug for raw filenames."""
    text = text.lower()
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text[:max_len] or "document"


def write_jsonl(documents: list[Document], target_file: Path) -> None:
    """Writes list of Document objects to JSONL file with deterministic formatting."""
    target_file.parent.mkdir(parents=True, exist_ok=True)
    with target_file.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(source_file: Path) -> list[dict[str, Any]]:
    """Reads a JSONL file into list of dicts."""
    if not source_file.exists():
        return []
    docs = []
    with source_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line.strip()))
    return docs
