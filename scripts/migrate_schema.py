#!/usr/bin/env python3
"""Migrate legacy JSONL records to the canonical category/ID convention."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT_DIR / "data" / "documents"
CANONICAL = {
    "admission": "01_admission",
    "ministry_regulations": "02_ministry_regulations",
    "cutoff_scores": "03_cutoff_scores",
    "curriculum": "04_curriculum",
    "tuition_scholarship": "05_tuition_scholarship",
    "career": "06_career",
    "career_orientation": "07_career_orientation",
    "enrollment_procedure": "08_enrollment_procedure",
    "training_regulations": "09_training_regulations",
    "dormitory_housing": "10_dormitory_housing",
    "student_life": "11_student_life",
    "student_support": "12_student_support",
    "training_systems": "13_training_systems",
    "about_haui": "14_about_haui",
}


def migrate_file(path: Path) -> int:
    prefix = path.stem.removesuffix(".jsonl")
    category = CANONICAL.get(prefix, prefix if prefix in CANONICAL.values() else None)
    if category is None:
        raise ValueError(f"Cannot infer canonical category from {path.name}")

    records: list[dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        record["category"] = category
        for key in (
            "subcategory",
            "year",
            "published_date",
            "effective_date",
            "expiry_date",
            "university",
            "authority",
            "document_number",
            "document_type",
        ):
            if record.get(key) is None:
                record.pop(key, None)
        old_id = str(record.get("id", ""))
        suffix_match = re.search(r"_(\d{3,})$", old_id)
        if not suffix_match:
            raise ValueError(f"{path.name}:{line_no}: ID has no numeric suffix: {old_id}")
        record["id"] = f"{category}_{suffix_match.group(1)}"
        records.append(record)

    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", type=Path, help="JSONL files; defaults to all corpus files")
    args = parser.parse_args()
    files = args.files or sorted(DOCS_DIR.glob("*.jsonl"))
    total = sum(migrate_file(path) for path in files)
    print(f"Migrated {total} records across {len(files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
