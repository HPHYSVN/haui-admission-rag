#!/usr/bin/env python3
"""Validation script for HaUI Admission RAG corpus.

Validates:
1. Valid JSON format on every line
2. All required fields exist
3. IDs are globally unique across all JSONL files and match the {category}_{number} format
4. title is non-empty
5. content is non-empty and meaningful
6. source_url is a valid HTTP/HTTPS URL
7. category is one of the 14 valid categories
8. source_type is one of the valid source types
9. language is 'vi'
10. status is 'active' or 'archived'
11. dates (crawled_date, published_date, effective_date, expiry_date) are valid ISO 8601 (YYYY-MM-DD)
12. year is an integer within valid range (1990 <= year <= 2030)
13. No disallowed extra fields (strict schema adherence)
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import jsonschema

ROOT_DIR = Path(__file__).resolve().parents[1]
SCHEMA_FILE = ROOT_DIR / "schemas" / "document.schema.json"
DEFAULT_DOCS_DIR = ROOT_DIR / "data" / "documents"

REQUIRED_FIELDS = [
    "id",
    "title",
    "content",
    "source_url",
    "source_name",
    "source_type",
    "category",
    "crawled_date",
    "language",
    "status",
]

ALLOWED_FIELDS = set(REQUIRED_FIELDS) | {
    "subcategory",
    "year",
    "published_date",
    "effective_date",
    "expiry_date",
    "university",
    "authority",
    "document_number",
    "document_type",
}

VALID_CATEGORIES = {
    "01_admission",
    "02_ministry_regulations",
    "03_cutoff_scores",
    "04_curriculum",
    "05_tuition_scholarship",
    "06_career",
    "07_career_orientation",
    "08_enrollment_procedure",
    "09_training_regulations",
    "10_dormitory_housing",
    "11_student_life",
    "12_student_support",
    "13_training_systems",
    "14_about_haui",
}

VALID_SOURCE_TYPES = {
    "official_website",
    "official_document",
    "official_portal",
    "government_document",
    "government_portal",
}

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ID_RE = re.compile(r"^(0[1-9]|1[0-4])_[a-z0-9_]+_\d{3,}$")


def is_valid_date(val: Any) -> bool:
    if not isinstance(val, str) or not DATE_RE.match(val):
        return False
    try:
        date.fromisoformat(val)
        return True
    except ValueError:
        return False


def is_valid_url(val: Any) -> bool:
    if not isinstance(val, str):
        return False
    try:
        parsed = urlparse(val)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def load_schema() -> dict[str, Any] | None:
    if SCHEMA_FILE.exists():
        try:
            return json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Warning: Cannot load schema file: {exc}", file=sys.stderr)
    return None


def validate_document(
    doc: dict[str, Any],
    file_path: Path,
    line_no: int,
    schema_validator: jsonschema.Draft202012Validator | None,
) -> tuple[list[str], int, int]:
    """Validates a single document object.

    Returns:
        (errors, missing_required_count, extra_field_count)
    """
    errors: list[str] = []
    missing_count = 0
    extra_count = 0

    # 1. Required fields
    for field in REQUIRED_FIELDS:
        if field not in doc:
            errors.append(f"Missing required field '{field}'")
            missing_count += 1

    # 2. Disallowed fields
    extra_fields = set(doc.keys()) - ALLOWED_FIELDS
    if extra_fields:
        errors.append(f"Disallowed extra fields: {sorted(extra_fields)}")
        extra_count += len(extra_fields)

    # 3. ID format
    doc_id = doc.get("id")
    if not doc_id or not isinstance(doc_id, str):
        errors.append("Invalid or missing 'id'")
    else:
        if not ID_RE.match(doc_id):
            errors.append(f"ID '{doc_id}' does not match required format '{{category}}_{{number:03d}}'")
        category = doc.get("category")
        if category and not doc_id.startswith(f"{category}_"):
            errors.append(f"ID '{doc_id}' prefix does not match category '{category}'")

    # 4. Title
    title = doc.get("title")
    if not isinstance(title, str) or not title.strip():
        errors.append("Field 'title' is empty or not a string")

    # 5. Content
    content = doc.get("content")
    if not isinstance(content, str) or not content.strip():
        errors.append("Field 'content' is empty or not a string")
    elif len(content.strip()) < 50:
        errors.append(f"Field 'content' too short ({len(content.strip())} chars), likely incomplete")

    # 6. Source URL
    source_url = doc.get("source_url")
    if not is_valid_url(source_url):
        errors.append(f"Invalid 'source_url': {source_url}")

    # 7. Category
    category = doc.get("category")
    if category not in VALID_CATEGORIES:
        errors.append(f"Invalid 'category': {category}")

    # 8. Source type
    source_type = doc.get("source_type")
    if source_type not in VALID_SOURCE_TYPES:
        errors.append(f"Invalid 'source_type': {source_type}")

    # 9. Language
    language = doc.get("language")
    if language != "vi":
        errors.append(f"Field 'language' must be 'vi', got '{language}'")

    # 10. Status
    status = doc.get("status")
    if status not in {"active", "archived"}:
        errors.append(f"Field 'status' must be 'active' or 'archived', got '{status}'")

    # 11. Dates
    crawled_date = doc.get("crawled_date")
    if not is_valid_date(crawled_date):
        errors.append(f"Invalid 'crawled_date': {crawled_date}")

    for date_field in ["published_date", "effective_date", "expiry_date"]:
        if date_field in doc and doc[date_field] is not None:
            if not is_valid_date(doc[date_field]):
                errors.append(f"Invalid '{date_field}': {doc[date_field]}")

    # 12. Year
    if "year" in doc and doc["year"] is not None:
        year = doc["year"]
        if not isinstance(year, int) or isinstance(year, bool) or not (1990 <= year <= 2030):
            errors.append(f"Invalid 'year': {year} (must be integer between 1990 and 2030)")

    # 13. JSONSchema check
    if schema_validator:
        for err in schema_validator.iter_errors(doc):
            err_msg = f"Schema error at '{'.'.join(str(p) for p in err.path)}': {err.message}"
            if err_msg not in errors:
                errors.append(err_msg)

    prefix = f"[{file_path.name}:{line_no}]"
    formatted_errors = [f"{prefix} {e}" for e in errors]
    return formatted_errors, missing_count, extra_count


def validate_corpus(target_dir: Path) -> int:
    files = sorted(target_dir.glob("*.jsonl"))
    if not files:
        print(f"Error: No JSONL files found in '{target_dir}'", file=sys.stderr)
        return 1

    schema_data = load_schema()
    schema_validator = jsonschema.Draft202012Validator(schema_data) if schema_data else None

    total_documents = 0
    total_missing_fields = 0
    total_duplicate_ids = 0
    all_errors: list[str] = []
    seen_ids: dict[str, tuple[str, int]] = {}
    category_counts: dict[str, int] = {}

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                clean_line = line.strip()
                if not clean_line:
                    continue

                total_documents += 1
                try:
                    doc = json.loads(clean_line)
                except json.JSONDecodeError as exc:
                    all_errors.append(f"[{file_path.name}:{line_no}] Invalid JSON: {exc}")
                    continue

                if not isinstance(doc, dict):
                    all_errors.append(f"[{file_path.name}:{line_no}] Line is not a JSON object")
                    continue

                doc_id = doc.get("id")
                if isinstance(doc_id, str):
                    if doc_id in seen_ids:
                        prev_file, prev_line = seen_ids[doc_id]
                        all_errors.append(
                            f"[{file_path.name}:{line_no}] Duplicate ID '{doc_id}' (first seen in {prev_file}:{prev_line})"
                        )
                        total_duplicate_ids += 1
                    else:
                        seen_ids[doc_id] = (file_path.name, line_no)

                cat = doc.get("category", "unknown")
                category_counts[cat] = category_counts.get(cat, 0) + 1

                errs, missing_c, _ = validate_document(doc, file_path, line_no, schema_validator)
                all_errors.extend(errs)
                total_missing_fields += missing_c

    print("========================================")
    print("      HaUI ADMISSION CORPUS REPORT      ")
    print("========================================")
    print(f"Documents: {total_documents}")
    print(f"Errors: {len(all_errors)}")
    print(f"Duplicate IDs: {total_duplicate_ids}")
    print(f"Missing required fields: {total_missing_fields}")
    print("----------------------------------------")
    print("Document count by category:")
    for cat in sorted(category_counts.keys()):
        print(f"  - {cat}: {category_counts[cat]} documents")
    print("========================================")

    if all_errors:
        print("\nVALIDATION FAILED with errors:")
        for err in all_errors[:50]:
            print(f"  * {err}")
        if len(all_errors) > 50:
            print(f"  ... and {len(all_errors) - 50} more errors.")
        return 1

    print("\nVALIDATION PASSED")
    return 0


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DOCS_DIR
    if not target.exists():
        print(f"Error: Directory '{target}' does not exist.", file=sys.stderr)
        return 1
    return validate_corpus(target)


if __name__ == "__main__":
    raise SystemExit(main())
