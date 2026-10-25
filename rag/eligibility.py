"""Corpus eligibility policy for time-sensitive admissions questions.

The corpus deliberately keeps older admission projects, cutoff notices, and
superseded regulations: they are useful when a user explicitly asks about a past
year and they preserve provenance. They must not silently answer a current-year
question, however, and a draft regulation must never be treated as law.

This module keeps that policy in one reviewed JSON file under ``data/workflow``
and merges it onto chunks at load time. It does not delete or rewrite corpus
files, and review state remains the gate that decides whether final chunks are
loaded at all.

Policy scopes
-------------
``cycle``       A notice/project tied to one admission year. It may answer a
                question for that exact year, and is otherwise excluded from a
                year-sensitive query.
``regulation``  A legal/regulatory document with an effective date range. It is
                eligible when the requested year falls inside that range.
``evergreen``   Stable institutional information (campus, student life, career
                orientation) that is not tied to an admission cycle.
``quarantine``  Explicitly unsafe as evidence, for example a draft that has not
                been issued. Never retrieved.

No policy is allowed to turn an old document into current-year evidence. When a
year-sensitive question resolves to 2026 and retrieval only finds evergreen or
out-of-year material, the pipeline abstains instead of borrowing another year's
facts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

from rag.filters import YEAR_RE, query_filters

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ELIGIBILITY_FILE = ROOT_DIR / "data" / "workflow" / "document_eligibility.json"
DEFAULT_ADMISSION_YEAR = 2026

VALID_SCOPES = frozenset({"cycle", "regulation", "evergreen", "quarantine"})

#: Categories whose answers should default to the current admission cycle when
#: the user did not name a year. Other groups (career, dormitory, student life)
#: are mostly evergreen and must not be forced into 2026.
YEAR_SENSITIVE_CATEGORIES = frozenset(
    {
        "01_admission",
        "02_ministry_regulations",
        "03_cutoff_scores",
        "04_curriculum",
        "05_tuition_scholarship",
        "08_enrollment_procedure",
        "09_training_regulations",
    }
)

#: Wording that makes a question time-sensitive even when category hints are
#: absent. Kept deliberately narrow: broad words such as "đào tạo" would turn
#: institutional-history questions into current-year admission questions.
YEAR_SENSITIVE_HINTS = (
    "tuyển sinh",
    "xét tuyển",
    "điểm chuẩn",
    "điểm trúng tuyển",
    "điểm xét tuyển",
    "học phí",
    "học bổng",
    "nhập học",
    "chỉ tiêu",
    "nguyện vọng",
    "phương thức tuyển",
    "quy chế tuyển sinh",
    "ngưỡng đầu vào",
)


class EligibilityError(ValueError):
    """The curated eligibility file is missing or malformed."""


@dataclass(frozen=True)
class EligibilityRule:
    scope: str
    effective_from: date | None = None
    effective_to: date | None = None
    note: str = ""
    year: int | None = None


@dataclass(frozen=True)
class QueryScope:
    """The time window a retrieval query is allowed to use."""

    target_year: int | None
    explicit_year: int | None
    year_sensitive: bool
    category: str | None = None

    @property
    def is_time_scoped(self) -> bool:
        return self.target_year is not None


def _parse_date(value: Any, *, field: str, document_id: str) -> date | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise EligibilityError(f"{document_id}: {field} must be a YYYY-MM-DD string")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise EligibilityError(f"{document_id}: invalid {field} {value!r}") from exc


def _parse_year(value: Any, *, document_id: str) -> int | None:
    if value is None or value == "":
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise EligibilityError(f"{document_id}: year must be an integer")
    if not 1990 <= value <= 2100:
        raise EligibilityError(f"{document_id}: year is outside the supported range")
    return value


def _parse_rule(payload: Mapping[str, Any], *, document_id: str) -> EligibilityRule:
    scope = payload.get("scope")
    if scope not in VALID_SCOPES:
        raise EligibilityError(
            f"{document_id}: scope must be one of {sorted(VALID_SCOPES)}, got {scope!r}"
        )
    effective_from = _parse_date(
        payload.get("effective_from"),
        field="effective_from",
        document_id=document_id,
    )
    effective_to = _parse_date(
        payload.get("effective_to"),
        field="effective_to",
        document_id=document_id,
    )
    if effective_from and effective_to and effective_to < effective_from:
        raise EligibilityError(
            f"{document_id}: effective_to must not be before effective_from"
        )
    note = payload.get("note", "")
    if not isinstance(note, str):
        raise EligibilityError(f"{document_id}: note must be a string")
    return EligibilityRule(
        scope=scope,
        effective_from=effective_from,
        effective_to=effective_to,
        note=note,
        year=_parse_year(payload.get("year"), document_id=document_id),
    )


@dataclass(frozen=True)
class EligibilityPolicy:
    default_year: int
    defaults: Mapping[str, Mapping[str, Any]]
    documents: Mapping[str, Mapping[str, Any]]
    source: Path | None = None

    def rule_for(self, document: Mapping[str, Any]) -> EligibilityRule:
        document_id = str(document.get("id") or document.get("document_id") or "unknown")
        category = str(document.get("category") or "")
        merged: dict[str, Any] = {"scope": "evergreen"}
        merged.update(self.defaults.get(category, {}))
        merged.update(self.documents.get(document_id, {}))
        # A per-document cycle year is metadata, not a reason to ignore the
        # source document's own year. Only explicit policy years override it.
        return _parse_rule(merged, document_id=document_id)

    def segment(self, document: Mapping[str, Any]) -> dict[str, Any]:
        rule = self.rule_for(document)
        metadata: dict[str, Any] = {
            "eligibility_scope": rule.scope,
            "eligibility_note": rule.note,
            "eligibility_year": rule.year,
        }
        if rule.effective_from:
            metadata["eligibility_from"] = rule.effective_from.isoformat()
        if rule.effective_to:
            metadata["eligibility_to"] = rule.effective_to.isoformat()
        return metadata

    def missing_documents(self, document_ids: Iterable[str]) -> list[str]:
        return sorted(set(document_ids) - set(self.documents))


def load_eligibility_policy(
    path: Path = DEFAULT_ELIGIBILITY_FILE,
) -> EligibilityPolicy:
    if not path.exists():
        raise EligibilityError(f"Eligibility policy not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EligibilityError(f"Invalid eligibility JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EligibilityError(f"Eligibility policy must be a JSON object: {path}")

    default_year = payload.get("default_year", DEFAULT_ADMISSION_YEAR)
    if not isinstance(default_year, int) or isinstance(default_year, bool):
        raise EligibilityError("default_year must be an integer")

    raw_defaults = payload.get("defaults", {})
    raw_documents = payload.get("documents", {})
    if not isinstance(raw_defaults, dict) or not isinstance(raw_documents, dict):
        raise EligibilityError("defaults and documents must be JSON objects")

    defaults: dict[str, Mapping[str, Any]] = {}
    for category, rule in raw_defaults.items():
        if not isinstance(rule, dict):
            raise EligibilityError(f"Default for {category!r} must be an object")
        # Validate through the same parser, using a synthetic id for messages.
        _parse_rule({"scope": "evergreen", **rule}, document_id=f"default:{category}")
        defaults[str(category)] = dict(rule)

    documents: dict[str, Mapping[str, Any]] = {}
    for document_id, rule in raw_documents.items():
        if not isinstance(rule, dict):
            raise EligibilityError(f"Policy for {document_id!r} must be an object")
        parsed = _parse_rule(
            {"scope": defaults.get(str(rule.get("category", "")), {}).get("scope", "evergreen"), **rule},
            document_id=str(document_id),
        )
        documents[str(document_id)] = {**dict(rule), "scope": parsed.scope}
    return EligibilityPolicy(
        default_year=default_year,
        defaults=defaults,
        documents=documents,
        source=path,
    )


def apply_eligibility(
    chunks: Iterable[dict[str, Any]],
    policy: EligibilityPolicy | None,
) -> list[dict[str, Any]]:
    """Return copies of ``chunks`` with a merged eligibility segment."""
    if policy is None:
        return list(chunks)
    decorated: list[dict[str, Any]] = []
    for chunk in chunks:
        record = dict(chunk)
        source_document = {
            "id": record.get("document_id"),
            "category": record.get("category"),
        }
        record.update(policy.segment(source_document))
        decorated.append(record)
    return decorated


def resolve_query_scope(
    query: str,
    *,
    policy: EligibilityPolicy | None = None,
) -> QueryScope:
    """Resolve the year window for a question.

    An explicit year always wins. Otherwise, a time-sensitive question defaults
    to the current admission year from the policy (2026 by default). Evergreen
    questions keep ``target_year=None`` so they are not restricted to one cycle.
    """
    explicit_years = [int(year) for year in YEAR_RE.findall(query)]
    explicit = explicit_years[-1] if explicit_years else None
    filters = query_filters(query)
    category = filters.get("category")
    lowered = query.casefold()
    hinted = any(hint in lowered for hint in YEAR_SENSITIVE_HINTS)
    sensitive = bool(explicit) or category in YEAR_SENSITIVE_CATEGORIES or hinted
    default_year = policy.default_year if policy is not None else DEFAULT_ADMISSION_YEAR
    return QueryScope(
        target_year=explicit or (default_year if sensitive else None),
        explicit_year=explicit,
        year_sensitive=sensitive,
        category=category,
    )


def _year_of(chunk: Mapping[str, Any]) -> int | None:
    value = chunk.get("year")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _rule_dates(chunk: Mapping[str, Any]) -> tuple[date | None, date | None]:
    def parse(value: Any) -> date | None:
        if not value:
            return None
        try:
            return date.fromisoformat(str(value))
        except ValueError:
            return None

    return parse(chunk.get("eligibility_from")), parse(chunk.get("eligibility_to"))


def is_eligible(chunk: Mapping[str, Any], scope: QueryScope) -> bool:
    """Whether a chunk may be used for the resolved query scope."""
    if chunk.get("eligibility_scope") == "quarantine":
        return False
    if scope.target_year is None:
        return True

    eligibility = str(chunk.get("eligibility_scope") or "evergreen")
    year = _year_of(chunk)
    target = scope.target_year

    if eligibility == "evergreen":
        return True
    if eligibility == "cycle":
        # Unknown-year chunks are not silently assigned to the current cycle;
        # they remain eligible so the existing corpus/tests do not lose generic
        # material, but they never count as current-year evidence.
        return year is None or year == target
    if eligibility == "regulation":
        effective_from, effective_to = _rule_dates(chunk)
        if effective_from is None:
            effective_from = date(year, 1, 1) if year else None
        if effective_to is None and year and year > target:
            return False
        if effective_from and target < effective_from.year:
            return False
        if effective_to and target > effective_to.year:
            return False
        return True
    # Unknown scopes (older chunks without policy metadata) behave like cycle
    # metadata: exact year when known, otherwise keep them available.
    return year is None or year == target


def filter_chunks(
    chunks: Iterable[dict[str, Any]],
    scope: QueryScope,
) -> list[dict[str, Any]]:
    return [chunk for chunk in chunks if is_eligible(chunk, scope)]


def has_current_evidence(
    results: Iterable[tuple[float, dict[str, Any]]],
    target_year: int,
) -> bool:
    """True when at least one retrieved chunk is positive current-year evidence.

    Evergreen content is useful supporting context but is not proof that a
    current admission-cycle fact exists. This distinction is what prevents a
    2025 cutoff page from answering a 2026 cutoff question.
    """
    for _, chunk in results:
        if chunk.get("eligibility_scope") == "quarantine":
            continue
        if _year_of(chunk) == target_year:
            return True
        effective_from, effective_to = _rule_dates(chunk)
        if effective_from and effective_from.year <= target_year:
            if effective_to is None or target_year <= effective_to.year:
                return True
    return False


__all__ = [
    "DEFAULT_ADMISSION_YEAR",
    "DEFAULT_ELIGIBILITY_FILE",
    "EligibilityError",
    "EligibilityPolicy",
    "EligibilityRule",
    "QueryScope",
    "apply_eligibility",
    "filter_chunks",
    "has_current_evidence",
    "is_eligible",
    "load_eligibility_policy",
    "resolve_query_scope",
]
