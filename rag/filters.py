"""Query-to-metadata constraints shared by hybrid retrieval."""

from __future__ import annotations

import re
from typing import Any

YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
QUERY_STOP_WORDS = {
    "bao", "bằng", "bị", "cho", "chính", "có", "của", "đang", "để", "đi", "điều", "được",
    "gì", "hay", "khi", "là", "mức", "nào", "như", "ở", "ra", "sao", "theo",
    "sách", "sinh", "thế", "thì", "tôi", "trong", "và", "về", "viên", "với",
    "yêu", "cầu", "năm", "mấy",
}

CATEGORY_HINTS: dict[str, tuple[str, ...]] = {
    "03_cutoff_scores": ("điểm chuẩn", "điểm trúng tuyển", "điểm xét tuyển trúng tuyển"),
    "05_tuition_scholarship": ("học phí", "học bổng", "miễn giảm học phí"),
    "01_admission": ("phương thức tuyển sinh", "tuyển sinh", "đăng ký xét tuyển"),
    "08_enrollment_procedure": ("nhập học", "hồ sơ nhập học", "thủ tục nhập học"),
    "04_curriculum": ("chương trình đào tạo", "chương trình học", "môn học"),
}


def query_filters(query: str) -> dict[str, Any]:
    lowered = query.casefold()
    years = YEAR_RE.findall(lowered)
    filters: dict[str, Any] = {}
    if years:
        filters["year"] = int(years[-1])
    for category, phrases in CATEGORY_HINTS.items():
        if any(phrase in lowered for phrase in phrases):
            filters["category"] = category
            break
    return filters


def matching_documents(
    documents: list[dict[str, Any]],
    filters: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates = documents
    if "year" in filters:
        candidates = [document for document in candidates if document.get("year") == filters["year"]]
    if "category" in filters:
        candidates = [document for document in candidates if document.get("category") == filters["category"]]
    return candidates


def query_term_coverage(query: str, document: dict[str, Any]) -> float:
    meaningful = {
        word.casefold()
        for word in WORD_RE.findall(query)
        if word.casefold() not in QUERY_STOP_WORDS and not word.isdigit() and len(word) > 1
    }
    if not meaningful:
        return 1.0
    heading_path = document.get("heading_path", document.get("section_path", [])) or []
    content = " ".join(
        [
            str(document.get("title", "")),
            " ".join(str(item) for item in heading_path),
            str(document.get("text", "")),
        ]
    )
    present = set(WORD_RE.findall(content.casefold()))
    return len(meaningful & present) / len(meaningful)
