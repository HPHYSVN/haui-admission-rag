from __future__ import annotations

import html
import re
import unicodedata
from typing import Any

# Regex patterns
DATE_PATTERNS = [
    # "ngày 15 tháng 02 năm 2026" or "ngày 15/02/2026"
    re.compile(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", re.IGNORECASE),
    # "15/02/2026" or "15-02-2026"
    re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b"),
    # ISO "2026-02-15"
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
]

YEAR_PATTERNS = [
    re.compile(r"năm\s+(20\d{2})", re.IGNORECASE),
    re.compile(r"khóa\s+(20\d{2})", re.IGNORECASE),
    re.compile(r"tuyển\s+sinh\s+(?:đại\s+học\s+)?(?:năm\s+)?(20\d{2})", re.IGNORECASE),
    re.compile(r"\b(202[0-9])\b"),
]

DOC_NUMBER_PATTERNS = [
    re.compile(r"\b(\d{1,4}/(?:20\d{2}/)?(?:TT|QĐ|TB|KH|VBHN)-[A-ZĐ_]+)\b", re.IGNORECASE),
    re.compile(r"Số:\s*(\d{1,4}/(?:20\d{2}/)?(?:TT|QĐ|TB|KH|VBHN)-[A-ZĐ_]+)", re.IGNORECASE),
]


def clean_text(text: str) -> str:
    """Cleans raw text: unescape html, normalize unicode, strip excessive whitespace."""
    if not text:
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace horizontal tabs and multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_date(text: str) -> str | None:
    """Extracts date in ISO YYYY-MM-DD format."""
    if not text:
        return None

    for pat in DATE_PATTERNS:
        match = pat.search(text)
        if match:
            groups = match.groups()
            if len(groups) == 3:
                if len(groups[0]) == 4:  # ISO YYYY-MM-DD
                    y, m, d = int(groups[0]), int(groups[1]), int(groups[2])
                else:  # DD, MM, YYYY
                    d, m, y = int(groups[0]), int(groups[1]), int(groups[2])
                if 1990 <= y <= 2030 and 1 <= m <= 12 and 1 <= d <= 31:
                    return f"{y:04d}-{m:02d}-{d:02d}"
    return None


def extract_year(title: str, text: str, fallback_url: str = "") -> int | None:
    """Extracts admission/academic year reliably without picking up irrelevant numbers."""
    # Priority 1: Check title first (most authoritative for year)
    title_match = re.search(r"năm\s+(20\d{2})", title, re.IGNORECASE)
    if title_match:
        y = int(title_match.group(1))
        if 2015 <= y <= 2030:
            return y

    year_in_title = re.findall(r"\b(20\d{2})\b", title)
    if year_in_title:
        years = [int(y) for y in year_in_title if 2015 <= int(y) <= 2030]
        if years:
            return years[-1]

    # Priority 2: Check URL
    url_match = re.findall(r"\b(20\d{2})\b", fallback_url)
    if url_match:
        years = [int(y) for y in url_match if 2015 <= int(y) <= 2030]
        if years:
            return years[-1]

    # Priority 3: Check start of text (first 500 chars)
    first_chunk = text[:500]
    for pat in YEAR_PATTERNS:
        m = pat.search(first_chunk)
        if m:
            y = int(m.group(1))
            if 2015 <= y <= 2030:
                return y

    return None


def extract_document_number(text: str) -> str | None:
    """Extracts legal or official administrative document numbers."""
    for pat in DOC_NUMBER_PATTERNS:
        match = pat.search(text)
        if match:
            return match.group(1).strip()
    return None


def is_irrelevant_table(header_text: str, rows_text: str) -> bool:
    """Checks if a table is an irrelevant equipment list or facility inventory."""
    keywords = [
        "máy phay cnc",
        "danh mục trang thiết bị chính",
        "phòng thực hành",
        "máy tiện",
        "panme",
        "thước cặp",
        "bình cứu hỏa",
        "điều hòa funiki",
        "máy in",
        "máy phún xạ",
    ]
    combined = (header_text + " " + rows_text).lower()
    matches = sum(1 for kw in keywords if kw in combined)
    return matches >= 2


def trim_admission_project_noise(text: str) -> str:
    """Remove long institutional appendices that are not useful for admission Q&A.

    Older admission projects include hundreds of lines of facility inventories,
    library counts, staff lists, finance statistics, and employment surveys.
    They are official, but they are not useful for the admission RAG scope and
    make retrieval noisy. Keep the admission plan before these appendices.
    """
    if not text:
        return ""

    range_patterns = [
        (
            r"\n\s*II\.\s*Thông tin về các điều kiện bảo đảm chất lượng.*?",
            r"\n\s*III\.\s*Các thông tin",
        ),
        (
            r"\n\s*II\.\s*Thông tin về các điều kiện đảm bảo chất lượng.*?",
            r"\n\s*III\.\s*Các thông tin",
        ),
        (
            r"\n\s*1\.10\.\s*Điều kiện bảo đảm chất lượng.*?",
            r"\n\s*1\.11\.",
        ),
        (
            r"\n\s*1\.10\.\s*Điều kiện đảm bảo chất lượng.*?",
            r"\n\s*1\.11\.",
        ),
        (
            r"\n\s*1\.10\.\s*Điều kiện bảo đảm chất lượng.*?",
            r"\n\s*II\.\s*TUYỂN SINH",
        ),
    ]
    for start_pattern, end_pattern in range_patterns:
        start = re.search(start_pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not start:
            continue
        end = re.search(end_pattern, text[start.end() :], flags=re.IGNORECASE)
        if not end:
            continue
        end_start = start.end() + end.start()
        text = text[: start.start()] + "\n\n" + text[end_start:]

    stop_patterns = [
        r"\n\s*1\.10\.\s*Điều kiện bảo đảm chất lượng",
        r"\n\s*1\.10\.\s*Điều kiện đảm bảo chất lượng",
        r"\n\s*4\.\s*Các điều kiện bảo đảm chất lượng",
        r"\n\s*4\.\s*Thông tin về các [Đđ]iều kiện",
        r"\n\s*4\.1\.\s*Cơ sở vật chất",
        r"\n\s*4\.2\.\s*Danh sách giảng viên",
        r"\n\s*Danh sách giảng viên cơ hữu",
        r"\n\s*DANH SÁCH GIẢNG VIÊN",
        r"\n\s*Danh mục trang thiết bị",
        r"\n\s*PHỤ LỤC\s+\d*",
        r"\n\s*Phụ lục\s+\d*",
        r"\n\s*5\.\s*Tình hình việc làm",
        r"\n\s*6\.\s*Tài chính",
    ]

    cut_at: int | None = None
    for pattern in stop_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match and (cut_at is None or match.start() < cut_at):
            cut_at = match.start()

    if cut_at is not None:
        text = text[:cut_at]

    return clean_text(text)
