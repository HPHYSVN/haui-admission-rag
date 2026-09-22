#!/usr/bin/env python3
"""Crawler and processor for Category 02: Ministry Regulations (Quy định & Quy chế Bộ GD&ĐT).

Collects official regulations, circulars, decisions, and drafts from:
- Cơ sở dữ liệu văn bản Chính phủ (datafiles.chinhphu.vn)
- Cổng thông tin tuyển sinh Bộ Giáo dục và Đào tạo (tuyensinh.moet.gov.vn)
"""

from __future__ import annotations

import logging
import re
import sys
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from crawler.common.cleaner import extract_date, extract_document_number, extract_year
from crawler.common.http import HttpClient
from crawler.common.models import Document, SourceSeed, SourceType, Status
from crawler.common.parser import parse_html_document, parse_pdf_document
from crawler.common.utils import make_id, slugify, write_jsonl

logger = logging.getLogger(__name__)

CATEGORY = "02_ministry_regulations"
RAW_DIR = ROOT_DIR / "data" / "raw" / CATEGORY
OUTPUT_FILE = ROOT_DIR / "data" / "documents" / f"{CATEGORY}.jsonl"

SEEDS = [
    SourceSeed(
        category=CATEGORY,
        subcategory="admission_regulation",
        source_name="Cơ sở dữ liệu văn bản Chính phủ",
        source_type=SourceType.GOVERNMENT_DOCUMENT.value,
        authority="Bộ Giáo dục và Đào tạo",
        document_number="06/2026/TT-BGDĐT",
        document_type="Thông tư",
        url="https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/3/06-bgddt.pdf",
        title_hint="Quy chế tuyển sinh các ngành đào tạo trình độ đại học và ngành Giáo dục Mầm non trình độ cao đẳng (Thông tư 06/2026/TT-BGDĐT)",
    ),
    SourceSeed(
        category=CATEGORY,
        subcategory="draft_regulation",
        source_name="Cơ sở dữ liệu văn bản Chính phủ",
        source_type=SourceType.GOVERNMENT_DOCUMENT.value,
        authority="Bộ Giáo dục và Đào tạo",
        document_type="Dự thảo",
        url="https://datafiles.chinhphu.vn/cpp/files/duthaovbpl/2026/Thang9/duthaobgddt.pdf",
        title_hint="Dự thảo Thông tư sửa đổi, bổ sung một số điều các quy chế tuyển sinh và xác định số lượng tuyển sinh",
    ),
    SourceSeed(
        category=CATEGORY,
        subcategory="admission_capacity",
        source_name="Cổng thông tin tuyển sinh Bộ GD&ĐT",
        source_type=SourceType.GOVERNMENT_PORTAL.value,
        authority="Bộ Giáo dục và Đào tạo",
        document_number="34/2026/TT-BGDĐT",
        document_type="Thông tư",
        url="https://tuyensinh.moet.gov.vn/ts/van-ban/thong-tu-34-2026-tt-bgddt-cua-bo-giao-duc-va-dao-tao-quy-dinh-ve-viec-xac-dinh-so-luong-tuyen-sinh-d--2b075bf4-690e-4c62-879d-28912298a169",
        title_hint="Thông tư 34/2026/TT-BGDĐT về việc xác định số lượng tuyển sinh đại học, thạc sĩ, tiến sĩ và cao đẳng mầm non",
    ),
]

ATTACHMENT_RE = re.compile(r"doawloadfile\('([0-9a-fA-F-]{36})'\)")
MOET_FILE_URL = "https://datafile.moet.gov.vn/FileDinhKem/DownloadFile?id={file_id}"


def load_attached_pdf(raw_html: bytes, seed: SourceSeed, http_client: HttpClient) -> Path | None:
    """Follow the official attached PDF when the portal page only has a summary."""
    match = ATTACHMENT_RE.search(raw_html.decode("utf-8", errors="ignore"))
    if not match:
        return None

    pdf_path = RAW_DIR / f"{slugify(seed.title_hint)}.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 1000:
        return pdf_path

    file_url = MOET_FILE_URL.format(file_id=match.group(1))
    raw_bytes, _ = http_client.get(file_url)
    if not raw_bytes.startswith(b"%PDF"):
        return None
    pdf_path.write_bytes(raw_bytes)
    return pdf_path


def process_ministry_source(
    seed: SourceSeed,
    index: int,
    http_client: HttpClient,
    crawled_date: str,
) -> Document:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    is_pdf = seed.url.lower().endswith(".pdf")
    suffix = ".pdf" if is_pdf else ".html"
    raw_file = RAW_DIR / f"{slugify(seed.title_hint)}{suffix}"

    if raw_file.exists() and raw_file.stat().st_size > 500:
        raw_bytes = raw_file.read_bytes()
    else:
        raw_bytes, _ = http_client.get(seed.url)
        raw_file.write_bytes(raw_bytes)

    if is_pdf:
        title, content = parse_pdf_document(raw_file)
        if not title or len(title) < 10:
            title = seed.title_hint
    else:
        attached_pdf = load_attached_pdf(raw_bytes, seed, http_client)
        if attached_pdf is not None:
            title, content = parse_pdf_document(attached_pdf)
            if not title or len(title) < 10:
                title = seed.title_hint
        else:
            title, content = parse_html_document(raw_bytes)
            if not title or len(title) < 5:
                title = seed.title_hint

    doc_id = make_id(CATEGORY, index)
    doc_year = extract_year(seed.title_hint, content, seed.url) or 2026
    doc_date = extract_date(content[:1000])

    # For Thông tư 06/2026/TT-BGDĐT: published 15/02/2026
    if "06/2026/TT-BGDĐT" in (seed.document_number or ""):
        doc_date = "2026-02-15"
        doc_year = 2026
    elif "34/2026/TT-BGDĐT" in (seed.document_number or ""):
        doc_date = "2026-04-19"
        doc_year = 2026
    elif seed.document_type == "Dự thảo":
        doc_date = "2026-09-17"
        doc_year = 2026

    doc_number = seed.document_number or extract_document_number(content[:1000])

    return Document(
        id=doc_id,
        title=seed.title_hint,
        content=content,
        source_url=seed.url,
        source_name=seed.source_name,
        source_type=seed.source_type,
        category=CATEGORY,
        crawled_date=crawled_date,
        language="vi",
        status=Status.ACTIVE.value,
        subcategory=seed.subcategory,
        year=doc_year,
        published_date=doc_date,
        effective_date=doc_date,
        authority=seed.authority,
        document_number=doc_number,
        document_type=seed.document_type,
    )


def crawl_ministry_regulations() -> list[Document]:
    http_client = HttpClient()
    crawled_date = date.today().isoformat()
    documents: list[Document] = []

    print(f"[{CATEGORY}] Processing {len(SEEDS)} official ministry regulations...")
    for idx, seed in enumerate(SEEDS, 1):
        try:
            doc = process_ministry_source(seed, idx, http_client, crawled_date)
            documents.append(doc)
            print(f"  ✓ [{doc.id}] ({doc.year}) {doc.document_number or ''} {doc.title[:75]}")
        except Exception as exc:
            print(f"  ✗ Failed {seed.url}: {exc}")

    write_jsonl(documents, OUTPUT_FILE)
    print(f"[{CATEGORY}] Wrote {len(documents)} clean documents to {OUTPUT_FILE}")
    return documents


if __name__ == "__main__":
    crawl_ministry_regulations()
