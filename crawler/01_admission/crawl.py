#!/usr/bin/env python3
"""Crawler and processor for Category 01: Admission (Tuyển sinh HaUI).

Collects official HaUI admission projects, admission guidelines, methods,
and official notices from HaUI official websites:
- tuyensinh.haui.edu.vn
- www.haui.edu.vn
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from crawler.common.cleaner import clean_text, extract_date, extract_document_number, extract_year, trim_admission_project_noise
from crawler.common.http import HttpClient
from crawler.common.models import Document, SourceSeed, SourceType, Status
from crawler.common.parser import parse_html_document, parse_pdf_document
from crawler.common.utils import make_id, slugify, write_jsonl

logger = logging.getLogger(__name__)

CATEGORY = "01_admission"
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw" / CATEGORY
OUTPUT_FILE = ROOT_DIR / "data" / "documents" / f"{CATEGORY}.jsonl"

SEEDS = [
    SourceSeed(
        category=CATEGORY,
        subcategory="admission_project",
        source_name="Trường Đại học Công nghiệp Hà Nội",
        source_type=SourceType.OFFICIAL_WEBSITE.value,
        authority="Trường Đại học Công nghiệp Hà Nội",
        university="HaUI",
        document_number="356/QĐ-ĐHCN",
        document_type="Đề án",
        url="https://tuyensinh.haui.edu.vn/dai-hoc-chinh-quy/thong-tin-tuyen-sinh-trinh-do-dai-hoc-nam-2026/69b4e60495dfe0072a789cf6",
        title_hint="Thông tin tuyển sinh trình độ đại học năm 2026",
    ),
    SourceSeed(
        category=CATEGORY,
        subcategory="admission_notice",
        source_name="Trường Đại học Công nghiệp Hà Nội",
        source_type=SourceType.OFFICIAL_WEBSITE.value,
        authority="Trường Đại học Công nghiệp Hà Nội",
        university="HaUI",
        document_type="Thông báo",
        url="https://www.haui.edu.vn/vn/thong-bao/tuyen-sinh-dai-hoc-chinh-quy-nam-2026/67648",
        title_hint="Tuyển sinh đại học chính quy năm 2026",
    ),
    SourceSeed(
        category=CATEGORY,
        subcategory="admission_notice",
        source_name="Trường Đại học Công nghiệp Hà Nội",
        source_type=SourceType.OFFICIAL_WEBSITE.value,
        authority="Trường Đại học Công nghiệp Hà Nội",
        university="HaUI",
        document_type="Thông báo",
        url="https://www.haui.edu.vn/vn/thong-bao/tuyen-sinh-dai-hoc-nam-2026/67155",
        title_hint="Tuyển sinh đại học năm 2026",
    ),
    SourceSeed(
        category=CATEGORY,
        subcategory="admission_methods",
        source_name="Trường Đại học Công nghiệp Hà Nội",
        source_type=SourceType.OFFICIAL_WEBSITE.value,
        authority="Trường Đại học Công nghiệp Hà Nội",
        university="HaUI",
        document_type="Hướng dẫn",
        url="https://tuyensinh.haui.edu.vn/tuyen-sinh-dai-hoc-chinh-quy/cac-phuong-thuc-tuyen-sinh-dai-hoc-chinh-quy-2026/69ba2f5195dfe0072a789cf9",
        title_hint="Các phương thức tuyển sinh đại học chính quy 2026",
    ),
    SourceSeed(
        category=CATEGORY,
        subcategory="admission_guide",
        source_name="Trường Đại học Công nghiệp Hà Nội",
        source_type=SourceType.OFFICIAL_WEBSITE.value,
        authority="Trường Đại học Công nghiệp Hà Nội",
        university="HaUI",
        document_type="Hướng dẫn",
        url="https://tuyensinh.haui.edu.vn/huong-dan-dkxt%2C-nhap-hoc/huong-dan-dang-ky-du-tuyen-dai-hoc-chinh-quy-nam-2026/6a041a8daab4fe76b6ca54d7",
        title_hint="Hướng dẫn đăng ký dự tuyển đại học chính quy năm 2026",
    ),
    SourceSeed(
        category=CATEGORY,
        subcategory="admission_project",
        source_name="Trường Đại học Công nghiệp Hà Nội",
        source_type=SourceType.OFFICIAL_WEBSITE.value,
        authority="Trường Đại học Công nghiệp Hà Nội",
        university="HaUI",
        document_type="Đề án",
        url="https://tuyensinh.haui.edu.vn/dai-hoc-chinh-quy/thong-tin-tuyen-sinh-dai-hoc-nam-2025/68104542f721616a54f64960",
        title_hint="Thông tin tuyển sinh đại học năm 2025",
    ),
    SourceSeed(
        category=CATEGORY,
        subcategory="admission_project",
        source_name="Trường Đại học Công nghiệp Hà Nội",
        source_type=SourceType.OFFICIAL_WEBSITE.value,
        authority="Trường Đại học Công nghiệp Hà Nội",
        university="HaUI",
        document_type="Đề án",
        url="https://tuyensinh.haui.edu.vn/dai-hoc-chinh-quy/de-an-tuyen-sinh-trinh-do-dai-hoc-nam-2024/660e71a4811e513340b707a4",
        title_hint="Đề án tuyển sinh trình độ Đại học năm 2024",
    ),
    SourceSeed(
        category=CATEGORY,
        subcategory="admission_project",
        source_name="Trường Đại học Công nghiệp Hà Nội",
        source_type=SourceType.OFFICIAL_WEBSITE.value,
        authority="Trường Đại học Công nghiệp Hà Nội",
        university="HaUI",
        document_type="Đề án",
        url="https://tuyensinh.haui.edu.vn/dai-hoc-chinh-quy/de-an-tuyen-sinh-trinh-do-dai-hoc-nam-2023/6433c495ed3e304aa4d6e041",
        title_hint="Đề án tuyển sinh trình độ đại học năm 2023",
    ),
    SourceSeed(
        category=CATEGORY,
        subcategory="admission_project",
        source_name="Trường Đại học Công nghiệp Hà Nội",
        source_type=SourceType.OFFICIAL_WEBSITE.value,
        authority="Trường Đại học Công nghiệp Hà Nội",
        university="HaUI",
        document_type="Đề án",
        url="https://tuyensinh.haui.edu.vn/dai-hoc-chinh-quy/de-an-tuyen-sinh-trinh-do-dai-hoc-nam-2022-dieu-chinh-ngay-2262022/62b9272ad311c51544a6e847",
        title_hint="Đề án tuyển sinh trình độ đại học năm 2022 (Điều chỉnh ngày 22/6/2022)",
    ),
    SourceSeed(
        category=CATEGORY,
        subcategory="admission_project",
        source_name="Trường Đại học Công nghiệp Hà Nội",
        source_type=SourceType.OFFICIAL_WEBSITE.value,
        authority="Trường Đại học Công nghiệp Hà Nội",
        university="HaUI",
        document_type="Đề án",
        url="https://tuyensinh.haui.edu.vn/dai-hoc-chinh-quy/de--an-tuyen-sinh-trinh-do-dai-hoc-nam-2021-dieu-chinh-ngay-1082021/6124609566fa9c2e00bd3298",
        title_hint="Đề án tuyển sinh trình độ đại học năm 2021 (Điều chỉnh ngày 10/8/2021)",
    ),
    SourceSeed(
        category=CATEGORY,
        subcategory="admission_project",
        source_name="Trường Đại học Công nghiệp Hà Nội",
        source_type=SourceType.OFFICIAL_WEBSITE.value,
        authority="Trường Đại học Công nghiệp Hà Nội",
        university="HaUI",
        document_type="Đề án",
        url="https://tuyensinh.haui.edu.vn/dai-hoc-chinh-quy/de-an-tuyen-sinh-trinh-do-dai-hoc-nam-2020/5ed2f708dd95ed512cb8d00a",
        title_hint="Đề án tuyển sinh trình độ đại học năm 2020",
    ),
    SourceSeed(
        category=CATEGORY,
        subcategory="admission_project",
        source_name="Trường Đại học Công nghiệp Hà Nội",
        source_type=SourceType.OFFICIAL_WEBSITE.value,
        authority="Trường Đại học Công nghiệp Hà Nội",
        university="HaUI",
        document_type="Đề án",
        url="https://tuyensinh.haui.edu.vn/dai-hoc-chinh-quy/de-an-tuyen-sinh-dai-hoc-chinh-quy-nam-2019/5c91e3351392a4284838bc59",
        title_hint="Đề án tuyển sinh đại học chính quy năm 2019",
    ),
]


def process_source(
    seed: SourceSeed,
    index: int,
    http_client: HttpClient,
    crawled_date: str,
) -> Document:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_file = RAW_DIR / f"{slugify(seed.title_hint)}.html"

    if raw_file.exists() and raw_file.stat().st_size > 1000:
        raw_bytes = raw_file.read_bytes()
    else:
        raw_bytes, _ = http_client.get(seed.url)
        raw_file.write_bytes(raw_bytes)

    title, content = parse_html_document(raw_bytes)
    if not title or len(title) < 5:
        title = seed.title_hint
    content = trim_admission_project_noise(content)

    doc_id = make_id(CATEGORY, index)
    doc_year = extract_year(title, content, seed.url)
    doc_date = extract_date(content[:1000])
    doc_number = seed.document_number or extract_document_number(content[:1000])

    # Status: 2025/2026 are active, earlier years are archived
    status = Status.ACTIVE.value if (doc_year and doc_year >= 2025) else Status.ARCHIVED.value

    return Document(
        id=doc_id,
        title=title,
        content=content,
        source_url=seed.url,
        source_name=seed.source_name,
        source_type=seed.source_type,
        category=CATEGORY,
        crawled_date=crawled_date,
        language="vi",
        status=status,
        subcategory=seed.subcategory,
        year=doc_year,
        published_date=doc_date,
        university=seed.university,
        authority=seed.authority,
        document_number=doc_number,
        document_type=seed.document_type,
    )


def crawl_admission() -> list[Document]:
    http_client = HttpClient()
    crawled_date = date.today().isoformat()
    documents: list[Document] = []

    print(f"[{CATEGORY}] Processing {len(SEEDS)} official admission documents...")
    for idx, seed in enumerate(SEEDS, 1):
        try:
            doc = process_source(seed, idx, http_client, crawled_date)
            documents.append(doc)
            print(f"  ✓ [{doc.id}] ({doc.year}) {doc.title[:75]}")
        except Exception as exc:
            print(f"  ✗ Failed {seed.url}: {exc}")

    write_jsonl(documents, OUTPUT_FILE)
    print(f"[{CATEGORY}] Wrote {len(documents)} clean documents to {OUTPUT_FILE}")
    return documents


if __name__ == "__main__":
    crawl_admission()
