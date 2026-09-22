#!/usr/bin/env python3
"""Crawler and processor for Category 03: Cutoff Scores (Điểm chuẩn trúng tuyển HaUI).

Collects official cutoff score announcements, early admission scores, and
entrance criteria across admission years (2022 - 2026) from:
- tuyensinh.haui.edu.vn/diem-chuan-trung-tuyen-dai-hoc
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bs4 import BeautifulSoup

from crawler.common.cleaner import clean_text, extract_date, extract_document_number, extract_year
from crawler.common.http import HttpClient
from crawler.common.models import Document, SourceSeed, SourceType, Status
from crawler.common.parser import parse_html_document
from crawler.common.utils import make_id, slugify, write_jsonl

logger = logging.getLogger(__name__)

CATEGORY = "03_cutoff_scores"
RAW_DIR = ROOT_DIR / "data" / "raw" / CATEGORY
OUTPUT_FILE = ROOT_DIR / "data" / "documents" / f"{CATEGORY}.jsonl"

SEEDS = [
    SourceSeed(
        category=CATEGORY,
        subcategory="cutoff_results",
        source_name="Trường Đại học Công nghiệp Hà Nội",
        source_type=SourceType.OFFICIAL_WEBSITE.value,
        authority="Hội đồng tuyển sinh Trường Đại học Công nghiệp Hà Nội",
        university="HaUI",
        document_type="Thông báo",
        url="https://tuyensinh.haui.edu.vn/diem-chuan-trung-tuyen-dai-hoc/ket-qua-xet-tuyen-dai-hoc-chinh-quy-dot-2-chuong-trinh-dao-tao-bang-tieng-anh-nam-2026/6a9aac2615f3e30f913e9cac",
        title_hint="Kết quả xét tuyển đại học chính quy đợt 2 chương trình đào tạo bằng Tiếng Anh năm 2026",
    ),
    SourceSeed(
        category=CATEGORY,
        subcategory="cutoff_results",
        source_name="Trường Đại học Công nghiệp Hà Nội",
        source_type=SourceType.OFFICIAL_WEBSITE.value,
        authority="Hội đồng tuyển sinh Trường Đại học Công nghiệp Hà Nội",
        university="HaUI",
        document_number="289/TB-ĐHCN",
        document_type="Thông báo",
        url="https://tuyensinh.haui.edu.vn/diem-chuan-trung-tuyen-dai-hoc/ket-qua-xet-tuyen-dai-hoc-chinh-quy-nam-2026/6a799d11bcd57668ce75276d",
        title_hint="Thông báo kết quả xét tuyển đại học chính quy năm 2026 (Thông báo 289/TB-ĐHCN)",
    ),
    SourceSeed(
        category=CATEGORY,
        subcategory="cutoff_results",
        source_name="Trường Đại học Công nghiệp Hà Nội",
        source_type=SourceType.OFFICIAL_WEBSITE.value,
        authority="Hội đồng tuyển sinh Trường Đại học Công nghiệp Hà Nội",
        university="HaUI",
        document_number="290/TB-ĐHCN",
        document_type="Thông báo",
        url="https://tuyensinh.haui.edu.vn/diem-chuan-trung-tuyen-dai-hoc/ket-qua-xet-tuyen-dai-hoc-chinh-quy-chuong-trinh-dao-tao-bang-tieng-anh-nam-2026/6a799ed0bcd57668ce75276e",
        title_hint="Thông báo kết quả xét tuyển đại học chính quy chương trình đào tạo bằng Tiếng Anh năm 2026 (Thông báo 290/TB-ĐHCN)",
    ),
    SourceSeed(
        category=CATEGORY,
        subcategory="cutoff_results",
        source_name="Trường Đại học Công nghiệp Hà Nội",
        source_type=SourceType.OFFICIAL_WEBSITE.value,
        authority="Hội đồng tuyển sinh Trường Đại học Công nghiệp Hà Nội",
        university="HaUI",
        document_type="Thông báo",
        url="https://tuyensinh.haui.edu.vn/diem-chuan-trung-tuyen-dai-hoc/ket-qua-xet-tuyen-dai-hoc-chinh-quy-nam-2025/68affb443021ae3984e7776a",
        title_hint="Kết quả xét tuyển đại học chính quy năm 2025",
    ),
    SourceSeed(
        category=CATEGORY,
        subcategory="cutoff_results_early",
        source_name="Trường Đại học Công nghiệp Hà Nội",
        source_type=SourceType.OFFICIAL_WEBSITE.value,
        authority="Hội đồng tuyển sinh Trường Đại học Công nghiệp Hà Nội",
        university="HaUI",
        document_type="Thông báo",
        url="https://tuyensinh.haui.edu.vn/diem-chuan-trung-tuyen-dai-hoc/ket-qua-xet-tuyen-dai-hoc-chinh-quy-nam-2024--theo-cac-phuong-thuc-xet-tuyen-som/667251809a9cfd1248b9e33b",
        title_hint="Kết quả xét tuyển đại học chính quy năm 2024 theo các phương thức xét tuyển sớm",
    ),
    SourceSeed(
        category=CATEGORY,
        subcategory="cutoff_results",
        source_name="Trường Đại học Công nghiệp Hà Nội",
        source_type=SourceType.OFFICIAL_WEBSITE.value,
        authority="Hội đồng tuyển sinh Trường Đại học Công nghiệp Hà Nội",
        university="HaUI",
        document_type="Thông báo",
        url="https://tuyensinh.haui.edu.vn/diem-chuan-trung-tuyen-dai-hoc/ket-qua-xet-tuyen-dai-hoc-chinh-quy-nam-2024/66c0d1303a85982920fb9380",
        title_hint="Kết quả xét tuyển Đại học chính quy năm 2024",
    ),
    SourceSeed(
        category=CATEGORY,
        subcategory="cutoff_results_early",
        source_name="Trường Đại học Công nghiệp Hà Nội",
        source_type=SourceType.OFFICIAL_WEBSITE.value,
        authority="Hội đồng tuyển sinh Trường Đại học Công nghiệp Hà Nội",
        university="HaUI",
        document_type="Thông báo",
        url="https://tuyensinh.haui.edu.vn/diem-chuan-trung-tuyen-dai-hoc/ket-qua-xet-tuyen-dai-hoc-chinh-quy-nam-2023-theo-cac-phuong-thuc-2,-4,-5,-6-/649eb9a218495549ec635a3a",
        title_hint="Kết quả xét tuyển đại học chính quy năm 2023 theo các phương thức 2, 4, 5, 6",
    ),
    SourceSeed(
        category=CATEGORY,
        subcategory="cutoff_results",
        source_name="Trường Đại học Công nghiệp Hà Nội",
        source_type=SourceType.OFFICIAL_WEBSITE.value,
        authority="Hội đồng tuyển sinh Trường Đại học Công nghiệp Hà Nội",
        university="HaUI",
        document_type="Thông báo",
        url="https://tuyensinh.haui.edu.vn/diem-chuan-trung-tuyen-dai-hoc/ket-qua-xet-tuyen-dai-hoc-chinh-quy-nam-2023/64e775bd18495549ec635a50",
        title_hint="Kết quả xét tuyển đại học chính quy năm 2023",
    ),
    SourceSeed(
        category=CATEGORY,
        subcategory="cutoff_results",
        source_name="Trường Đại học Công nghiệp Hà Nội",
        source_type=SourceType.OFFICIAL_WEBSITE.value,
        authority="Hội đồng tuyển sinh Trường Đại học Công nghiệp Hà Nội",
        university="HaUI",
        document_type="Thông báo",
        url="https://tuyensinh.haui.edu.vn/diem-chuan-trung-tuyen-dai-hoc/diem-chuan-trung-tuyen-cac-phuong-thuc-xet-tuyen-dai-hoc-chinh-quy-nam-2022/6407e983ed3e304aa4d6de71",
        title_hint="Điểm chuẩn trúng tuyển các phương thức xét tuyển đại học chính quy năm 2022",
    ),
]


def build_image_notice_content(raw_html: bytes, base_url: str, title: str) -> str:
    """Keep a pointer to official scan images when the page body has no extractable table."""
    soup = BeautifulSoup(raw_html, "lxml")
    container = soup.select_one(".news-details-content") or soup.select_one(".single-latest-text") or soup
    image_urls: list[str] = []
    for img in container.find_all("img"):
        src = img.get("src")
        if not src:
            continue
        if any(token in src.lower() for token in ["logo", "banner", "facebook", "zalo", "tiktok"]):
            continue
        image_urls.append(urljoin(base_url, src))

    image_urls = list(dict.fromkeys(image_urls))
    if not image_urls:
        return ""

    lines = [
        f"### {title}",
        "",
        "Nội dung thông báo được Nhà trường công bố dưới dạng ảnh scan trên trang nguồn chính thức.",
        "Bản văn chưa được số hóa thành bảng điểm nên corpus chỉ lưu liên kết ảnh gốc để đối chiếu:",
    ]
    lines.extend(f"- {url}" for url in image_urls)
    return clean_text("\n".join(lines))


def process_cutoff_source(
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

    if len(content.strip()) < 500 or "Đơn vị Lao động Anh hùng thời kỳ đổi mới" in content:
        image_content = build_image_notice_content(raw_bytes, seed.url, seed.title_hint)
        if image_content:
            content = image_content

    # Clean out any image references if any lingered
    lines = [l for l in content.splitlines() if not l.strip().startswith("Ảnh đính kèm") and not l.strip().startswith("OCR tham khảo")]
    content = clean_text("\n".join(lines))

    doc_id = make_id(CATEGORY, index)
    doc_year = extract_year(title, content, seed.url) or 2026
    doc_date = extract_date(content[:1000])

    if "2026" in seed.title_hint:
        doc_year = 2026
    elif "2025" in seed.title_hint:
        doc_year = 2025
    elif "2024" in seed.title_hint:
        doc_year = 2024
    elif "2023" in seed.title_hint:
        doc_year = 2023
    elif "2022" in seed.title_hint:
        doc_year = 2022

    status = Status.ACTIVE.value if doc_year >= 2025 else Status.ARCHIVED.value

    doc_content = content if len(content.strip()) >= 50 else seed.title_hint

    return Document(
        id=doc_id,
        title=seed.title_hint,
        content=doc_content,
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
        document_number=seed.document_number,
        document_type=seed.document_type,
    )


def crawl_cutoff_scores() -> list[Document]:
    http_client = HttpClient()
    crawled_date = date.today().isoformat()
    documents: list[Document] = []

    print(f"[{CATEGORY}] Processing {len(SEEDS)} official cutoff score documents...")
    for idx, seed in enumerate(SEEDS, 1):
        try:
            doc = process_cutoff_source(seed, idx, http_client, crawled_date)
            documents.append(doc)
            print(f"  ✓ [{doc.id}] ({doc.year}) {doc.title[:75]}")
        except Exception as exc:
            print(f"  ✗ Failed {seed.url}: {exc}")

    write_jsonl(documents, OUTPUT_FILE)
    print(f"[{CATEGORY}] Wrote {len(documents)} clean documents to {OUTPUT_FILE}")
    return documents


if __name__ == "__main__":
    crawl_cutoff_scores()
