from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pdfplumber
from bs4 import BeautifulSoup, NavigableString, Tag

from crawler.common.cleaner import clean_text, is_irrelevant_table


def parse_html_table(table: Tag) -> str:
    """Converts an HTML table into a semantic key-value or structured text representation.

    Follows Rule 9: Preserve relationships between row and column headers.
    """
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [clean_text(td.get_text(" ", strip=True)) for td in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(cells)

    if not rows:
        return ""

    header = rows[0]
    header_str = " ".join(header)
    remaining_rows_str = " ".join(" ".join(r) for r in rows[1:5])

    # Check if this table is irrelevant facility equipment inventory
    if is_irrelevant_table(header_str, remaining_rows_str):
        return ""

    lines = ["\n[Bảng biểu]:"]
    has_valid_header = len(header) > 1 and any(len(h) > 1 for h in header)

    for r_idx, cells in enumerate(rows[1:] if has_valid_header and len(rows) > 1 else rows):
        if has_valid_header and len(header) == len(cells):
            # Pair each column header with its cell value
            parts = []
            for h, c in zip(header, cells):
                h_clean = h.strip()
                c_clean = c.strip()
                if not h_clean and not c_clean:
                    continue
                if h_clean:
                    parts.append(f"{h_clean}: {c_clean}")
                else:
                    parts.append(c_clean)
            lines.append("- " + " | ".join(parts))
        else:
            lines.append("- " + " | ".join(c for c in cells if c))

    return "\n".join(lines) + "\n"


def extract_meaningful_container(soup: BeautifulSoup) -> Tag:
    """Finds the main content element and discards navigation/layout wrapper."""
    selectors = [
        ".news-details-content .single-latest-text",
        ".news-details-content",
        ".admission-project__content-page",
        ".content-page__head_content",
        ".new-detail",
        "article",
        "main",
        ".main-content-area",
    ]
    for selector in selectors:
        found = soup.select_one(selector)
        if found and len(clean_text(found.get_text())) > 100:
            return found
    return soup.body or soup


def parse_html_document(raw_html: bytes) -> tuple[str, str]:
    """Parses raw HTML bytes, cleanly strips junk, menus, ads, headers, footers.

    Returns: (title, cleaned_content)
    """
    soup = BeautifulSoup(raw_html, "lxml")

    # Remove all unwanted elements
    for tag_name in [
        "script",
        "style",
        "noscript",
        "iframe",
        "svg",
        "form",
        "nav",
        "header",
        "footer",
        "aside",
    ]:
        for node in soup.find_all(tag_name):
            node.decompose()

    # Remove widgets, ads, sliders, comments, share buttons, thumbnails
    for selector in [
        ".sidebar-widget",
        ".tags-and-links",
        ".social-links",
        ".fb-page",
        ".fb-like",
        ".fb-save",
        ".slider-area",
        ".header-two",
        ".mobile-menu-area",
        ".single-sidebar-widget",
        ".recent-content",
        ".related-tag",
    ]:
        for node in soup.select(selector):
            node.decompose()

    # Find title
    title = ""
    h3 = soup.select_one(".single-latest-text h3")
    if h3:
        title = clean_text(h3.get_text())
    elif soup.find("h1"):
        title = clean_text(soup.find("h1").get_text())
    elif soup.title:
        title = clean_text(soup.title.get_text())
        # Remove site branding from title
        title = re.sub(r"\s*[-|]\s*Cổng thông tin tuyển sinh.*$", "", title, flags=re.IGNORECASE)
        title = re.sub(r"\s*[-|]\s*Trường Đại học Công nghiệp Hà Nội.*$", "", title, flags=re.IGNORECASE)

    container = extract_meaningful_container(soup)

    # Process and convert tables inside container
    for table in container.find_all("table"):
        table_text = parse_html_table(table)
        table_node = soup.new_tag("div")
        if table_text.strip():
            table_node.string = table_text.strip()
            table.replace_with(table_node)
        else:
            table.decompose()

    # Remove all image tags inside container (No junk images or logos!)
    for img in container.find_all("img"):
        img.decompose()

    blocks: list[str] = []
    for elem in container.find_all(["h1", "h2", "h3", "h4", "h5", "p", "li", "blockquote", "div"]):
        # Skip divs that contain other block elements to avoid duplicate nested text
        if elem.name == "div" and elem.find(["p", "div", "h1", "h2", "h3", "h4", "h5", "ul", "ol"]):
            continue
        text = clean_text(elem.get_text())
        if not text:
            continue

        if elem.name in {"h1", "h2", "h3", "h4", "h5"}:
            blocks.append(f"\n### {text}")
        elif elem.name == "li":
            blocks.append(f"- {text}")
        else:
            blocks.append(text)

    if not blocks:
        content = clean_text(container.get_text("\n"))
    else:
        content = "\n\n".join(blocks)

    # Final cleanup of excessive markers or empty lines
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = clean_text(content)

    return title, content


def parse_pdf_document(pdf_path: Path) -> tuple[str, str]:
    """Extracts text and structured tables from a PDF file using pdfplumber."""
    pages_text: list[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, 1):
            page_blocks: list[str] = []
            text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
            text = clean_text(text)
            if text:
                page_blocks.append(f"[Trang {page_idx}]\n{text}")

            tables = page.extract_tables() or []
            for t in tables:
                rows = [[clean_text(cell or "") for cell in row] for row in t if any(row)]
                if not rows:
                    continue
                header = rows[0]
                lines = [f"\n[Bảng biểu từ trang {page_idx}]:"]
                for r in rows[1:] if len(rows) > 1 else rows:
                    if len(r) == len(header) and len(rows) > 1:
                        parts = [f"{h}: {c}" for h, c in zip(header, r) if h or c]
                        lines.append("- " + " | ".join(parts))
                    else:
                        lines.append("- " + " | ".join(c for c in r if c))
                page_blocks.append("\n".join(lines))

            if page_blocks:
                pages_text.append("\n\n".join(page_blocks))

    full_content = "\n\n".join(pages_text)
    full_content = clean_text(full_content)

    # Extract title from the beginning lines
    lines = [l.strip() for l in full_content.splitlines() if len(l.strip()) > 10]
    title = ""
    for line in lines[:5]:
        if not any(token in line.upper() for token in ["BỘ GIÁO DỤC", "CỘNG HÒA", "ĐỘC LẬP"]):
            title = line
            break
    if not title and lines:
        title = lines[0]

    return title[:180], full_content
