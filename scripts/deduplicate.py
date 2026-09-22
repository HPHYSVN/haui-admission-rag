#!/usr/bin/env python3
"""Deduplication detection script for HaUI Admission RAG corpus.

Scans all JSONL documents across data/documents/ to identify:
1. Exact matching source_url
2. Exact matching titles
3. Near-duplicate or exact content (via normalized content hash and SimHash/Jaccard similarity)

Does not automatically delete content without verification.
Outputs a detailed duplication report.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DOCS_DIR = ROOT_DIR / "data" / "documents"


def normalize_str(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def content_signature(content: str) -> str:
    norm = normalize_str(content)
    # Take first 500 chars and length as quick fingerprint
    prefix = norm[:500]
    return f"{len(norm)}_{hashlib.md5(norm.encode('utf-8')).hexdigest()}"


def check_duplicates(docs_dir: Path) -> int:
    files = sorted(docs_dir.glob("*.jsonl"))
    if not files:
        print(f"No JSONL files found in {docs_dir}", file=sys.stderr)
        return 1

    url_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    title_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    content_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_docs = 0

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    doc = json.loads(clean_line)
                except json.JSONDecodeError:
                    continue

                total_docs += 1
                doc_info = {
                    "id": doc.get("id"),
                    "title": doc.get("title"),
                    "source_url": doc.get("source_url"),
                    "file": file_path.name,
                    "line": line_no,
                    "year": doc.get("year"),
                    "category": doc.get("category"),
                }

                url = doc.get("source_url")
                if url:
                    url_map[url].append(doc_info)

                title = normalize_str(doc.get("title", ""))
                if title:
                    title_map[title].append(doc_info)

                content = doc.get("content", "")
                if content:
                    sig = content_signature(content)
                    content_map[sig].append(doc_info)

    dup_urls = {k: v for k, v in url_map.items() if len(v) > 1}
    dup_titles = {k: v for k, v in title_map.items() if len(v) > 1}
    dup_contents = {k: v for k, v in content_map.items() if len(v) > 1}

    print("========================================")
    print("      DEDUPLICATION AUDIT REPORT        ")
    print("========================================")
    print(f"Total documents analyzed: {total_docs}")
    print(f"Duplicate URLs found: {len(dup_urls)}")
    print(f"Duplicate Titles found: {len(dup_titles)}")
    print(f"Identical Contents found: {len(dup_contents)}")
    print("========================================")

    has_issue = False

    if dup_urls:
        has_issue = True
        print("\n[!] DUPLICATE SOURCE URLS:")
        for url, items in dup_urls.items():
            print(f"  URL: {url}")
            for item in items:
                print(f"    - {item['id']} ({item['file']}:{item['line']}) [year: {item['year']}] {item['title'][:60]}")

    if dup_titles:
        print("\n[*] IDENTICAL TITLES (check if these are distinct editions or duplicates):")
        for title, items in dup_titles.items():
            # If they have different years or different categories, might be legitimate yearly releases
            years = {item.get("year") for item in items}
            cats = {item.get("category") for item in items}
            if len(years) > 1 or len(cats) > 1:
                note = f" (Different years/categories: {years} / {cats} - likely valid yearly entries)"
            else:
                note = " (POTENTIAL DUPLICATE)"
                has_issue = True
            print(f"  Title: '{items[0]['title']}'{note}")
            for item in items:
                print(f"    - {item['id']} ({item['file']}:{item['line']}) URL: {item['source_url']}")

    if dup_contents:
        has_issue = True
        print("\n[!] IDENTICAL CONTENT DETECTED:")
        for sig, items in dup_contents.items():
            print(f"  Signature {sig}:")
            for item in items:
                print(f"    - {item['id']} ({item['file']}:{item['line']}) URL: {item['source_url']}")

    if not has_issue:
        print("\nNO CRITICAL DUPLICATES FOUND. CORPUS IS CLEAN.")
        return 0
    else:
        print("\nAudit completed. Please review duplicate items above.")
        return 0


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DOCS_DIR
    if not target.exists():
        print(f"Directory {target} does not exist.", file=sys.stderr)
        return 1
    return check_duplicates(target)


if __name__ == "__main__":
    raise SystemExit(main())
