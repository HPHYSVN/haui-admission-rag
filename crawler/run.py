#!/usr/bin/env python3
"""Main crawler pipeline runner for HaUI Admission RAG corpus.

Usage:
  python crawler/run.py --all
  python crawler/run.py --category 01_admission
  python crawler/run.py --category 02_ministry_regulations
  python crawler/run.py --category 03_cutoff_scores
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MODULE_MAPPING = {
    "01_admission": ("crawler.01_admission.crawl", "crawl_admission"),
    "02_ministry_regulations": ("crawler.02_ministry_regulations.crawl", "crawl_ministry_regulations"),
    "03_cutoff_scores": ("crawler.03_cutoff_scores.crawl", "crawl_cutoff_scores"),
}


def get_runner(category: str):
    mod_name, func_name = MODULE_MAPPING[category]
    mod = importlib.import_module(mod_name)
    return getattr(mod, func_name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HaUI admission corpus crawlers.")
    parser.add_argument(
        "--category",
        choices=list(MODULE_MAPPING.keys()),
        help="Run crawler for a specific category",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all implemented category crawlers",
    )

    args = parser.parse_args()

    if not args.category and not args.all:
        print("[INFO] Defaulting to running all implemented crawlers (--all):")
        args.all = True

    categories_to_run = list(MODULE_MAPPING.keys()) if args.all else [args.category]

    total_docs = 0
    print("========================================")
    print("   STARTING HAUI CORPUS CRAWLER RUN     ")
    print("========================================")

    for cat in categories_to_run:
        runner = get_runner(cat)
        docs = runner()
        total_docs += len(docs)
        print("----------------------------------------")

    print(f"PIPELINE COMPLETED: Total {total_docs} documents processed across {len(categories_to_run)} categories.")
    print("Next step: Validate dataset using:")
    print("  uv run python scripts/validate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
