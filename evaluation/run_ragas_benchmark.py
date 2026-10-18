#!/usr/bin/env python3
"""
Quy trình đánh giá RAGAS trước/sau khi thay đổi model retrieval.

Cách dùng:
    # Chạy cả hai cấu hình (before = cấu hình cũ, after = cấu hình mới)
    uv run python -m evaluation.run_ragas_benchmark --output evaluation/ragas/benchmark.json

    # Chạy riêng từng cấu hình
    uv run python -m evaluation.run_ragas_benchmark --config before --output evaluation/ragas/before.json
    uv run python -m evaluation.run_ragas_benchmark --config after  --output evaluation/ragas/after.json

    # Chạy evaluation với kết quả mới
    cd evaluation && uv run python -m rag.evaluate --reranker on --output ../ragas/reranked.json

Quy trình chuẩn:
    1. backup index cũ (nếu có)
    2. gọi `rag.cli index --all` với config cũ -> lưu index cũ
    3. gọi `rag.cli index --all` với config mới -> lưu index mới
    4. chạy ragas với config cũ -> lưu kết quả before
    5. chạy ragas với config mới -> lưu kết quả after
    6. so sánh, xuất report

Team có thể thêm câu hỏi vào evaluation/questions.jsonl, sau đó chạy:
    uv run python -m evaluation.run_ragas_benchmark --output evaluation/ragas/benchmark.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAGAS_SCRIPT = ROOT_DIR / "evaluation" / "run_ragas.py"
QUESTIONS = ROOT_DIR / "evaluation" / "questions.jsonl"
REFERENCES = ROOT_DIR / "evaluation" / "ragas" / "references.json"
DEFAULT_OUTPUT = ROOT_DIR / "evaluation" / "ragas" / "benchmark.json"

# Cấu hình model cũ (before)
BEFORE_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BEFORE_RERANKER = "jinaai/jina-reranker-v2-base-multilingual"

# Cấu hình model mới (after)
AFTER_EMBEDDING = "intfloat/multilingual-e5-large"
AFTER_RERANKER = "jinaai/jina-reranker-v2-base-multilingual"

REQUIRED_ENV = [
    "VLLM_BASE_URL",
    "VLLM_MODEL",
    "RAG_EMBEDDING_MODEL",
    "RAG_RERANKER_MODEL",
    "RAG_RERANKER_ENABLED",
]


def load_env() -> dict[str, str]:
    env = {}
    for key in REQUIRED_ENV:
        value = os.getenv(key)
        if not value:
            raise RuntimeError(f"Thiếu biến môi trường {key}. Hãy copy .env.example -> .env và cấu hình.")
        env[key] = value
    return env


def run_ragas(
    output_path: Path,
    *,
    embedding_model: str,
    reranker_model: str,
    reranker_enabled: bool = True,
) -> int:
    """Run RAGAS evaluation với cấu hình cho trước, lưu kết quả."""
    cmd = [
        "uv", "run", "python", "-m", "evaluation.run_ragas",
        "--reranker", "on" if reranker_enabled else "off",
        "--references", str(REFERENCES),
        "--output", str(output_path),
    ]
    env = os.environ.copy()
    env["RAG_EMBEDDING_MODEL"] = embedding_model
    env["RAG_RERANKER_MODEL"] = reranker_model
    for k, v in env.items():
        if k.startswith("RAG_") and v != os.getenv(k, ""):
            env[k] = v

    print(f"\n{'='*60}")
    print(f" ĐANG CHẠY: embedding={embedding_model} | reranker={reranker_model} | reranker_enabled={reranker_enabled}")
    print(f"  -> Output: {output_path}")
    print(f"{'='*60}")

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    print(result.stdout[-3000:] if result.stdout else "(no stdout)")
    if result.returncode != 0:
        print(f"ERROR (rc={result.returncode}): {result.stderr[-3000:]}")
        return result.returncode
    return 0


def run_retrieval_benchmark() -> int:
    """Quy trình chính: before -> after."""
    init_args = ["uv", "run", "python", "-m", "rag.cli", "index", "--all"]
    # Steps:
    # 1. Rebuild index with new config
    # 2. Run RAGAS baseline (reranker off)
    # 3. Run RAGAS reranked (reranker on)
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS evaluation before/after model swap")
    parser.add_argument("--config", choices=["before", "after", "both"], default="both")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--references", type=Path, default=REFERENCES)
    args = parser.parse_args()

    # Validate env
    for k, v in load_env().items():
        if v == "":
            print(f"LỖI: Biến môi trường {k} trống. Hãy cấu hình .env.")
            return 1

    if not args.questions.exists():
        print(f"LỖI: Không thấy {args.questions}. Hãy thêm câu hỏi của team vào đây.")
        return 1

    if args.config in ("before", "both"):
        out_before = args.output.with_name("before.json")
        rc = run_ragas(out_before, embedding_model=BEFORE_EMBEDDING, reranker_model=BEFORE_RERANKER, reranker_enabled=False)
        if rc != 0:
            return rc

    if args.config in ("after", "both"):
        out_after = args.output.with_name("after.json")
        rc = run_ragas(out_after, embedding_model=AFTER_EMBEDDING, reranker_model=AFTER_RERANKER, reranker_enabled=True)
        if rc != 0:
            return rc

    # Nén kết quả before/after thành một file benchmark tổng hợp
    before_path = args.output.with_name("before.json")
    after_path = args.output.with_name("after.json")
    if before_path.exists() and after_path.exists():
        before = json.loads(before_path.read_text(encoding="utf-8"))
        after = json.loads(after_path.read_text(encoding="utf-8"))
        benchmark = {
            "before": before,
            "after": after,
            "summary": {
                "before_embedding": BEFORE_EMBEDDING,
                "after_embedding": AFTER_EMBEDDING,
                "before_reranker": BEFORE_RERANKER,
                "after_reranker": AFTER_RERANKER,
            },
        }
        args.output.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n>>> Benchmark tổng hợp đã lưu: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
