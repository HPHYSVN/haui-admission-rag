"""Evaluate basic retrieval coverage with human-authored test questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.bm25 import HybridRetriever
from rag.config import DEFAULT_TOP_K, DEFAULT_VECTOR_INDEX_DIR
from rag.pipeline import DEFAULT_CHUNKS_DIR, load_chunks


def load_cases(test_file: Path) -> list[dict]:
    try:
        content = test_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot load test cases from {test_file}: {exc}") from exc
    try:
        cases = json.loads(content)
    except json.JSONDecodeError:
        cases = []
        for line_no, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{test_file}:{line_no}: invalid JSON: {exc}") from exc
    if not isinstance(cases, list) or not cases:
        raise ValueError("Test question file must contain a non-empty JSON array")
    if not all(isinstance(case, dict) for case in cases):
        raise ValueError("Every evaluation case must be a JSON object")
    return cases


def evaluate(
    test_file: Path,
    chunks_dir: Path,
    top_k: int,
    index_dir: Path = DEFAULT_VECTOR_INDEX_DIR,
) -> tuple[int, int]:
    cases = load_cases(test_file)
    chunks = load_chunks(chunks_dir)
    retriever = HybridRetriever(chunks, index_dir=index_dir)
    passed = 0
    for case in cases:
        question = case.get("question")
        expected_ids = case.get("expected_document_ids")
        if expected_ids is None and isinstance(case.get("expected_document_id"), str):
            expected_ids = [case["expected_document_id"]]
        expected_chunks = case.get("expected_chunk_ids", [])
        expected_year = case.get("expected_year")
        if (
            not isinstance(question, str)
            or not question.strip()
            or not isinstance(expected_ids, list)
            or not all(isinstance(item, str) for item in expected_ids)
            or not isinstance(expected_chunks, list)
            or not all(isinstance(item, str) for item in expected_chunks)
            or (
                expected_year is not None
                and (not isinstance(expected_year, int) or isinstance(expected_year, bool))
            )
        ):
            raise ValueError(f"Invalid test case: {case!r}")
        results = retriever.search(question, top_k=top_k)
        retrieved_ids = [chunk["document_id"] for _, chunk in results]
        retrieved_chunk_ids = [chunk["chunk_id"] for _, chunk in results]
        document_match = (
            not retrieved_ids
            if not expected_ids
            else any(item in retrieved_ids for item in expected_ids)
        )
        chunk_match = (
            not expected_chunks
            or any(item in retrieved_chunk_ids for item in expected_chunks)
        )
        year_match = (
            expected_year is None
            or not results
            or all(chunk.get("year") == expected_year for _, chunk in results)
        )
        success = document_match and chunk_match and year_match
        passed += int(success)
        mark = "PASS" if success else "FAIL"
        print(f"[{mark}] {question}")
        print(
            f"  expected_documents={expected_ids}; expected_chunks={expected_chunks}; "
            f"retrieved_documents={retrieved_ids}; retrieved_chunks={retrieved_chunk_ids}"
        )
    print(f"\nRecall@{top_k}: {passed}/{len(cases)} ({passed / len(cases):.1%})")
    return passed, len(cases)


def main() -> int:
    parser = argparse.ArgumentParser()
    default_questions = Path(__file__).resolve().parents[1] / "evaluation" / "questions.jsonl"
    parser.add_argument("--questions", type=Path, default=default_questions)
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_VECTOR_INDEX_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    try:
        passed, total = evaluate(args.questions, args.chunks_dir, args.top_k, args.index_dir)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Evaluation error: {exc}\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
