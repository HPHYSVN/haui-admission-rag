"""Integrity checks for the grounded Ragas evaluation dataset."""

from __future__ import annotations

import json
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.eligibility import DEFAULT_ELIGIBILITY_FILE, load_eligibility_policy

QUESTIONS_FILE = ROOT / "evaluation" / "questions.jsonl"
CHUNKS_DIR = ROOT / "data" / "chunks" / "final"
CATEGORIES = tuple(f"{index:02d}" for index in range(1, 15))
QUESTIONS_PER_CATEGORY = 10


def _load_questions() -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(
        QUESTIONS_FILE.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise AssertionError(f"line {line_number} is not a JSON object")
        rows.append(payload)
    return rows


def _load_documents() -> dict[str, dict]:
    documents: dict[str, dict] = {}
    for path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            chunk = json.loads(line)
            if chunk.get("status") != "final":
                continue
            document_id = chunk.get("document_id")
            if document_id and document_id not in documents:
                documents[document_id] = chunk
    return documents


class EvaluationDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.questions = _load_questions()
        cls.documents = _load_documents()
        cls.policy = load_eligibility_policy(DEFAULT_ELIGIBILITY_FILE)

    def test_dataset_has_ten_grounded_questions_per_category(self) -> None:
        counts = Counter(question["dataset"] for question in self.questions)
        self.assertEqual(len(self.questions), len(CATEGORIES) * QUESTIONS_PER_CATEGORY)
        self.assertEqual(set(counts), set(CATEGORIES))
        for category in CATEGORIES:
            self.assertEqual(
                counts[category],
                QUESTIONS_PER_CATEGORY,
                f"category {category} has {counts[category]} questions",
            )

    def test_question_ids_are_unique_and_match_their_category(self) -> None:
        ids = [question["id"] for question in self.questions]
        self.assertEqual(len(ids), len(set(ids)))
        for question in self.questions:
            self.assertTrue(
                question["id"].startswith(f"{question['dataset']}-"),
                question["id"],
            )

    def test_questions_are_unique(self) -> None:
        normalized = [
            " ".join(question["question"].casefold().split())
            for question in self.questions
        ]
        duplicates = [
            text for text, count in Counter(normalized).items() if count > 1
        ]
        self.assertEqual(duplicates, [])

    def test_expected_documents_exist_and_are_not_quarantined(self) -> None:
        for question in self.questions:
            document_ids = question.get("expected_document_ids")
            self.assertIsInstance(document_ids, list, question["id"])
            self.assertTrue(document_ids, question["id"])
            for document_id in document_ids:
                self.assertIn(document_id, self.documents, question["id"])
                override = self.policy.documents.get(document_id, {})
                self.assertNotEqual(
                    override.get("scope"),
                    "quarantine",
                    f"{question['id']} targets quarantined {document_id}",
                )

    def test_expected_year_is_valid(self) -> None:
        for question in self.questions:
            year = question.get("expected_year")
            if year is None:
                continue
            self.assertIsInstance(year, int, question["id"])
            self.assertFalse(isinstance(year, bool), question["id"])
            self.assertGreaterEqual(year, 1990, question["id"])
            self.assertLessEqual(year, 2100, question["id"])

    def test_expected_year_matches_eligible_document_metadata(self) -> None:
        for question in self.questions:
            year = question.get("expected_year")
            if year is None:
                continue
            for document_id in question["expected_document_ids"]:
                document = self.documents[document_id]
                override = self.policy.documents.get(document_id, {})
                scope = override.get("scope") or self.policy.defaults.get(
                    str(document.get("category") or ""), {}
                ).get("scope", "evergreen")
                if override.get("year") is not None:
                    self.assertEqual(
                        override["year"], year, f"{question['id']} / {document_id}"
                    )
                elif scope == "regulation" and override.get("effective_from"):
                    effective_from = int(override["effective_from"][:4])
                    effective_to = override.get("effective_to")
                    effective_to_year = int(effective_to[:4]) if effective_to else 2100
                    self.assertGreaterEqual(
                        year, effective_from, f"{question['id']} / {document_id}"
                    )
                    self.assertLessEqual(
                        year, effective_to_year, f"{question['id']} / {document_id}"
                    )
                elif scope == "evergreen":
                    document_year = document.get("year")
                    if document_year is not None:
                        self.assertEqual(
                            document_year, year, f"{question['id']} / {document_id}"
                        )
                else:
                    self.assertEqual(
                        document.get("year"), year, f"{question['id']} / {document_id}"
                    )

    def test_dataset_covers_a_broad_set_of_source_documents(self) -> None:
        sources: dict[str, set[str]] = defaultdict(set)
        for question in self.questions:
            sources[question["dataset"]].update(question["expected_document_ids"])
        for category in CATEGORIES:
            self.assertTrue(
                sources[category],
                f"category {category} has no grounded source document",
            )
        self.assertGreaterEqual(len(set().union(*sources.values())), 50)


if __name__ == "__main__":
    unittest.main()
