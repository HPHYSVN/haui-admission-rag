"""Time-window eligibility policy for the admission corpus."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.eligibility import (
    DEFAULT_ELIGIBILITY_FILE,
    EligibilityError,
    apply_eligibility,
    filter_chunks,
    has_current_evidence,
    is_eligible,
    load_eligibility_policy,
    resolve_query_scope,
)


def _policy(root: Path, documents: dict, defaults: dict | None = None):
    path = root / "eligibility.json"
    path.write_text(
        json.dumps(
            {
                "default_year": 2026,
                "defaults": defaults or {},
                "documents": documents,
            }
        ),
        encoding="utf-8",
    )
    return load_eligibility_policy(path)


class PolicyLoadingTests(unittest.TestCase):
    def test_missing_policy_is_not_silently_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(EligibilityError):
                load_eligibility_policy(Path(temporary) / "missing.json")

    def test_invalid_scope_and_dates_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(EligibilityError):
                _policy(root, {"d1": {"scope": "current"}})
            with self.assertRaises(EligibilityError):
                _policy(
                    root,
                    {
                        "d1": {
                            "scope": "regulation",
                            "effective_from": "2026-12-01",
                            "effective_to": "2026-01-01",
                        }
                    },
                )

    def test_repository_policy_is_valid_and_declares_the_current_cycle(self) -> None:
        policy = load_eligibility_policy(DEFAULT_ELIGIBILITY_FILE)
        self.assertEqual(policy.default_year, 2026)
        self.assertIn("01_admission", policy.defaults)
        self.assertTrue(policy.documents)

    def test_document_override_wins_over_category_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            policy = _policy(
                Path(temporary),
                {"draft": {"scope": "quarantine", "note": "not issued"}},
                defaults={"02_ministry_regulations": {"scope": "regulation"}},
            )
            document = {"id": "draft", "category": "02_ministry_regulations"}
            segment = policy.segment(document)
        self.assertEqual(segment["eligibility_scope"], "quarantine")


class QueryScopeTests(unittest.TestCase):
    def test_explicit_year_always_wins(self) -> None:
        scope = resolve_query_scope("Điểm chuẩn năm 2024 ngành Kỹ thuật phần mềm")
        self.assertEqual(scope.target_year, 2024)
        self.assertEqual(scope.explicit_year, 2024)
        self.assertTrue(scope.year_sensitive)

    def test_year_sensitive_question_defaults_to_2026(self) -> None:
        scope = resolve_query_scope("Học phí học kỳ này là bao nhiêu?")
        self.assertEqual(scope.target_year, 2026)
        self.assertIsNone(scope.explicit_year)

    def test_evergreen_question_is_not_pinned_to_2026(self) -> None:
        scope = resolve_query_scope("Ký túc xá có chỗ gửi xe không?")
        self.assertIsNone(scope.target_year)
        self.assertFalse(scope.year_sensitive)

    def test_policy_default_year_is_configurable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "eligibility.json"
            path.write_text(
                json.dumps({"default_year": 2025, "defaults": {}, "documents": {}}),
                encoding="utf-8",
            )
            policy = load_eligibility_policy(path)
            scope = resolve_query_scope("Tuyển sinh năm nay?", policy=policy)
        self.assertEqual(scope.target_year, 2025)


class EligibilityRuleTests(unittest.TestCase):
    def _scope(self, year: int | None):
        return resolve_query_scope(f"điểm chuẩn {year}" if year else "điểm chuẩn")

    def test_cycle_chunks_only_answer_their_own_year(self) -> None:
        scope = self._scope(2026)
        self.assertFalse(
            is_eligible({"year": 2025, "eligibility_scope": "cycle"}, scope)
        )
        self.assertTrue(
            is_eligible({"year": 2026, "eligibility_scope": "cycle"}, scope)
        )

    def test_unknown_year_cycle_chunk_is_not_current_evidence_but_stays_retrievable(self) -> None:
        scope = self._scope(2026)
        chunk = {"eligibility_scope": "cycle"}
        self.assertTrue(is_eligible(chunk, scope))
        self.assertFalse(has_current_evidence([(1.0, chunk)], 2026))

    def test_evergreen_is_allowed_but_never_current_evidence(self) -> None:
        scope = self._scope(2026)
        chunk = {"year": 2019, "eligibility_scope": "evergreen"}
        self.assertTrue(is_eligible(chunk, scope))
        self.assertFalse(has_current_evidence([(1.0, chunk)], 2026))

    def test_regulation_is_limited_to_its_effective_range(self) -> None:
        scope = self._scope(2026)
        chunk = {
            "year": 2024,
            "eligibility_scope": "regulation",
            "eligibility_from": "2024-03-01",
            "eligibility_to": "2027-02-28",
        }
        self.assertTrue(is_eligible(chunk, scope))
        self.assertTrue(has_current_evidence([(1.0, chunk)], 2026))
        self.assertFalse(is_eligible(chunk, self._scope(2028)))

    def test_quarantine_is_never_eligible(self) -> None:
        chunk = {"year": 2026, "eligibility_scope": "quarantine"}
        for year in (2024, 2025, 2026):
            self.assertFalse(is_eligible(chunk, self._scope(year)))
        self.assertFalse(has_current_evidence([(1.0, chunk)], 2026))
        self.assertEqual(filter_chunks([chunk], self._scope(2026)), [])

    def test_yearless_question_does_not_filter_but_still_hides_quarantine(self) -> None:
        scope = resolve_query_scope("Ký túc xá ở đâu?")
        chunks = [
            {"id": "old", "year": 2019, "eligibility_scope": "cycle"},
            {"id": "draft", "year": 2026, "eligibility_scope": "quarantine"},
        ]
        self.assertEqual([chunk["id"] for chunk in filter_chunks(chunks, scope)], ["old"])

    def test_apply_eligibility_returns_copies_without_mutating_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            policy = _policy(
                Path(temporary),
                {},
                defaults={"03_cutoff_scores": {"scope": "cycle"}},
            )
            original = {"document_id": "d1", "category": "03_cutoff_scores", "year": 2025}
            decorated = apply_eligibility([original], policy)
        self.assertEqual(decorated[0]["eligibility_scope"], "cycle")
        self.assertNotIn("eligibility_scope", original)


if __name__ == "__main__":
    unittest.main()
