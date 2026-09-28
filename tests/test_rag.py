from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.apply_review import apply_reviews, read_review
from scripts.chunk import chunk_document, split_body
from rag.bm25 import BM25Retriever, HybridRetriever, tokenize
from rag.chunk_review import finalize_chunks, review_chunks
from rag.document_workflow import get_state, review_document, set_state
from rag.embeddings import VectorIndex
from rag.filters import matching_documents, query_filters, query_term_coverage
from rag.pipeline import RAGPipeline, VLLMChatClient, create_messages, load_chunks
from rag.review import (
    ReviewPaths,
    pending_chunks,
    pending_documents,
    progress_label,
    run_review,
)


class BM25Tests(unittest.TestCase):
    def test_tokenize_vietnamese_and_score_matching_document(self) -> None:
        docs = [
            {"text": "Học bổng mức 1 năm 2026", "document_id": "scholarship"},
            {"text": "Hướng dẫn nhập học đại học", "document_id": "enrollment"},
        ]
        results = BM25Retriever(docs).search("học bổng năm 2026", top_k=1)
        self.assertEqual(results[0][1]["document_id"], "scholarship")
        self.assertIn("học", tokenize("Học bổng"))

    def test_empty_query_returns_no_results(self) -> None:
        retriever = BM25Retriever([{"text": "Nội dung tài liệu", "document_id": "one"}])
        self.assertEqual(retriever.search("   "), [])

    def test_rejects_non_positive_top_k(self) -> None:
        retriever = BM25Retriever([{"text": "Nội dung tài liệu", "document_id": "one"}])
        with self.assertRaises(ValueError):
            retriever.search("tài liệu", top_k=0)


class PromptTests(unittest.TestCase):
    def test_prompt_preserves_provenance_without_asking_for_citation_tokens(self) -> None:
        messages = create_messages(
            "Hỏi về học bổng",
            [(1.2, {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "title": "Học bổng",
                "source_url": "https://example.org",
                "text": "Mức 1",
                "year": 2026,
                "category": "05_tuition_scholarship",
            })],
        )
        self.assertIn("không bịa", messages[0]["content"])
        self.assertIn("chunk_id: chunk-1", messages[1]["content"])
        self.assertIn("source_url: https://example.org", messages[1]["content"])
        self.assertNotIn("[Nguồn 1]", messages[0]["content"])

    def test_no_relevant_retrieval_skips_generation_and_returns_no_sources(self) -> None:
        client = Mock()
        with patch("rag.pipeline.HybridRetriever") as retriever_type:
            retriever_type.return_value.search.return_value = []
            pipeline = RAGPipeline(
                [{"chunk_id": "c1", "document_id": "d1", "text": "text"}],
                client,
            )
            response = pipeline.ask("question")
        self.assertEqual(response.answer, "Chưa tìm thấy tài liệu liên quan để trả lời câu hỏi này.")
        self.assertEqual(response.sources, [])
        client.answer.assert_not_called()

    def test_response_sources_are_constructed_from_retrieved_chunks(self) -> None:
        client = Mock()
        client.answer.return_value = "Trả lời"
        chunk = {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "title": "Title",
            "source_url": "https://example.org",
            "year": 2026,
            "category": "01_admission",
            "text": "Content",
        }
        with patch("rag.pipeline.HybridRetriever") as retriever_type:
            retriever_type.return_value.search.return_value = [(0.91, chunk)]
            response = RAGPipeline([chunk], client).ask("question")
        self.assertEqual(response.answer, "Trả lời")
        self.assertEqual(response.sources, [{
            "source_id": "doc-1",
            "chunk_id": "chunk-1",
            "title": "Title",
            "url": "https://example.org",
            "year": 2026,
            "category": "01_admission",
            "score": 0.91,
        }])


class RetrievalMetadataTests(unittest.TestCase):
    def test_explicit_year_is_a_strict_filter(self) -> None:
        chunks = [
            {"year": 2025, "category": "03_cutoff_scores"},
            {"year": 2026, "category": "03_cutoff_scores"},
        ]
        filters = query_filters("điểm chuẩn Kỹ thuật phần mềm năm 2026")
        self.assertEqual(filters, {"year": 2026, "category": "03_cutoff_scores"})
        self.assertEqual(matching_documents(chunks, filters), [chunks[1]])

    def test_unknown_requested_year_has_no_candidates(self) -> None:
        chunks = [{"year": 2025, "category": "03_cutoff_scores"}]
        self.assertEqual(
            matching_documents(chunks, query_filters("điểm chuẩn năm 2026")),
            [],
        )

    def test_irrelevant_terms_are_not_counted_as_evidence(self) -> None:
        query = "Chính sách hoàn tiền mua vé tàu vũ trụ cho sinh viên là gì?"
        generic_chunk = {
            "title": "Chính sách hỗ trợ sinh viên",
            "text": "Chính sách dành cho sinh viên HaUI.",
        }
        self.assertEqual(query_term_coverage(query, generic_chunk), 0.0)

    def test_semantic_similarity_alone_does_not_return_a_lexically_unrelated_source(self) -> None:
        class FakeEmbedder:
            def embed(self, texts):
                return [[1.0, 0.0] if text == "space refund policy" else [0.5, 0.866] for text in texts]

        chunks = [{
            "chunk_id": "support-1",
            "document_id": "support",
            "title": "Chính sách hỗ trợ sinh viên",
            "text": "Các chương trình hỗ trợ sinh viên HaUI.",
            "source_url": "https://example.org",
        }]
        with tempfile.TemporaryDirectory() as temporary:
            retriever = HybridRetriever(
                chunks,
                index_dir=Path(temporary),
                embedder=FakeEmbedder(),
            )
            self.assertEqual(retriever.search("space refund policy", top_k=3), [])


class ChunkingTests(unittest.TestCase):
    def test_long_sentence_is_not_cut_at_arbitrary_character_limit(self) -> None:
        sentence = "Không cắt một câu dài vì đã tới ngưỡng ký tự"
        self.assertEqual(split_body(sentence, max_chars=10), [sentence])

    def test_chunk_proposals_keep_source_and_structure_metadata(self) -> None:
        document = {
            "id": "01_admission_001",
            "title": "Thông tin tuyển sinh",
            "content": "### Phương thức tuyển sinh\n\n- Phương thức 1\n- Phương thức 2",
            "category": "01_admission",
            "year": 2026,
            "source_url": "https://example.org/admission",
        }
        chunk = chunk_document(document)[0]
        self.assertEqual(chunk["status"], "proposal")
        self.assertEqual(chunk["heading_path"], ["### Phương thức tuyển sinh"])
        self.assertEqual(chunk["content_type"], "list")
        self.assertEqual(chunk["source_url"], document["source_url"])
        self.assertIsNone(chunk["text_override"])


class FinalChunkLoadingTests(unittest.TestCase):
    def test_stale_final_chunks_are_excluded_after_document_reenters_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chunks_dir = root / "final"
            chunks_dir.mkdir()
            (chunks_dir / "category.jsonl").write_text(
                json.dumps({
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "title": "Title",
                    "text": "Final text",
                    "source_url": "https://example.org",
                    "status": "final",
                }) + "\n",
                encoding="utf-8",
            )
            state_file = root / "documents.json"
            state_file.write_text(
                json.dumps({"d1": {"status": "review_pending"}}),
                encoding="utf-8",
            )
            with patch("rag.pipeline.DEFAULT_STATE_FILE", state_file):
                with self.assertRaises(ValueError):
                    load_chunks(chunks_dir)


class VectorIndexTests(unittest.TestCase):
    def test_builds_and_updates_document_embeddings_without_model_dependency(self) -> None:
        class FakeEmbedder:
            def __init__(self):
                self.calls = []

            def embed(self, texts):
                self.calls.append(texts)
                return [[1.0, 0.0] if "học bổng" in text else [0.0, 1.0] for text in texts]

        with tempfile.TemporaryDirectory() as temporary:
            embedder = FakeEmbedder()
            index = VectorIndex(Path(temporary), model_name="fake-model", embedder=embedder)
            chunks = [
                {
                    "chunk_id": "scholarship-1",
                    "document_id": "scholarship",
                    "title": "Học bổng",
                    "text": "học bổng mức 1",
                },
                {
                    "chunk_id": "fees-1",
                    "document_id": "fees",
                    "title": "Học phí",
                    "text": "mức học phí",
                },
            ]
            self.assertEqual(index.build(chunks), 2)
            results = index.search("học bổng", chunks, top_k=1)
            self.assertEqual(results[0][1]["chunk_id"], "scholarship-1")
            self.assertEqual(index.build(chunks, document_id="fees"), 1)
            self.assertEqual(index.search("học phí", chunks, top_k=1)[0][1]["chunk_id"], "fees-1")

    def test_changed_final_text_reembeds_only_its_document(self) -> None:
        class FakeEmbedder:
            def __init__(self):
                self.calls = []

            def embed(self, texts):
                self.calls.append(texts)
                return [[1.0, 0.0] for _ in texts]

        with tempfile.TemporaryDirectory() as temporary:
            embedder = FakeEmbedder()
            index = VectorIndex(Path(temporary), model_name="fake-model", embedder=embedder)
            chunks = [
                {"chunk_id": "c1", "document_id": "d1", "title": "Title", "text": "old text"},
                {"chunk_id": "c2", "document_id": "d2", "title": "Title", "text": "unchanged"},
            ]
            index.build(chunks)
            chunks[0]["text"] = "new text"
            index.ensure_current(chunks)
            self.assertEqual(len(embedder.calls), 2)
            records = index._load()
            self.assertEqual(records["c1"]["content_hash"], index._content_hash(chunks[0]))


class ReviewTests(unittest.TestCase):
    def test_apply_review_changes_only_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "documents.jsonl"
            source.write_text(
                '{"id":"doc-1","title":"Title","content":"old","category":"01_admission"}\n',
                encoding="utf-8",
            )
            review_dir = root / "reviews"
            review_dir.mkdir()
            (review_dir / "doc-1.md").write_text(
                "<!-- document-id: doc-1 -->\n"
                "<!-- source-url: https://example.org -->\n"
                "<!-- content-begin -->\nnew content\n<!-- content-end -->\n",
                encoding="utf-8",
            )
            with patch("scripts.apply_review.set_state"):
                self.assertEqual(apply_reviews(source, review_dir), 1)
            record = json.loads(source.read_text(encoding="utf-8"))
            self.assertEqual(record["content"], "new content")
            self.assertEqual(record["title"], "Title")
            self.assertEqual(record["category"], "01_admission")

    def test_review_without_markers_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.md"
            path.write_text("# not a review", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_review(path)

    def test_chunk_edit_and_approval_are_saved_separately_from_source_document(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal_file = root / "proposals" / "01.jsonl"
            reviewed_file = root / "reviewed" / "01.jsonl"
            final_file = root / "final" / "01.jsonl"
            proposal_file.parent.mkdir()
            proposal_file.write_text(
                json.dumps({
                    "chunk_id": "doc-1__section-001__chunk-001",
                    "document_id": "doc-1",
                    "title": "Title",
                    "text": "proposal text",
                    "source_url": "https://example.org",
                    "status": "proposal",
                }) + "\n",
                encoding="utf-8",
            )
            review_chunks(
                proposal_file,
                reviewed_file,
                action="edit",
                chunk_id="doc-1__section-001__chunk-001",
                text="human-approved text",
            )
            reviewed = json.loads(reviewed_file.read_text(encoding="utf-8"))
            self.assertEqual(reviewed["text"], "proposal text")
            self.assertEqual(reviewed["text_override"], "human-approved text")
            with patch("rag.chunk_review.get_state", return_value="chunk_review"), patch(
                "rag.chunk_review.set_state"
            ):
                count = finalize_chunks(
                    proposal_file,
                    reviewed_file,
                    final_file,
                    document_id="doc-1",
                )
            self.assertEqual(count, 1)
            final = json.loads(final_file.read_text(encoding="utf-8"))
            self.assertEqual(final["text"], "human-approved text")
            self.assertEqual(final["status"], "final")

    def test_split_requires_two_nonempty_parts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal = root / "proposals.jsonl"
            reviewed = root / "reviewed.jsonl"
            proposal.write_text(
                json.dumps({"chunk_id": "c1", "document_id": "d1", "text": "body"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                review_chunks(
                    proposal,
                    reviewed,
                    action="split",
                    chunk_id="c1",
                    text="only one side",
                )

    def test_split_creates_reviewable_children_and_finalizes_only_approved_children(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal = root / "proposals.jsonl"
            reviewed = root / "reviewed.jsonl"
            final = root / "final.jsonl"
            proposal.write_text(
                json.dumps({"chunk_id": "c1", "document_id": "d1", "text": "combined"}) + "\n",
                encoding="utf-8",
            )
            review_chunks(
                proposal,
                reviewed,
                action="split",
                chunk_id="c1",
                text="first part\n---CHUNK-SPLIT---\nsecond part",
            )
            review_chunks(proposal, reviewed, action="approve", chunk_id="c1__a")
            review_chunks(proposal, reviewed, action="approve", chunk_id="c1__b")
            with patch("rag.chunk_review.get_state", return_value="chunk_review"), patch(
                "rag.chunk_review.set_state"
            ):
                self.assertEqual(finalize_chunks(proposal, reviewed, final, document_id="d1"), 2)
            final_chunks = [
                json.loads(line)
                for line in final.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([chunk["text"] for chunk in final_chunks], ["first part", "second part"])

    def test_merge_preserves_both_chunk_provenances(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal = root / "proposals.jsonl"
            reviewed = root / "reviewed.jsonl"
            proposal.write_text(
                "\n".join(json.dumps(chunk) for chunk in [
                    {
                        "chunk_id": "c1", "document_id": "d1", "text": "first",
                        "heading_path": ["Section 1"], "source_position": {"section": 1},
                    },
                    {
                        "chunk_id": "c2", "document_id": "d1", "text": "second",
                        "heading_path": ["Section 2"], "source_position": {"section": 2},
                    },
                ]) + "\n",
                encoding="utf-8",
            )
            review_chunks(
                proposal,
                reviewed,
                action="merge",
                chunk_id="c1",
                other_chunk_id="c2",
            )
            merged_id = "c1__merged__c2"
            review_chunks(proposal, reviewed, action="approve", chunk_id=merged_id)
            records = [
                json.loads(line)
                for line in reviewed.read_text(encoding="utf-8").splitlines()
            ]
            merged = next(record for record in records if record["chunk_id"] == merged_id)
            self.assertEqual(merged["text"], "first\n\nsecond")
            self.assertEqual(merged["heading_path"], ["Section 1", "Section 2"])
            self.assertEqual(merged["source_positions"], [{"section": 1}, {"section": 2}])

    def test_review_file_can_progress_one_document_without_losing_another(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal = root / "proposals.jsonl"
            reviewed = root / "reviewed.jsonl"
            chunks = [
                {"chunk_id": "a1", "document_id": "a", "text": "a", "proposal_revision": "a-v1"},
                {"chunk_id": "b1", "document_id": "b", "text": "b", "proposal_revision": "b-v1"},
            ]
            proposal.write_text(
                "\n".join(json.dumps(chunk) for chunk in chunks) + "\n",
                encoding="utf-8",
            )
            review_chunks(proposal, reviewed, action="approve", chunk_id="a1")
            review_chunks(proposal, reviewed, action="edit", chunk_id="b1", text="human edit b")
            records = [
                json.loads(line)
                for line in reviewed.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual({record["chunk_id"] for record in records}, {"a1", "b1"})
            self.assertEqual(next(record for record in records if record["chunk_id"] == "a1")["status"], "approved")
            self.assertEqual(next(record for record in records if record["chunk_id"] == "b1")["text_override"], "human edit b")

    def test_stale_chunk_reviews_are_not_silently_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal = root / "proposals.jsonl"
            reviewed = root / "reviewed.jsonl"
            record = {"chunk_id": "c1", "document_id": "d1", "text": "new text", "proposal_revision": "v2"}
            proposal.write_text(json.dumps(record) + "\n", encoding="utf-8")
            reviewed.write_text(
                json.dumps({**record, "text": "old text", "proposal_revision": "v1", "status": "approved"})
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "stale"):
                review_chunks(proposal, reviewed, action="approve", chunk_id="c1")


class InteractiveReviewTests(unittest.TestCase):
    def _paths(self, root: Path) -> ReviewPaths:
        return ReviewPaths(
            documents_dir=root / "documents",
            review_dir=root / "review",
            proposals_dir=root / "proposals",
            reviewed_dir=root / "reviewed",
            final_dir=root / "final",
            state_file=root / "workflow" / "documents.json",
            session_file=root / "workflow" / "review-session.json",
        )

    def test_document_transitions_pending_selection_and_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._paths(root)
            paths.documents_dir.mkdir()
            records = [
                {"id": "doc-1", "category": "01_admission", "content": "One"},
                {"id": "doc-2", "category": "01_admission", "content": "Two"},
            ]
            (paths.documents_dir / "category.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(len(pending_documents(paths)), 2)
            self.assertEqual(review_document("doc-1", "approve", state_file=paths.state_file), "approved")
            self.assertEqual(review_document("doc-2", "reject", state_file=paths.state_file), "rejected")
            self.assertEqual(pending_documents(paths), [])
            self.assertEqual(get_state("doc-1", paths.state_file), "approved")
            self.assertEqual(progress_label(2, 5), "2 / 5")
            with self.assertRaises(ValueError):
                progress_label(6, 5)

    def test_picker_paginates_beyond_ten_and_category_selector_shows_all_categories(self) -> None:
        from rag.review import _select_category, _select_item

        categories = {f"{index:02d}_category": 0 for index in range(1, 15)}
        category_output: list[str] = []
        self.assertEqual(
            _select_category(
                categories,
                kind="document",
                input_fn=lambda _: "14",
                output_fn=category_output.append,
            ),
            "14_category",
        )
        self.assertEqual(sum("pending)" in line for line in category_output), 14)

        items = [
            (Path("proposals.jsonl"), {"chunk_id": f"chunk-{index}", "text": "content"})
            for index in range(1, 26)
        ]
        answers = iter([">", "11"])
        selected = _select_item(
            items,
            kind="chunk",
            input_fn=lambda _: next(answers),
            output_fn=lambda _: None,
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected[1]["chunk_id"], "chunk-11")
        year_match = _select_item(
            [(Path("proposals.jsonl"), {"chunk_id": "chunk-2026", "year": 2026})],
            kind="chunk",
            input_fn=lambda _: "2026",
            output_fn=lambda _: None,
        )
        self.assertEqual(year_match[1]["chunk_id"], "chunk-2026")

    def test_approved_final_chunk_can_be_edited_and_refinalized(self) -> None:
        from rag.review import review_chunk_queue

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._paths(root)
            paths.proposals_dir.mkdir()
            proposal_file = paths.proposals_dir / "01_admission.jsonl"
            proposal_file.write_text(
                json.dumps({
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "category": "01_admission",
                    "title": "Admission",
                    "source_url": "https://example.org",
                    "text": "Original approved text",
                    "status": "proposal",
                }) + "\n",
                encoding="utf-8",
            )
            reviewed_file = paths.reviewed_dir / proposal_file.name
            review_chunks(proposal_file, reviewed_file, action="approve", chunk_id="chunk-1")
            set_state("doc-1", "approved", state_file=paths.state_file)
            finalize_chunks(
                proposal_file,
                reviewed_file,
                paths.final_dir / proposal_file.name,
                document_id="doc-1",
                state_file=paths.state_file,
            )
            with proposal_file.open("a", encoding="utf-8") as proposals:
                proposals.write(
                    json.dumps({
                        "chunk_id": "chunk-2",
                        "document_id": "doc-1",
                        "category": "01_admission",
                        "title": "Admission",
                        "source_url": "https://example.org",
                        "text": "Still pending proposal",
                        "status": "proposal",
                    }) + "\n"
                )

            answers = iter([
                "01_admission",
                "chunk-1",
                "e",
                "Corrected final text",
                ":save",
                "q",
                "q",
            ])
            output: list[str] = []
            review_chunk_queue(
                paths,
                amend=True,
                input_fn=lambda _: next(answers),
                output_fn=output.append,
            )

            final_record = json.loads(
                (paths.final_dir / proposal_file.name).read_text(encoding="utf-8")
            )
            self.assertEqual(final_record["text"], "Corrected final text")
            self.assertEqual(final_record["status"], "final")
            self.assertEqual(get_state("doc-1", paths.state_file), "final")
            self.assertEqual(
                [chunk["chunk_id"] for _, chunk in pending_chunks(paths)],
                ["chunk-2"],
            )
            self.assertTrue(any("Updated final chunk chunk-1" in line for line in output))

    def test_approving_last_pending_chunk_auto_finalizes_document(self) -> None:
        from rag.review import review_chunk_queue

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._paths(root)
            paths.proposals_dir.mkdir()
            proposal = paths.proposals_dir / "category.jsonl"
            proposal.write_text(
                "\n".join(
                    json.dumps({
                        "chunk_id": chunk_id,
                        "document_id": "doc-1",
                        "category": "01_admission",
                        "title": "Title",
                        "source_url": "https://example.org",
                        "text": chunk_id,
                        "status": "proposal",
                    })
                    for chunk_id in ("c1", "c2")
                ) + "\n",
                encoding="utf-8",
            )
            set_state("doc-1", "chunk_proposed", state_file=paths.state_file)
            output: list[str] = []
            answers = iter([
                "01_admission",
                "c1",
                "a",
                "c2",
                "a",
                "q",
                "q",
            ])
            review_chunk_queue(
                paths,
                input_fn=lambda _: next(answers),
                output_fn=output.append,
            )

            final_file = paths.final_dir / proposal.name
            final = [json.loads(line) for line in final_file.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({chunk["chunk_id"] for chunk in final}, {"c1", "c2"})
            self.assertEqual(get_state("doc-1", paths.state_file), "final")
            self.assertTrue(any("Finalized 2 chunks" in line for line in output))
            proposal_records = [json.loads(line) for line in proposal.read_text(encoding="utf-8").splitlines()]
            reviewed_records = [
                json.loads(line)
                for line in (paths.reviewed_dir / proposal.name).read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(all(chunk["status"] == "proposal" for chunk in proposal_records))
            self.assertTrue(all(chunk["status"] == "approved" for chunk in reviewed_records))

    def test_finalize_reviewed_menu_completes_previously_approved_chunks(self) -> None:
        from rag.review import finalize_ready_documents

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._paths(root)
            paths.proposals_dir.mkdir()
            proposal = paths.proposals_dir / "category.jsonl"
            proposal.write_text(
                json.dumps({
                    "chunk_id": "c1",
                    "document_id": "doc-1",
                    "category": "01_admission",
                    "title": "Title",
                    "source_url": "https://example.org",
                    "text": "Approved content",
                    "status": "proposal",
                }) + "\n",
                encoding="utf-8",
            )
            review_chunks(proposal, paths.reviewed_dir / proposal.name, action="approve", chunk_id="c1")
            set_state("doc-1", "chunk_review", state_file=paths.state_file)

            output: list[str] = []
            answers = iter(["01_admission", "doc-1"])
            finalize_ready_documents(
                paths,
                input_fn=lambda _: next(answers),
                output_fn=output.append,
            )

            final = json.loads((paths.final_dir / proposal.name).read_text(encoding="utf-8"))
            self.assertEqual(final["chunk_id"], "c1")
            self.assertEqual(final["text"], "Approved content")
            self.assertEqual(get_state("doc-1", paths.state_file), "final")
            self.assertTrue(any("Finalized 1 chunks" in line for line in output))

    def test_full_document_to_final_chunk_review_and_later_edit_example(self) -> None:
        from rag.review import review_chunk_queue, review_documents
        from scripts.chunk import write_proposals

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._paths(root)
            paths.documents_dir.mkdir()
            source_file = paths.documents_dir / "01_admission.jsonl"
            document = {
                "id": "01_admission_001",
                "category": "01_admission",
                "title": "Admission example",
                "content": "Original content that will be reviewed.",
                "source_url": "https://example.org/admission",
            }
            source_file.write_text(json.dumps(document) + "\n", encoding="utf-8")

            def revise_document(markdown: Path, _editor: str | None = None) -> None:
                markdown.write_text(
                    "<!-- document-id: 01_admission_001 -->\n"
                    "<!-- source-url: https://example.org/admission -->\n"
                    "<!-- content-begin -->\n"
                    "Reviewed source content for chunking.\n"
                    "<!-- content-end -->\n",
                    encoding="utf-8",
                )

            document_answers = iter(["01_admission", "01_admission_001", "a"])
            with patch("rag.review._launch_editor", side_effect=revise_document):
                review_documents(
                    paths,
                    input_fn=lambda _: next(document_answers),
                    output_fn=lambda _: None,
                )
            reviewed_document = json.loads(source_file.read_text(encoding="utf-8"))
            self.assertEqual(
                reviewed_document["content"],
                "Reviewed source content for chunking.",
            )
            self.assertEqual(get_state("01_admission_001", paths.state_file), "approved")

            proposal_file = paths.proposals_dir / source_file.name
            write_proposals([reviewed_document], proposal_file)
            set_state("01_admission_001", "chunk_proposed", state_file=paths.state_file)
            original_chunk = json.loads(proposal_file.read_text(encoding="utf-8"))
            child_a = f"{original_chunk['chunk_id']}__a"
            child_b = f"{original_chunk['chunk_id']}__b"
            chunk_answers = iter([
                "01_admission",
                original_chunk["chunk_id"],
                "s",
                "Reviewed source",
                "---CHUNK-SPLIT---",
                "content for chunking.",
                ":save",
                child_a,
                "a",
                child_b,
                "a",
            ])
            review_chunk_queue(
                paths,
                input_fn=lambda _: next(chunk_answers),
                output_fn=lambda _: None,
            )

            reviewed_file = paths.reviewed_dir / proposal_file.name
            final_file = paths.final_dir / proposal_file.name
            final_before_amend = [
                json.loads(line)
                for line in final_file.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(final_before_amend), 2)
            self.assertEqual(get_state("01_admission_001", paths.state_file), "final")
            amend_answers = iter([
                "01_admission",
                child_a,
                "e",
                "Revised after final approval.",
                ":save",
                "q",
                "q",
            ])
            review_chunk_queue(
                paths,
                amend=True,
                input_fn=lambda _: next(amend_answers),
                output_fn=lambda _: None,
            )
            final_chunks = [
                json.loads(line)
                for line in final_file.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                next(chunk for chunk in final_chunks if chunk["chunk_id"] == child_a)["text"],
                "Revised after final approval.",
            )
            self.assertEqual(len(final_chunks), 2)

    def test_document_edit_uses_markdown_and_preserves_other_source_fields(self) -> None:
        from rag.review import review_documents

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._paths(root)
            paths.documents_dir.mkdir()
            source_file = paths.documents_dir / "category.jsonl"
            original = {
                "id": "doc-1",
                "category": "01_admission",
                "title": "Keep title",
                "content": "Original",
                "source_url": "https://example.org",
            }
            source_file.write_text(json.dumps(original) + "\n", encoding="utf-8")

            def editor(path: Path, _editor: str | None = None) -> None:
                path.write_text(
                    "<!-- document-id: doc-1 -->\n"
                    "<!-- source-url: https://example.org -->\n"
                    "<!-- content-begin -->\nReviewed content\n<!-- content-end -->\n",
                    encoding="utf-8",
                )

            answers = iter(["01_admission", "doc-1", "a"])
            with patch("rag.review._launch_editor", side_effect=editor):
                review_documents(
                    paths,
                    input_fn=lambda _: next(answers),
                    output_fn=lambda _: None,
                )
            saved = json.loads(source_file.read_text(encoding="utf-8"))
            self.assertEqual(saved["content"], "Reviewed content")
            self.assertEqual(saved["title"], "Keep title")
            self.assertEqual(get_state("doc-1", paths.state_file), "approved")
            self.assertTrue((paths.review_dir / "01_admission" / "doc-1.md").exists())

    def test_document_picker_searches_title_without_printing_long_content(self) -> None:
        from rag.review import review_documents

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._paths(root)
            paths.documents_dir.mkdir()
            source_file = paths.documents_dir / "category.jsonl"
            records = [
                {
                    "id": "doc-1",
                    "category": "01_admission",
                    "title": "Ordinary document",
                    "content": "Long source content must stay out of the terminal.",
                    "source_url": "https://example.org/1",
                },
                {
                    "id": "doc-2",
                    "category": "01_admission",
                    "title": "Unique scholarship guide",
                    "content": "Another long source content.",
                    "source_url": "https://example.org/2",
                },
            ]
            source_file.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            output: list[str] = []
            answers = iter(["01_admission", "Unique scholarship guide", "a", "q", "q"])
            with patch("rag.review._launch_editor"):
                review_documents(
                    paths,
                    input_fn=lambda _: next(answers),
                    output_fn=output.append,
                )
            self.assertEqual(get_state("doc-2", paths.state_file), "approved")
            self.assertEqual(get_state("doc-1", paths.state_file), "cleaned")
            self.assertNotIn("Long source content must stay out of the terminal.", "\n".join(output))

    def test_editor_opens_vscode_without_blocking_the_terminal(self) -> None:
        from rag.review import _launch_editor

        with tempfile.TemporaryDirectory() as temporary:
            markdown = Path(temporary) / "review.md"
            with (
                patch("rag.review.shutil.which", return_value="/usr/bin/code"),
                patch("rag.review.subprocess.Popen") as popen,
                patch.dict("os.environ", {"VISUAL": "", "EDITOR": ""}),
            ):
                opened_in_background = _launch_editor(markdown)
            self.assertEqual(
                popen.call_args.args[0],
                ["code", "--reuse-window", str(markdown)],
            )
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertTrue(opened_in_background)

    def test_editor_does_not_block_when_editor_env_uses_code_wait(self) -> None:
        from rag.review import _launch_editor

        with (
            patch("rag.review.subprocess.Popen") as popen,
            patch("rag.review.subprocess.run") as run,
            patch.dict("os.environ", {"VISUAL": "", "EDITOR": "code --wait"}),
        ):
            _launch_editor(Path("/tmp/review.md"))
        self.assertEqual(popen.call_args.args[0], ["code", str(Path("/tmp/review.md"))])
        run.assert_not_called()

    def test_rejecting_document_does_not_apply_unapproved_markdown_edits(self) -> None:
        from rag.review import review_documents

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._paths(root)
            paths.documents_dir.mkdir()
            source_file = paths.documents_dir / "category.jsonl"
            source_file.write_text(
                json.dumps({
                    "id": "doc-1",
                    "category": "01_admission",
                    "title": "Original title",
                    "content": "Original source content",
                    "source_url": "https://example.org",
                }) + "\n",
                encoding="utf-8",
            )

            def edit_markdown(path: Path, _editor: str | None = None) -> None:
                path.write_text(
                    "<!-- document-id: doc-1 -->\n"
                    "<!-- source-url: https://example.org -->\n"
                    "<!-- content-begin -->\nUnapproved edit\n<!-- content-end -->\n",
                    encoding="utf-8",
                )

            answers = iter([
                "01_admission",
                "doc-1",
                "r",
                "y",
                "not an authoritative source",
            ])
            with patch("rag.review._launch_editor", side_effect=edit_markdown):
                review_documents(
                    paths,
                    input_fn=lambda _: next(answers),
                    output_fn=lambda _: None,
                )
            saved = json.loads(source_file.read_text(encoding="utf-8"))
            state = json.loads(paths.state_file.read_text(encoding="utf-8"))["doc-1"]
            self.assertEqual(saved["content"], "Original source content")
            self.assertEqual(state["status"], "rejected")
            self.assertEqual(state["note"], "not an authoritative source")

    def test_document_review_resumes_after_quit_without_losing_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._paths(root)
            paths.documents_dir.mkdir()
            (paths.documents_dir / "category.jsonl").write_text(
                "\n".join(
                    json.dumps({
                        "id": doc_id,
                        "category": "01_admission",
                        "title": doc_id,
                        "content": "content",
                        "source_url": "https://example.org",
                    })
                    for doc_id in ("doc-1", "doc-2")
                ) + "\n",
                encoding="utf-8",
            )
            answers = iter(["1", "01_admission", "doc-1", "a", "q", "q"])
            with patch("rag.review._launch_editor"):
                self.assertEqual(
                    run_review(paths, input_fn=lambda _: next(answers), output_fn=lambda _: None),
                    0,
                )
            self.assertEqual(get_state("doc-1", paths.state_file), "approved")
            self.assertEqual([record["id"] for _, record in pending_documents(paths)], ["doc-2"])
            answers = iter(["5", "01_admission", "doc-2", "a"])
            with patch("rag.review._launch_editor"):
                self.assertEqual(
                    run_review(paths, input_fn=lambda _: next(answers), output_fn=lambda _: None),
                    0,
                )
            self.assertEqual(pending_documents(paths), [])

    def test_chunk_ui_reuses_review_service_for_edit_split_merge_reject(self) -> None:
        from rag.review import review_chunk_queue

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._paths(root)
            paths.proposals_dir.mkdir()
            proposal_file = paths.proposals_dir / "category.jsonl"
            records = [
                {
                    "chunk_id": f"c{index}",
                    "document_id": "doc-1",
                    "title": "Title",
                    "text": f"Text {index}",
                    "category": "01_admission",
                    "year": 2026,
                    "source_url": "https://example.org",
                    "status": "proposal",
                }
                for index in range(1, 6)
            ]
            proposal_file.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            answers = iter([
                "01_admission",
                "c1",
                "e", "Edited text", ":save",
                "c2",
                "s", "First part", "---CHUNK-SPLIT---", "Second part", ":save",
                "c2__a",
                "a",
                "c2__b",
                "a",
                "c3",
                "m", "1", "y",
                "c3__merged__c4",
                "a",
                "c5",
                "r", "y", "duplicate proposal",
            ])
            output: list[str] = []
            with patch("rag.review.review_chunks", wraps=review_chunks) as service:
                review_chunk_queue(paths, input_fn=lambda _: next(answers), output_fn=output.append)

            actions = [call.kwargs["action"] for call in service.call_args_list]
            self.assertEqual(
                actions,
                ["edit", "split", "approve", "approve", "merge", "approve", "reject"],
            )
            saved = [
                json.loads(line)
                for line in (paths.reviewed_dir / proposal_file.name)
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                next(chunk for chunk in saved if chunk["chunk_id"] == "c1")["text_override"],
                "Edited text",
            )
            self.assertTrue(any(chunk["chunk_id"] == "c3__merged__c4" for chunk in saved))
            self.assertEqual(
                next(chunk for chunk in saved if chunk["chunk_id"] == "c5")["status"],
                "rejected",
            )
            self.assertEqual(
                next(chunk for chunk in saved if chunk["chunk_id"] == "c5")["review_note"],
                "duplicate proposal",
            )
            self.assertEqual(pending_chunks(paths), [])
            self.assertTrue(any("Progress:" in line for line in output))

    def test_chunk_ui_resume_keeps_saved_decisions_after_quit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._paths(root)
            paths.proposals_dir.mkdir()
            proposal = paths.proposals_dir / "category.jsonl"
            proposal.write_text(
                "\n".join(
                    json.dumps({
                        "chunk_id": chunk_id,
                        "document_id": "doc-1",
                        "category": "01_admission",
                        "text": chunk_id,
                    })
                    for chunk_id in ("c1", "c2")
                ) + "\n",
                encoding="utf-8",
            )
            answers = iter(["2", "01_admission", "c1", "a", "q", "q"])
            self.assertEqual(
                run_review(paths, input_fn=lambda _: next(answers), output_fn=lambda _: None),
                0,
            )
            self.assertEqual([chunk["chunk_id"] for _, chunk in pending_chunks(paths)], ["c2"])
            answers = iter(["5", "01_admission", "c2", "a"])
            self.assertEqual(
                run_review(paths, input_fn=lambda _: next(answers), output_fn=lambda _: None),
                0,
            )
            self.assertEqual(pending_chunks(paths), [])


class VLLMTests(unittest.TestCase):
    def test_loads_local_dotenv_and_uses_served_model_default(self) -> None:
        from rag.pipeline import DEFAULT_VLLM_MODEL

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".env").write_text(
                "VLLM_MODEL=local-dotenv-model\nVLLM_BASE_URL=http://localhost:9000/v1\n",
                encoding="utf-8",
            )
            with (
                patch("rag.pipeline.ROOT_DIR", root),
                patch.dict("os.environ", {}, clear=True),
            ):
                client = VLLMChatClient()
                self.assertEqual(client.model, "local-dotenv-model")
                self.assertEqual(client.base_url, "http://localhost:9000/v1")

        with patch.dict("os.environ", {}, clear=True):
            client = VLLMChatClient()
        self.assertEqual(client.model, DEFAULT_VLLM_MODEL)

    @patch("rag.pipeline.requests.post")
    def test_sends_bearer_token_when_configured(self, post: Mock) -> None:
        post.return_value = Mock(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "ok"}}]},
        )
        client = VLLMChatClient(model="served-model", api_key="test-secret")
        client.answer("question", [])
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"],
            "Bearer test-secret",
        )

    @patch("rag.pipeline.requests.post")
    def test_calls_openai_compatible_chat_completion(self, post: Mock) -> None:
        post.return_value = Mock(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "Trả lời có nguồn"}}]},
        )
        client = VLLMChatClient(base_url="http://localhost:8000/v1", model="served-model")
        answer = client.answer("question", [])
        self.assertEqual(answer, "Trả lời có nguồn")
        self.assertEqual(post.call_args.args[0], "http://localhost:8000/v1/chat/completions")


if __name__ == "__main__":
    unittest.main()
