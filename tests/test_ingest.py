"""Safe file/URL ingestion with human review, dedup, and update flows."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.document_workflow import get_state
from rag.ingest import (
    IngestError,
    document_path,
    extract_content,
    fetch_url,
    find_duplicate,
    import_source,
    load_documents,
    next_document_id,
    resolve_local_file,
    set_document_decision,
    validate_category,
    validate_url,
)

PUBLIC_IP = "93.184.216.34"


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
        chunk_size: int = 4,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body
        self._chunk_size = chunk_size
        self.closed = False

    def iter_content(self, chunk_size: int = 64 * 1024):
        step = min(chunk_size, self._chunk_size)
        for index in range(0, len(self._body), step):
            yield self._body[index : index + step]

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, *responses: FakeResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, **kwargs) -> FakeResponse:
        self.calls.append(url)
        if not self.responses:
            raise AssertionError("unexpected extra request")
        return self.responses.pop(0)


class CategoryAndPathTests(unittest.TestCase):
    def test_category_cannot_traverse_or_change_file_extension(self) -> None:
        for category in ("../etc/passwd", "01/evil", "01_admission/../../x", "", "01_Admission"):
            with self.subTest(category=category):
                with self.assertRaises(IngestError):
                    validate_category(category, Path("/tmp/documents"))

    def test_document_path_stays_inside_the_documents_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = document_path("01_admission", directory)
        self.assertEqual(path.name, "01_admission.jsonl")
        self.assertEqual(path.parent, directory)

    def test_local_file_rejects_outside_root_and_unsupported_suffixes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside"
            allowed = root / "allowed"
            outside.mkdir()
            allowed.mkdir()
            secret = outside / "secret.md"
            secret.write_text("secret", encoding="utf-8")
            with self.assertRaises(IngestError):
                resolve_local_file(str(secret), allowed_root=allowed)
            executable = allowed / "payload.exe"
            executable.write_text("x", encoding="utf-8")
            with self.assertRaises(IngestError):
                resolve_local_file(str(executable), allowed_root=allowed)
            accepted = allowed / "source.md"
            accepted.write_text("ok", encoding="utf-8")
            self.assertEqual(resolve_local_file(str(accepted), allowed_root=allowed), accepted)

    def test_symlink_escape_is_resolved_before_the_root_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            allowed = root / "allowed"
            allowed.mkdir()
            secret = root / "secret.md"
            secret.write_text("secret", encoding="utf-8")
            link = allowed / "link.md"
            link.symlink_to(secret)
            with self.assertRaises(IngestError):
                resolve_local_file(str(link), allowed_root=allowed)


class UrlSafetyTests(unittest.TestCase):
    def test_blocks_local_private_and_metadata_targets(self) -> None:
        blocked = [
            "file:///etc/passwd",
            "ftp://example.org/file",
            "http://user:pass@example.org/",
            "http://localhost/",
            "http://service.localhost/",
            "http://metadata.google.internal/",
            "http://127.0.0.1/",
            "http://127.1/",
            "http://10.0.0.5/",
            "http://192.168.1.1/",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]/",
            "http://0.0.0.0/",
            "http://2130706433/",
        ]
        for url in blocked:
            with self.subTest(url=url):
                with self.assertRaises(IngestError):
                    validate_url(url)

    def test_accepts_a_public_literal_address(self) -> None:
        self.assertEqual(validate_url(f"http://{PUBLIC_IP}/page"), f"http://{PUBLIC_IP}/page")

    def test_hostname_that_resolves_to_a_private_address_is_rejected(self) -> None:
        with patch(
            "rag.ingest.socket.getaddrinfo",
            return_value=[(2, 1, 6, "", ("10.0.0.5", 80))],
        ):
            with self.assertRaises(IngestError):
                validate_url("http://internal.example.org/")

    def test_fetch_revalidates_redirects_and_never_calls_the_internal_target(self) -> None:
        session = FakeSession(
            FakeResponse(status_code=302, headers={"location": "http://169.254.169.254/latest"})
        )
        with self.assertRaises(IngestError):
            fetch_url(f"http://{PUBLIC_IP}/start", session=session)
        self.assertEqual(session.calls, [f"http://{PUBLIC_IP}/start"])

    def test_fetch_follows_a_public_redirect_and_caps_size(self) -> None:
        session = FakeSession(
            FakeResponse(status_code=302, headers={"location": f"http://{PUBLIC_IP}/final"}),
            FakeResponse(
                body=b"hello world",
                headers={"content-type": "text/plain; charset=utf-8"},
            ),
        )
        fetched = fetch_url(f"http://{PUBLIC_IP}/start", session=session)
        self.assertEqual(fetched.data, b"hello world")
        self.assertEqual(fetched.final_url, f"http://{PUBLIC_IP}/final")
        self.assertEqual(fetched.content_type, "text/plain")

        too_large = FakeSession(
            FakeResponse(
                body=b"x" * 11,
                headers={"content-type": "text/plain", "content-length": "11"},
            )
        )
        with self.assertRaises(IngestError):
            fetch_url(f"http://{PUBLIC_IP}/large", session=too_large, max_bytes=10)

    def test_fetch_rejects_html_when_it_is_not_an_accepted_type(self) -> None:
        session = FakeSession(
            FakeResponse(body=b"x", headers={"content-type": "application/octet-stream"})
        )
        with self.assertRaises(IngestError):
            fetch_url(f"http://{PUBLIC_IP}/file", session=session)

    def test_too_many_redirects_is_an_error(self) -> None:
        session = FakeSession(
            *[
                FakeResponse(status_code=302, headers={"location": f"http://{PUBLIC_IP}/{i}"})
                for i in range(5)
            ]
        )
        with self.assertRaises(IngestError):
            fetch_url(f"http://{PUBLIC_IP}/loop", session=session)


class ExtractionTests(unittest.TestCase):
    def test_html_extracts_title_and_text_without_scripts_or_navigation(self) -> None:
        html = b"""
        <html><head><title>Thong tin tuyen sinh</title>
        <style>.x{display:none}</style></head>
        <body><nav>Menu</nav><main><h1>Tuyen sinh 2026</h1>
        <script>alert(1)</script><p>Hoc phi la 1 trieu.</p></main></body></html>
        """
        extracted = extract_content(html, content_type="text/html")
        self.assertEqual(extracted.title_hint, "Thong tin tuyen sinh")
        self.assertIn("Tuyen sinh 2026", extracted.content)
        self.assertIn("Hoc phi la 1 trieu.", extracted.content)
        self.assertNotIn("alert", extracted.content)
        self.assertNotIn("Menu", extracted.content)

    def test_json_accepts_content_or_a_string_list(self) -> None:
        one = extract_content(
            json.dumps({"title": "T", "content": "Noi dung"}).encode(),
            content_type="application/json",
        )
        self.assertEqual((one.title_hint, one.content), ("T", "Noi dung"))
        two = extract_content(
            json.dumps(["Dong mot", "Dong hai"]).encode(),
            content_type="application/json",
        )
        self.assertEqual(two.content, "Dong mot\n\nDong hai")
        with self.assertRaises(IngestError):
            extract_content(b'{"other": 1}', content_type="application/json")

    def test_plain_text_normalizes_line_endings_and_blank_runs(self) -> None:
        extracted = extract_content(b"a\r\n\r\n\r\n\r\nb  \r\n", content_type="text/plain")
        self.assertEqual(extracted.content, "a\n\nb")


class ImportFlowTests(unittest.TestCase):
    def _source(self, root: Path, text: str = "Noi dung tuyen sinh 2026.") -> Path:
        path = root / "source.md"
        path.write_text(text, encoding="utf-8")
        return path

    def _import(self, root: Path, source: Path, **overrides):
        options = {
            "category": "09_training_regulations",
            "documents_dir": root / "documents",
            "state_file": root / "workflow" / "documents.json",
            "metadata": {"title": "Quy che dao tao", "year": 2026},
        }
        options.update(overrides)
        return import_source(str(source), **options)

    def test_import_writes_a_review_pending_record_and_never_deletes_the_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            result = self._import(root, source)
            document_file = root / "documents" / "09_training_regulations.jsonl"

            self.assertTrue(result.wrote)
            self.assertTrue(result.created)
            self.assertEqual(result.document["id"], "09_training_regulations_001")
            self.assertEqual(result.document["category"], "09_training_regulations")
            self.assertEqual(result.document["year"], 2026)
            self.assertEqual(result.document["content"], "Noi dung tuyen sinh 2026.")
            self.assertTrue(result.document["source_url"].startswith("file://"))
            self.assertEqual(result.document["source_type"], "file_import")
            self.assertEqual(
                get_state(result.document["id"], root / "workflow" / "documents.json"),
                "review_pending",
            )
            records = [json.loads(line) for line in document_file.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 1)
            self.assertTrue(source.exists())

    def test_duplicate_is_refused_by_name_and_can_be_updated_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            first = self._import(root, source)

            with self.assertRaisesRegex(IngestError, first.document["id"]):
                self._import(root, source)

            source.write_text("Noi dung da cap nhat.", encoding="utf-8")
            updated = self._import(
                root,
                source,
                duplicate="update",
                metadata={"title": "Quy che da cap nhat", "year": 2026},
            )
            self.assertFalse(updated.created)
            self.assertEqual(updated.document["id"], first.document["id"])
            self.assertEqual(updated.document["content"], "Noi dung da cap nhat.")
            self.assertEqual(updated.document["title"], "Quy che da cap nhat")
            self.assertEqual(updated.duplicate_of, first.document["id"])
            self.assertEqual(
                get_state(first.document["id"], root / "workflow" / "documents.json"),
                "review_pending",
            )
            records = load_documents(root / "documents")
            self.assertEqual(len(records), 1)

    def test_explicit_update_id_replaces_content_and_resets_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            first = self._import(root, source, review="approve")
            self.assertEqual(
                get_state(first.document["id"], root / "workflow" / "documents.json"),
                "approved",
            )
            source.write_text("Ban sua doi.", encoding="utf-8")
            updated = self._import(
                root,
                source,
                update_id=first.document["id"],
                metadata={"title": "Ban sua doi", "year": 2026},
            )
            self.assertEqual(updated.document["id"], first.document["id"])
            self.assertEqual(updated.document["content"], "Ban sua doi.")
            self.assertEqual(
                get_state(first.document["id"], root / "workflow" / "documents.json"),
                "review_pending",
            )

    def test_dry_run_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            result = self._import(root, source, dry_run=True)
            self.assertFalse(result.wrote)
            self.assertFalse((root / "documents").exists())
            self.assertFalse((root / "workflow" / "documents.json").exists())

    def test_reject_and_approve_are_recorded_without_deleting_the_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            result = self._import(root, source, review="reject")
            state_file = root / "workflow" / "documents.json"
            self.assertEqual(get_state(result.document["id"], state_file), "rejected")
            self.assertTrue(result.path.exists())
            self.assertEqual(set_document_decision(result.document["id"], "approve", state_file=state_file), "approved")
            with self.assertRaises(IngestError):
                set_document_decision(result.document["id"], "delete", state_file=state_file)

    def test_metadata_cannot_override_server_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            for reserved in ("id", "content", "source_url", "category"):
                with self.subTest(reserved=reserved):
                    with self.assertRaises(IngestError):
                        self._import(root, source, metadata={reserved: "hijack"})

    def test_update_cannot_move_a_document_to_another_category(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            first = self._import(root, source)
            with self.assertRaises(IngestError):
                self._import(root, source, update_id=first.document["id"], category="01_admission")

    def test_next_document_id_skips_existing_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "documents"
            directory.mkdir()
            (directory / "01_admission.jsonl").write_text(
                json.dumps({"id": "01_admission_007", "category": "01_admission"}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(next_document_id("01_admission", directory), "01_admission_008")

    def test_dedup_detects_identical_content_under_a_different_url(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "documents"
            directory.mkdir()
            (directory / "01_admission.jsonl").write_text(
                json.dumps(
                    {
                        "id": "01_admission_001",
                        "category": "01_admission",
                        "source_url": "https://example.org/one",
                        "content": "Noi dung trung nhau.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            match = find_duplicate(
                "  noi   dung trung nhau. ",
                "https://example.org/two",
                documents_dir=directory,
            )
        self.assertIsNotNone(match)
        self.assertEqual(match[1]["id"], "01_admission_001")


if __name__ == "__main__":
    unittest.main()
