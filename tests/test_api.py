from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient


class ApiTests(unittest.TestCase):
    def test_chat_rejects_empty_question(self) -> None:
        with patch("app.main.runtime.rag", Mock()):
            from app.main import app

            response = TestClient(app).post("/api/v1/chat", json={"question": "  "})
        self.assertEqual(response.status_code, 422)

    def test_chat_returns_stable_response_and_request_id(self) -> None:
        rag = Mock()
        rag.ask.return_value = Mock(
            answer="Có thông tin trong tài liệu.",
            sources=[
                {
                    "source_id": "doc-1",
                    "chunk_id": "chunk-1",
                    "title": "Thông tin tuyển sinh",
                    "url": "https://tuyensinh.haui.edu.vn/example",
                    "year": 2026,
                    "category": "01_admission",
                    "score": 0.91,
                }
            ],
        )
        with patch("app.main.runtime.rag", rag):
            from app.main import app

            response = TestClient(app).post(
                "/api/v1/chat",
                json={"question": "Phương thức xét tuyển?", "history": []},
            )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer"], "Có thông tin trong tài liệu.")
        self.assertEqual(payload["sources"][0]["source_id"], "doc-1")
        self.assertTrue(payload["request_id"])
        rag.ask.assert_called_once()

    def test_chat_hides_backend_error_details(self) -> None:
        rag = Mock()
        rag.ask.side_effect = RuntimeError("secret model endpoint details")
        with patch("app.main.runtime.rag", rag):
            from app.main import app

            response = TestClient(app).post("/api/v1/chat", json={"question": "Học phí?"})
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("secret", response.text)


if __name__ == "__main__":
    unittest.main()
