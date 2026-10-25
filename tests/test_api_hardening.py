"""Additive API contract: health/ready fields, limits, codes, and secret hygiene."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.main as main
from app.core.middleware import BodySizeLimitMiddleware, RateLimitMiddleware
from rag.llm.errors import ProviderTimeout


def _rag(**overrides) -> Mock:
    rag = Mock()
    rag.chunk_count = 1224
    rag.provider_name = "ollama"
    rag.model_name = "qwen3.5:4b"
    rag.probe.return_value = True
    for key, value in overrides.items():
        setattr(rag, key, value)
    return rag


class HealthTests(unittest.TestCase):
    def test_health_keeps_original_fields_and_adds_diagnostics(self) -> None:
        with (
            patch("app.main.runtime.rag", _rag()),
            patch("app.main.runtime.model_configured", True),
            patch("app.main.index_ready", return_value=True),
        ):
            response = TestClient(main.app).get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ready")
        self.assertTrue(payload["rag_loaded"])
        self.assertTrue(payload["model_configured"])
        self.assertTrue(payload["index_ready"])
        self.assertEqual(payload["chunk_count"], 1224)
        self.assertEqual(payload["provider"], "ollama")
        self.assertEqual(payload["model"], "qwen3.5:4b")
        self.assertTrue(payload["app_version"])

    def test_health_is_degraded_when_the_index_is_missing_but_still_200(self) -> None:
        with (
            patch("app.main.runtime.rag", _rag()),
            patch("app.main.runtime.model_configured", True),
            patch("app.main.index_ready", return_value=False),
        ):
            response = TestClient(main.app).get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "degraded")
        self.assertFalse(response.json()["index_ready"])

    def test_health_and_ready_do_not_leak_paths_urls_or_credentials(self) -> None:
        settings = main.settings
        rag = _rag()
        with (
            patch("app.main.runtime.rag", rag),
            patch("app.main.runtime.model_configured", True),
            patch("app.main.index_ready", return_value=True),
        ):
            client = TestClient(main.app)
            bodies = [
                client.get("/api/v1/health").text,
                client.get("/api/v1/ready").text,
            ]
        for body in bodies:
            self.assertNotIn(settings.index_dir, body)
            self.assertNotIn(settings.chunks_dir, body)
            self.assertNotIn("http://", body)
            self.assertNotIn("api_key", body.lower())
            self.assertNotIn("Traceback", body)


class ReadinessTests(unittest.TestCase):
    def test_ready_returns_200_only_when_all_checks_pass(self) -> None:
        with (
            patch("app.main.runtime.rag", _rag()),
            patch("app.main.index_ready", return_value=True),
        ):
            response = TestClient(main.app).get("/api/v1/ready")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(
            payload["checks"],
            {
                "chunks_loaded": True,
                "index_ready": True,
                "provider_reachable": True,
            },
        )
        self.assertTrue(payload["request_id"])

    def test_ready_returns_503_when_provider_is_unreachable(self) -> None:
        with (
            patch("app.main.runtime.rag", _rag(probe=Mock(return_value=False))),
            patch("app.main.index_ready", return_value=True),
        ):
            response = TestClient(main.app).get("/api/v1/ready")
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json()["checks"]["provider_reachable"])

    def test_ready_returns_503_when_index_is_missing(self) -> None:
        with (
            patch("app.main.runtime.rag", _rag()),
            patch("app.main.index_ready", return_value=False),
        ):
            response = TestClient(main.app).get("/api/v1/ready")
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json()["checks"]["index_ready"])


class ChatContractTests(unittest.TestCase):
    def test_success_response_includes_request_id_in_body_and_header(self) -> None:
        rag = _rag()
        rag.ask.return_value = Mock(
            answer="Trả lời",
            sources=[],
        )
        with patch("app.main.runtime.rag", rag):
            response = TestClient(main.app).post(
                "/api/v1/chat", json={"question": "Học phí?"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["x-request-id"], response.json()["request_id"]
        )

    def test_validation_error_uses_stable_code(self) -> None:
        with patch("app.main.runtime.rag", _rag()):
            response = TestClient(main.app).post("/api/v1/chat", json={"question": "  "})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["code"], "invalid_request")
        self.assertTrue(response.json()["request_id"])

    def test_configured_question_limit_has_its_own_code(self) -> None:
        with patch("app.main.runtime.rag", _rag()):
            response = TestClient(main.app).post(
                "/api/v1/chat", json={"question": "x" * 2001}
            )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["code"], "question_too_long")

    def test_configured_history_limit_has_its_own_code(self) -> None:
        history = [{"role": "user", "content": "x"}] * 9
        with patch("app.main.runtime.rag", _rag()):
            response = TestClient(main.app).post(
                "/api/v1/chat",
                json={"question": "Học phí?", "history": history},
            )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["code"], "history_too_long")

    def test_provider_failure_returns_typed_code_and_keeps_error_type(self) -> None:
        rag = _rag()
        rag.ask.side_effect = ProviderTimeout("slow", provider="vllm", model="m")
        with patch("app.main.runtime.rag", rag):
            response = TestClient(main.app).post(
                "/api/v1/chat", json={"question": "Học phí?"}
            )
        self.assertEqual(response.status_code, 502)
        payload = response.json()
        self.assertEqual(payload["code"], "provider_timeout")
        self.assertEqual(payload["error_type"], "ProviderTimeout")
        self.assertNotIn("slow", payload["detail"])

    def test_oversized_body_is_rejected_before_parsing(self) -> None:
        with patch("app.main.runtime.rag", _rag()):
            response = TestClient(main.app).post(
                "/api/v1/chat",
                json={"question": "x" * 70_000},
            )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["code"], "payload_too_large")

    def test_api_key_gate_is_opt_in(self) -> None:
        guarded = replace(main.settings, api_key="s3cret")
        rag = _rag()
        rag.ask.return_value = Mock(answer="ok", sources=[])
        with (
            patch("app.main.runtime.rag", rag),
            patch("app.main.settings", guarded),
        ):
            client = TestClient(main.app)
            missing = client.post("/api/v1/chat", json={"question": "Học phí?"})
            wrong = client.post(
                "/api/v1/chat",
                json={"question": "Học phí?"},
                headers={"X-API-Key": "wrong"},
            )
            accepted = client.post(
                "/api/v1/chat",
                json={"question": "Học phí?"},
                headers={"X-API-Key": "s3cret"},
            )
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(missing.json()["code"], "unauthorized")
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(accepted.status_code, 200)

    def test_api_key_unset_keeps_current_behaviour(self) -> None:
        rag = _rag()
        rag.ask.return_value = Mock(answer="ok", sources=[])
        with patch("app.main.runtime.rag", rag):
            response = TestClient(main.app).post(
                "/api/v1/chat", json={"question": "Học phí?"}
            )
        self.assertEqual(response.status_code, 200)


def _sse_events(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in body.strip().split("\n\n"):
        if not block.strip():
            continue
        name = ""
        data: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:") :].strip())
        events.append((name, json.loads("\n".join(data))))
    return events


class ChatStreamContractTests(unittest.TestCase):
    def _prepared(self, **overrides):
        prepared = Mock()
        prepared.target_year = 2026
        prepared.sources = [
            {
                "source_id": "doc-1",
                "chunk_id": "chunk-1",
                "title": "Tài liệu",
                "url": "https://example.org",
                "year": 2026,
                "category": "03_cutoff_scores",
                "score": 0.9,
            }
        ]
        prepared.direct_answer = None
        for key, value in overrides.items():
            setattr(prepared, key, value)
        return prepared

    def test_event_order_is_meta_tokens_sources_done(self) -> None:
        rag = _rag()
        rag.prepare.return_value = self._prepared()
        rag.stream_prepared.return_value = iter(["Xin ", "chào"])
        with patch("app.main.runtime.rag", rag):
            response = TestClient(main.app).post(
                "/api/v1/chat/stream", json={"question": "Điểm chuẩn 2026?"}
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertEqual(response.headers["x-accel-buffering"], "no")
        self.assertIn("no-transform", response.headers["cache-control"])
        events = _sse_events(response.text)
        self.assertEqual(
            [name for name, _ in events],
            ["meta", "token", "token", "sources", "done"],
        )
        self.assertEqual(
            "".join(payload["text"] for name, payload in events if name == "token"),
            "Xin chào",
        )
        meta = events[0][1]
        self.assertEqual(meta["request_id"], response.headers["x-request-id"])
        self.assertEqual(meta["year"], 2026)
        self.assertEqual(events[3][1]["sources"], self._prepared().sources)
        self.assertEqual(events[4][1], {"request_id": meta["request_id"], "finish_reason": "stop"})
        rag.stream_prepared.assert_called_once()
        rag.ask.assert_not_called()

    def test_abstention_streams_one_fragment_and_marks_finish_reason(self) -> None:
        rag = _rag()
        prepared = self._prepared(
            sources=[],
            direct_answer="Chưa tìm thấy tài liệu phù hợp cho năm 2026.",
        )
        rag.prepare.return_value = prepared
        rag.stream_prepared.return_value = iter([prepared.direct_answer])
        with patch("app.main.runtime.rag", rag):
            response = TestClient(main.app).post(
                "/api/v1/chat/stream", json={"question": "Điểm chuẩn 2026?"}
            )

        events = _sse_events(response.text)
        self.assertEqual([name for name, _ in events], ["meta", "token", "sources", "done"])
        self.assertEqual(events[1][1]["text"], prepared.direct_answer)
        self.assertEqual(events[3][1]["finish_reason"], "abstain")

    def test_provider_failure_after_headers_becomes_a_sanitized_error_event(self) -> None:
        rag = _rag()
        rag.prepare.return_value = self._prepared()

        def failing_stream():
            yield "phần đầu"
            raise ProviderTimeout("slow endpoint", provider="vllm", model="m")

        rag.stream_prepared.return_value = failing_stream()
        with patch("app.main.runtime.rag", rag):
            response = TestClient(main.app).post(
                "/api/v1/chat/stream", json={"question": "Điểm chuẩn 2026?"}
            )

        events = _sse_events(response.text)
        self.assertEqual([name for name, _ in events], ["meta", "token", "error"])
        error = events[-1][1]
        self.assertEqual(error["code"], "provider_timeout")
        self.assertEqual(error["error_type"], "ProviderTimeout")
        self.assertNotIn("slow endpoint", response.text)
        self.assertNotIn("Traceback", response.text)

    def test_stream_uses_the_same_limits_and_authentication(self) -> None:
        rag = _rag()
        rag.prepare.return_value = self._prepared()
        rag.stream_prepared.return_value = iter(["ok"])
        guarded = replace(main.settings, api_key="s3cret")
        with (
            patch("app.main.runtime.rag", rag),
            patch("app.main.settings", guarded),
        ):
            client = TestClient(main.app)
            missing = client.post("/api/v1/chat/stream", json={"question": "Học phí?"})
            accepted = client.post(
                "/api/v1/chat/stream",
                json={"question": "Học phí?"},
                headers={"X-API-Key": "s3cret"},
            )
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(missing.json()["code"], "unauthorized")
        self.assertEqual(accepted.status_code, 200)

    def test_stream_is_unavailable_when_the_corpus_did_not_load(self) -> None:
        with patch("app.main.runtime.rag", None):
            response = TestClient(main.app).post(
                "/api/v1/chat/stream", json={"question": "Học phí?"}
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], "service_unavailable")


class MiddlewareTests(unittest.TestCase):
    def test_body_size_middleware_returns_413(self) -> None:
        mini = FastAPI()
        mini.add_middleware(BodySizeLimitMiddleware, max_bytes=10)

        @mini.post("/api/v1/chat")
        async def chat() -> dict:
            return {"ok": True}

        response = TestClient(mini).post("/api/v1/chat", content=b"x" * 11)
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["code"], "payload_too_large")

    def test_rate_limit_middleware_is_opt_in_and_counts_chat_requests(self) -> None:
        mini = FastAPI()
        mini.add_middleware(RateLimitMiddleware, limit=2)

        @mini.post("/api/v1/chat")
        async def chat() -> dict:
            return {"ok": True}

        client = TestClient(mini)
        self.assertEqual(client.post("/api/v1/chat").status_code, 200)
        self.assertEqual(client.post("/api/v1/chat").status_code, 200)
        limited = client.post("/api/v1/chat")
        self.assertEqual(limited.status_code, 429)
        self.assertEqual(limited.json()["code"], "rate_limited")
        self.assertIn("retry-after", limited.headers)

    def test_rate_limit_does_not_apply_when_disabled(self) -> None:
        mini = FastAPI()
        mini.add_middleware(RateLimitMiddleware, limit=0)

        @mini.post("/api/v1/chat")
        async def chat() -> dict:
            return {"ok": True}

        client = TestClient(mini)
        for _ in range(5):
            self.assertEqual(client.post("/api/v1/chat").status_code, 200)


if __name__ == "__main__":
    unittest.main()
