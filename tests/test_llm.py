"""Provider abstraction, prompt profiles, and the shared factory."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.factory import build_pipeline, index_ready
from rag.llm import (
    LLMConfig,
    OpenAICompatibleProvider,
    ProviderBadResponse,
    ProviderConfigurationError,
    ProviderEmptyAnswer,
    ProviderHTTPError,
    ProviderTimeout,
    ProviderUnreachable,
    build_provider,
)
from rag.prompts import build_prompt_builder
from rag.prompts.generic import create_messages as generic_messages
from rag.prompts.haui_admission import create_messages as haui_messages


def _provider(**overrides) -> OpenAICompatibleProvider:
    options = {
        "base_url": "http://llm.internal:8000/v1",
        "model": "served-model",
        "api_key": "secret-key",
        "timeout": 5,
    }
    options.update(overrides)
    return OpenAICompatibleProvider(**options)


def _response(**overrides) -> Mock:
    defaults = {
        "status_code": 200,
        "text": '{"ok":true}',
        "json": lambda: {"choices": [{"message": {"content": "answer"}}]},
    }
    defaults.update(overrides)
    return Mock(**defaults)


class LLMConfigTests(unittest.TestCase):
    def test_llm_names_are_primary(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "ollama",
                "LLM_BASE_URL": "http://localhost:11434/v1",
                "LLM_MODEL": "qwen3.5:4b",
                "LLM_API_KEY": "k",
                "LLM_TIMEOUT": "7",
                "LLM_TEMPERATURE": "0.2",
                "LLM_MAX_TOKENS": "256",
                "LLM_TOP_P": "0.9",
                "LLM_EXTRA_BODY": '{"seed": 7}',
            },
            clear=True,
        ):
            config = LLMConfig.from_env()
        self.assertEqual(config.provider, "ollama")
        self.assertEqual(config.base_url, "http://localhost:11434/v1")
        self.assertEqual(config.model, "qwen3.5:4b")
        self.assertEqual(config.api_key, "k")
        self.assertEqual(config.timeout, 7)
        self.assertEqual(config.temperature, 0.2)
        self.assertEqual(config.max_tokens, 256)
        self.assertEqual(config.top_p, 0.9)
        self.assertEqual(config.extra_body, {"seed": 7})

    def test_vllm_names_still_work_as_aliases(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VLLM_BASE_URL": "http://localhost:8000/v1",
                "VLLM_MODEL": "legacy-model",
                "VLLM_API_KEY": "legacy-key",
            },
            clear=True,
        ):
            config = LLMConfig.from_env()
        self.assertEqual(config.provider, "openai_compatible")
        self.assertEqual(config.base_url, "http://localhost:8000/v1")
        self.assertEqual(config.model, "legacy-model")
        self.assertEqual(config.api_key, "legacy-key")

    def test_empty_api_key_is_none_and_defaults_hold(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = LLMConfig.from_env()
        self.assertIsNone(config.api_key)
        self.assertEqual(config.provider, "openai_compatible")
        self.assertEqual(config.max_tokens, 800)

    def test_bad_values_are_rejected(self) -> None:
        with patch.dict(os.environ, {"LLM_PROVIDER": "nope"}, clear=True):
            with self.assertRaises(ValueError):
                LLMConfig.from_env()
        with patch.dict(os.environ, {"LLM_TIMEOUT": "0"}, clear=True):
            with self.assertRaises(ValueError):
                LLMConfig.from_env()
        with patch.dict(os.environ, {"LLM_TEMPERATURE": "3"}, clear=True):
            with self.assertRaises(ValueError):
                LLMConfig.from_env()
        with patch.dict(os.environ, {"LLM_EXTRA_BODY": "[]"}, clear=True):
            with self.assertRaises(ValueError):
                LLMConfig.from_env()

    def test_build_provider_passes_provider_name_and_model(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "ollama",
                "LLM_MODEL": "qwen3.5:4b",
                "LLM_BASE_URL": "http://localhost:11434/v1",
            },
            clear=True,
        ):
            provider = build_provider()
        self.assertEqual(provider.name, "ollama")
        self.assertEqual(provider.model, "qwen3.5:4b")

    def test_switching_provider_does_not_require_rebuilding_embeddings(self) -> None:
        # Generation config is separate by construction: LLMConfig has no embedding
        # or index fields, so changing LLM_MODEL cannot invalidate a vector index.
        self.assertFalse(
            {field for field in LLMConfig.__dataclass_fields__} & {"index_dir", "embedding_model"}
        )


class ProviderBehaviourTests(unittest.TestCase):
    def test_sends_configured_generation_parameters_and_never_overrides_messages(self) -> None:
        session = Mock()
        session.post.return_value = _response()
        provider = _provider(
            temperature=0.3,
            max_tokens=123,
            top_p=0.8,
            extra_body={"seed": 9, "model": "hijack", "messages": [], "stream": True},
            session=session,
        )
        answer = provider.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(answer, "answer")
        body = session.post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "served-model")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(body["temperature"], 0.3)
        self.assertEqual(body["max_tokens"], 123)
        self.assertEqual(body["top_p"], 0.8)
        self.assertEqual(body["seed"], 9)
        self.assertEqual(
            session.post.call_args.kwargs["headers"]["Authorization"],
            "Bearer secret-key",
        )

    def test_missing_model_is_a_configuration_error(self) -> None:
        with self.assertRaises(ProviderConfigurationError):
            _provider(model="")
        with self.assertRaises(ProviderConfigurationError):
            _provider(base_url="not-a-url")

    def test_timeout_is_typed(self) -> None:
        session = Mock()
        session.post.side_effect = requests.Timeout("slow")
        with self.assertRaises(ProviderTimeout) as caught:
            _provider(session=session).complete([{"role": "user", "content": "q"}])
        self.assertEqual(caught.exception.code, "provider_timeout")

    def test_connection_failure_is_typed_and_hides_the_endpoint(self) -> None:
        session = Mock()
        session.post.side_effect = requests.ConnectionError(
            "cannot reach http://llm.internal:8000/v1"
        )
        with self.assertRaises(ProviderUnreachable) as caught:
            _provider(session=session).complete([{"role": "user", "content": "q"}])
        self.assertEqual(caught.exception.code, "provider_unreachable")
        self.assertNotIn("llm.internal", str(caught.exception))
        self.assertNotIn("secret-key", str(caught.exception))

    def test_http_error_keeps_status_and_truncates_body(self) -> None:
        session = Mock()
        session.post.return_value = _response(status_code=500, text="x" * 5000)
        with self.assertRaises(ProviderHTTPError) as caught:
            _provider(session=session).complete([{"role": "user", "content": "q"}])
        self.assertEqual(caught.exception.status_code, 500)
        self.assertEqual(caught.exception.code, "provider_http_error")
        self.assertLess(len(str(caught.exception)), 1200)

    def test_non_json_body_is_typed(self) -> None:
        session = Mock()
        session.post.return_value = _response(json=Mock(side_effect=ValueError("bad")))
        with self.assertRaises(ProviderBadResponse):
            _provider(session=session).complete([{"role": "user", "content": "q"}])

    def test_missing_choices_is_typed(self) -> None:
        session = Mock()
        session.post.return_value = _response(json=lambda: {"choices": []})
        with self.assertRaises(ProviderBadResponse):
            _provider(session=session).complete([{"role": "user", "content": "q"}])

    def test_empty_answer_is_typed(self) -> None:
        session = Mock()
        session.post.return_value = _response(
            json=lambda: {"choices": [{"message": {"content": "   "}}]}
        )
        with self.assertRaises(ProviderEmptyAnswer):
            _provider(session=session).complete([{"role": "user", "content": "q"}])

    def test_reasoning_only_answer_explains_the_token_budget_fix(self) -> None:
        session = Mock()
        session.post.return_value = _response(
            json=lambda: {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning": "Thinking about the question...",
                        }
                    }
                ]
            }
        )
        with self.assertRaises(ProviderEmptyAnswer) as caught:
            _provider(session=session).complete([{"role": "user", "content": "q"}])
        self.assertIn("LLM_MAX_TOKENS", str(caught.exception))

    def test_probe_reports_reachability_without_raising(self) -> None:
        session = Mock()
        session.get.return_value = Mock(status_code=200)
        self.assertTrue(_provider(session=session).probe())
        session.get.side_effect = requests.ConnectionError("down")
        self.assertFalse(_provider(session=session).probe())


class ProviderStreamTests(unittest.TestCase):
    def _stream_response(self, lines, **overrides) -> Mock:
        response = _response(**overrides)
        response.iter_lines.return_value = lines
        return response

    def test_stream_requests_sse_and_yields_deltas_in_order(self) -> None:
        session = Mock()
        response = self._stream_response(
            [
                b": keep-alive comment",
                b"",
                b'data: {"choices":[{"delta":{"content":"Xin "}}]}',
                b'data: {"choices":[{"delta":{"content":"chao"}}]}',
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
                b"data: [DONE]",
            ]
        )
        session.post.return_value = response
        provider = _provider(session=session)

        chunks = list(provider.stream([{"role": "user", "content": "hi"}]))

        self.assertEqual(chunks, ["Xin ", "chao"])
        body = session.post.call_args.kwargs["json"]
        self.assertTrue(body["stream"])
        self.assertEqual(
            session.post.call_args.kwargs["headers"]["Accept"], "text/event-stream"
        )
        self.assertTrue(session.post.call_args.kwargs["stream"])
        response.close.assert_called_once()

    def test_stream_accepts_message_frames_from_non_streaming_vendors(self) -> None:
        session = Mock()
        session.post.return_value = self._stream_response(
            [b'data: {"choices":[{"message":{"content":"mot"}}]}']
        )
        self.assertEqual(
            list(_provider(session=session).stream([{"role": "user", "content": "hi"}])),
            ["mot"],
        )

    def test_empty_stream_is_typed_and_closed(self) -> None:
        session = Mock()
        response = self._stream_response([b"data: [DONE]"])
        session.post.return_value = response
        with self.assertRaises(ProviderEmptyAnswer):
            list(_provider(session=session).stream([{"role": "user", "content": "hi"}]))
        response.close.assert_called_once()

    def test_stream_http_error_is_typed_and_closed(self) -> None:
        session = Mock()
        response = self._stream_response([], status_code=502, text="upstream down")
        session.post.return_value = response
        with self.assertRaises(ProviderHTTPError) as caught:
            list(_provider(session=session).stream([{"role": "user", "content": "hi"}]))
        self.assertEqual(caught.exception.status_code, 502)
        response.close.assert_called_once()

    def test_stream_malformed_frame_is_typed_and_closed(self) -> None:
        session = Mock()
        response = self._stream_response([b"data: not-json"])
        session.post.return_value = response
        with self.assertRaises(ProviderBadResponse):
            list(_provider(session=session).stream([{"role": "user", "content": "hi"}]))
        response.close.assert_called_once()

    def test_closing_the_generator_releases_the_http_response(self) -> None:
        session = Mock()
        response = self._stream_response(
            [
                b'data: {"choices":[{"delta":{"content":"one"}}]}',
                b'data: {"choices":[{"delta":{"content":"two"}}]}',
            ]
        )
        session.post.return_value = response
        stream = _provider(session=session).stream([{"role": "user", "content": "hi"}])
        self.assertEqual(next(stream), "one")
        stream.close()
        response.close.assert_called_once()


class PromptProfileTests(unittest.TestCase):
    def test_haui_profile_keeps_scope_and_year_rules(self) -> None:
        messages = haui_messages("Trường khác tuyển sinh thế nào?", [])
        self.assertIn("chỉ hỗ trợ thông tin tuyển sinh HaUI", messages[0]["content"])
        self.assertIn("chỉ dùng dữ kiện đúng năm đó", messages[0]["content"])

    def test_generic_profile_has_no_haui_policy_and_fences_context(self) -> None:
        chunk = {
            "chunk_id": "c1",
            "document_id": "d1",
            "title": "Doc",
            "source_url": "https://example.org",
            "text": "Body",
        }
        messages = generic_messages("Question?", [(1.0, chunk)])
        system = messages[0]["content"]
        self.assertNotIn("HaUI", system)
        self.assertNotIn("chỉ hỗ trợ", system)
        self.assertIn("<context>", messages[-1]["content"])
        self.assertIn("</context>", messages[-1]["content"])

    def test_profiles_are_selectable(self) -> None:
        self.assertEqual(build_prompt_builder("haui_admission").name, "haui_admission")
        self.assertEqual(build_prompt_builder("generic").name, "generic")
        with self.assertRaises(ValueError):
            build_prompt_builder("unknown")


class FakeEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0] if "học bổng" in text else [0.0, 1.0] for text in texts]


class FactoryTests(unittest.TestCase):
    def test_index_ready_is_a_cheap_file_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.assertFalse(index_ready(temporary))
            (Path(temporary) / "vectors.faiss").write_bytes(b"")
            self.assertTrue(index_ready(temporary))

    def _chunks_dir(self, root: Path) -> Path:
        chunks_dir = root / "final"
        chunks_dir.mkdir()
        chunks = [
            {
                "chunk_id": "c1",
                "document_id": "d1",
                "title": "Học bổng",
                "text": "học bổng mức 1 cho sinh viên",
                "source_url": "https://example.org/1",
                "category": "05_tuition_scholarship",
                "status": "final",
            },
            {
                "chunk_id": "c2",
                "document_id": "d2",
                "title": "Học phí",
                "text": "học phí học kỳ một",
                "source_url": "https://example.org/2",
                "status": "final",
            },
        ]
        (chunks_dir / "category.jsonl").write_text(
            "\n".join(json.dumps(chunk) for chunk in chunks) + "\n",
            encoding="utf-8",
        )
        state_file = root / "documents.json"
        state_file.write_text(
            json.dumps({"d1": {"status": "indexed"}, "d2": {"status": "indexed"}}),
            encoding="utf-8",
        )
        return chunks_dir

    def test_both_providers_share_identical_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chunks_dir = self._chunks_dir(root)
            state_file = root / "documents.json"
            index_dir = root / "index"

            def provider(provider_name: str, base_url: str):
                return build_provider(
                    LLMConfig(
                        provider=provider_name,
                        base_url=base_url,
                        model="same-model",
                    )
                )

            def pipeline(configured_provider):
                return build_pipeline(
                    chunks_dir,
                    index_dir,
                    provider=configured_provider,
                    prompt_builder=build_prompt_builder("haui_admission"),
                    alpha=0.5,
                    score_threshold=0.0,
                    embedder=FakeEmbedder(),
                    use_reranker=False,
                )

            with patch("rag.pipeline.DEFAULT_STATE_FILE", state_file):
                vllm_like = pipeline(
                    provider("vllm", "http://localhost:8000/v1")
                )
                ollama_like = pipeline(
                    provider("ollama", "http://localhost:11434/v1")
                )
                first = vllm_like.retrieve("học bổng", top_k=2)
                second = ollama_like.retrieve("học bổng", top_k=2)

        self.assertEqual(
            [(score, chunk["chunk_id"]) for score, chunk in first],
            [(score, chunk["chunk_id"]) for score, chunk in second],
        )
        self.assertEqual(first[0][1]["chunk_id"], "c1")

    def test_provider_path_generates_from_messages_and_sources_come_from_retrieval(self) -> None:
        class FakeProvider:
            name = "fake"
            model = "fake-model"

            def __init__(self) -> None:
                self.messages = None

            def complete(self, messages):
                self.messages = messages
                return "MODEL_TEXT <script>alert(1)</script>"

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chunks_dir = self._chunks_dir(root)
            state_file = root / "documents.json"
            provider = FakeProvider()
            with patch("rag.pipeline.DEFAULT_STATE_FILE", state_file):
                pipeline = build_pipeline(
                    chunks_dir,
                    root / "index",
                    provider=provider,
                    prompt_builder=build_prompt_builder("haui_admission"),
                    alpha=0.5,
                    score_threshold=0.0,
                    embedder=FakeEmbedder(),
                    use_reranker=False,
                )
                response = pipeline.ask(
                    "học bổng",
                    top_k=1,
                    history=[{"role": "user", "content": "trước đó"}],
                )

        self.assertEqual(response.answer, "MODEL_TEXT <script>alert(1)</script>")
        self.assertEqual([source["chunk_id"] for source in response.sources], ["c1"])
        self.assertIsNotNone(provider.messages)
        self.assertEqual(provider.messages[-1]["role"], "user")
        self.assertIn("NGỮ CẢNH", provider.messages[-1]["content"])
        self.assertIn("trước đó", [m["content"] for m in provider.messages])

    def test_factory_passes_provider_and_prompt_profile_to_the_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chunks_dir = self._chunks_dir(root)
            state_file = root / "documents.json"
            provider = _provider()
            builder = build_prompt_builder("generic")
            with patch("rag.pipeline.DEFAULT_STATE_FILE", state_file):
                pipeline = build_pipeline(
                    chunks_dir,
                    root / "index",
                    provider=provider,
                    prompt_builder=builder,
                    embedder=FakeEmbedder(),
                    use_reranker=False,
                )
        self.assertIs(pipeline.provider, provider)
        self.assertIs(pipeline.prompt_builder, builder)
        self.assertEqual(pipeline.provider_name, "openai_compatible")
        self.assertEqual(pipeline.model_name, "served-model")
        self.assertEqual(pipeline.chunk_count, 2)


if __name__ == "__main__":
    unittest.main()
