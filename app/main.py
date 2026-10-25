from __future__ import annotations

import asyncio
import json
import logging
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from app.core import middleware
from app.core.config import Settings, get_settings
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ReadyResponse,
    ReadinessChecks,
)
from app.services.rag_service import RagService
from rag.factory import index_ready
from rag.llm import LLMConfig, build_provider

APP_VERSION = "1.1.0"

logger = logging.getLogger("app")


@dataclass
class Runtime:
    settings: Settings
    rag: RagService | None = None
    startup_error: str | None = None
    provider_name: str = ""
    model_name: str = ""
    model_configured: bool = False


settings = get_settings()
runtime = Runtime(settings=settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        config = LLMConfig.from_env()
        runtime.provider_name = config.provider
        runtime.model_name = config.model
        runtime.model_configured = bool(config.model)
        provider = build_provider(config)
        runtime.rag = RagService(
            settings.chunks_dir,
            settings.index_dir,
            top_k=settings.top_k,
            provider=provider,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        runtime.startup_error = str(exc)
    yield


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "") or uuid4())


def _error(
    detail: str,
    code: str,
    status_code: int,
    request: Request,
    *,
    extra: dict | None = None,
) -> JSONResponse:
    content = {"detail": detail, "code": code, "request_id": _request_id(request)}
    if extra:
        content.update(extra)
    return JSONResponse(status_code=status_code, content=content)


def _sse(event: str, payload: dict) -> str:
    """Encode one Server-Sent Event without allowing raw newlines in JSON."""
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"


def _chat_guard(payload: ChatRequest, request: Request) -> JSONResponse | None:
    """Shared authentication/limits/corpus gate for JSON and SSE chat."""
    if settings.api_key:
        provided = request.headers.get("x-api-key", "")
        if not secrets.compare_digest(provided, settings.api_key):
            return _error(
                "Yêu cầu không được xác thực.",
                "unauthorized",
                401,
                request,
            )

    if len(payload.question) > settings.max_question_length:
        return _error(
            "Câu hỏi vượt quá giới hạn cho phép.",
            "question_too_long",
            422,
            request,
        )
    if len(payload.history) > settings.max_history_messages:
        return _error(
            "Lịch sử hội thoại vượt quá giới hạn cho phép.",
            "history_too_long",
            422,
            request,
        )
    if runtime.rag is None:
        return _error(
            "Dịch vụ tư vấn chưa sẵn sàng. Vui lòng thử lại sau.",
            "service_unavailable",
            503,
            request,
        )
    return None


app = FastAPI(
    title="HaUI Admissions Assistant API",
    version=APP_VERSION,
    description=(
        "Evidence-grounded admissions answers backed by the HaUI RAG corpus. "
        "Generation is LLM-agnostic through an OpenAI-compatible provider."
    ),
    lifespan=lifespan,
)
middleware.install(app, settings)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Never echo the raw input or pydantic internals.
    return _error(
        "Dữ liệu yêu cầu không hợp lệ.",
        "invalid_request",
        422,
        request,
    )


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error request_id=%s", _request_id(request))
    return _error(
        "Không thể xử lý yêu cầu lúc này. Vui lòng thử lại sau.",
        "internal_error",
        500,
        request,
    )


@app.get("/api/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Fast liveness check. Always HTTP 200, even when degraded.

    Reports disk-level index presence only; it never loads an embedding model or
    calls the LLM endpoint. Use ``/api/v1/ready`` for a deep check.
    """
    rag_loaded = runtime.rag is not None
    has_index = index_ready(settings.index_dir)
    return HealthResponse(
        status="ready" if rag_loaded and has_index else "degraded",
        rag_loaded=rag_loaded,
        model_configured=runtime.model_configured,
        index_ready=has_index,
        chunk_count=runtime.rag.chunk_count if runtime.rag else 0,
        provider=runtime.rag.provider_name if runtime.rag else runtime.provider_name,
        model=runtime.rag.model_name if runtime.rag else runtime.model_name,
        app_version=APP_VERSION,
    )


@app.get("/api/v1/ready", response_model=ReadyResponse)
async def ready(request: Request):
    """Deep readiness: chunks, on-disk index, and a short provider probe.

    200 only when every check passes; 503 otherwise, so Docker and orchestrators
    fail the healthcheck instead of routing traffic to a half-ready instance.
    """
    chunks_loaded = runtime.rag is not None
    has_index = index_ready(settings.index_dir)
    provider_reachable = (
        await run_in_threadpool(runtime.rag.probe) if runtime.rag is not None else False
    )
    is_ready = chunks_loaded and has_index and provider_reachable
    body = ReadyResponse(
        status="ready" if is_ready else "degraded",
        checks=ReadinessChecks(
            chunks_loaded=chunks_loaded,
            index_ready=has_index,
            provider_reachable=provider_reachable,
        ),
        chunk_count=runtime.rag.chunk_count if runtime.rag else 0,
        provider=runtime.rag.provider_name if runtime.rag else runtime.provider_name,
        model=runtime.rag.model_name if runtime.rag else runtime.model_name,
        app_version=APP_VERSION,
        request_id=_request_id(request),
    )
    return JSONResponse(
        status_code=200 if is_ready else 503,
        content=body.model_dump(),
    )


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request):
    request_id = _request_id(request)

    guard_error = _chat_guard(payload, request)
    if guard_error is not None:
        return guard_error
    assert runtime.rag is not None

    provider_label = str(getattr(runtime.rag, "provider_name", "") or "-")
    model_label = str(getattr(runtime.rag, "model_name", "") or "-")
    request.state.provider = provider_label
    request.state.model = model_label

    try:
        # requests + fastembed are synchronous; keep them off the event loop so one
        # question cannot block every other request served by this worker.
        response = await run_in_threadpool(
            runtime.rag.ask,
            payload.question,
            [message.model_dump() for message in payload.history],
        )
    except (RuntimeError, ValueError) as exc:
        logger.warning(
            "generation failed request_id=%s provider=%s model=%s error=%s",
            request_id,
            provider_label,
            model_label,
            type(exc).__name__,
        )
        return _error(
            "Không thể tạo câu trả lời từ mô hình lúc này.",
            str(getattr(exc, "code", "provider_error")),
            502,
            request,
            extra={"error_type": type(exc).__name__},
        )
    return ChatResponse(
        answer=response.answer,
        sources=response.sources,
        request_id=request_id,
    )


@app.post("/api/v1/chat/stream")
async def chat_stream(payload: ChatRequest, request: Request):
    """Server-Sent Events version of ``/chat``.

    Event order is ``meta``, zero or more ``token``, ``sources``, ``done``. A
    provider/retrieval failure during the stream becomes a sanitized ``error``
    event because HTTP headers have already been sent. The endpoint never builds
    the full answer before forwarding tokens, and cancellation closes the
    provider iterator.
    """
    request_id = _request_id(request)
    guard_error = _chat_guard(payload, request)
    if guard_error is not None:
        return guard_error
    assert runtime.rag is not None

    provider_label = str(getattr(runtime.rag, "provider_name", "") or "-")
    model_label = str(getattr(runtime.rag, "model_name", "") or "-")
    request.state.provider = provider_label
    request.state.model = model_label
    history = [message.model_dump() for message in payload.history]

    async def events():
        iterator = None
        prepared = None
        try:
            prepared = await run_in_threadpool(
                runtime.rag.prepare,
                payload.question,
                history,
            )
            yield _sse(
                "meta",
                {
                    "request_id": request_id,
                    "provider": provider_label,
                    "model": model_label,
                    "year": prepared.target_year,
                },
            )

            iterator = runtime.rag.stream_prepared(prepared)
            async for token in iterate_in_threadpool(iterator):
                if token:
                    yield _sse("token", {"text": token})

            # Sources are backend-built, never parsed from model output.
            yield _sse("sources", {"sources": prepared.sources})
            yield _sse(
                "done",
                {
                    "request_id": request_id,
                    "finish_reason": "abstain" if prepared.direct_answer else "stop",
                },
            )
        except asyncio.CancelledError:
            # Client disconnected or the server is shutting down. Re-raise so
            # Starlette can finish the ASGI response; finally closes the provider
            # stream and releases its HTTP connection.
            raise
        except (RuntimeError, ValueError) as exc:
            logger.warning(
                "streaming generation failed request_id=%s provider=%s model=%s error=%s",
                request_id,
                provider_label,
                model_label,
                type(exc).__name__,
            )
            yield _sse(
                "error",
                {
                    "code": str(getattr(exc, "code", "provider_error")),
                    "message": "Không thể tạo câu trả lời từ mô hình lúc này.",
                    "request_id": request_id,
                    "error_type": type(exc).__name__,
                },
            )
        except Exception:
            logger.exception("unhandled streaming error request_id=%s", request_id)
            yield _sse(
                "error",
                {
                    "code": "internal_error",
                    "message": "Không thể xử lý yêu cầu lúc này. Vui lòng thử lại sau.",
                    "request_id": request_id,
                },
            )
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover - defensive cleanup
                    logger.debug("stream iterator close failed request_id=%s", request_id)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
