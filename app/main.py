from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import Settings, get_settings
from app.schemas.chat import ChatRequest, ChatResponse, HealthResponse
from app.services.rag_service import RagService


@dataclass
class Runtime:
    settings: Settings
    rag: RagService | None = None
    startup_error: str | None = None


settings = get_settings()
runtime = Runtime(settings=settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        runtime.rag = RagService(settings.chunks_dir, settings.index_dir)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        runtime.startup_error = str(exc)
    yield


app = FastAPI(
    title="HaUI Admissions Assistant API",
    version="1.0.0",
    description="Evidence-grounded admissions answers backed by the existing HaUI RAG corpus.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(Exception)
async def unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    request_id = str(uuid4())
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Không thể xử lý yêu cầu lúc này. Vui lòng thử lại sau.",
            "request_id": request_id,
        },
    )


@app.get("/api/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    rag_loaded = runtime.rag is not None
    return HealthResponse(
        status="ready" if rag_loaded else "degraded",
        rag_loaded=rag_loaded,
        model_configured=bool(__import__("os").getenv("VLLM_MODEL")),
    )


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    request_id = str(uuid4())
    if len(payload.question) > settings.max_question_length:
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Câu hỏi vượt quá giới hạn cho phép.",
                "request_id": request_id,
            },
        )
    if len(payload.history) > settings.max_history_messages:
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Lịch sử hội thoại vượt quá giới hạn cho phép.",
                "request_id": request_id,
            },
        )
    if runtime.rag is None:
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Dịch vụ tư vấn chưa sẵn sàng. Vui lòng thử lại sau.",
                "request_id": request_id,
            },
        )
    try:
        response = runtime.rag.ask(
            payload.question,
            [message.model_dump() for message in payload.history],
        )
    except (RuntimeError, ValueError) as exc:
        return JSONResponse(
            status_code=502,
            content={
                "detail": "Không thể tạo câu trả lời từ mô hình lúc này.",
                "request_id": request_id,
                "error_type": type(exc).__name__,
            },
        )
    return ChatResponse(
        answer=response.answer,
        sources=response.sources,
        request_id=request_id,
    )
