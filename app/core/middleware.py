"""Small ASGI middlewares: request id/logging, body-size limit, opt-in rate limit.

No new dependency: everything is Starlette/ASGI standard library code. None of
these middlewares ever reads or logs the question text.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from uuid import uuid4

from starlette.requests import Request
from starlette.responses import JSONResponse

from app.core.config import Settings

logger = logging.getLogger("app.request")

REQUEST_ID_HEADER = "X-Request-ID"


def _request_id(scope: dict) -> str:
    state = scope.setdefault("state", {})
    return state.setdefault("request_id", str(uuid4()))


def _error(
    detail: str,
    code: str,
    status_code: int,
    request_id: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail, "code": code, "request_id": request_id},
        headers=headers,
    )


class RequestContextMiddleware:
    """Assign a request id, echo it, and log method/path/status/duration.

    The chat handler may put ``provider`` and ``model`` on the request state; they
    are included in the log line when present. Question text is never logged.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _request_id(scope)
        started = time.perf_counter()
        status_code = 500

        async def send_with_header(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER.lower().encode(), request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_header)
        finally:
            state = scope.get("state", {})
            logger.info(
                "request_id=%s method=%s path=%s status=%s duration_ms=%.1f "
                "provider=%s model=%s",
                request_id,
                scope.get("method", ""),
                scope.get("path", ""),
                status_code,
                (time.perf_counter() - started) * 1000,
                state.get("provider", "-"),
                state.get("model", "-"),
            )


class BodySizeLimitMiddleware:
    """Reject an oversized request from ``Content-Length`` before parsing it."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        raw_length = None
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                raw_length = value
                break
        if raw_length is not None:
            try:
                length = int(raw_length)
            except ValueError:
                length = 0
            if length > self.max_bytes:
                request_id = _request_id(scope)
                response = _error(
                    "Yêu cầu vượt quá kích thước cho phép.",
                    "payload_too_large",
                    413,
                    request_id,
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


class RateLimitMiddleware:
    """Opt-in fixed-window limiter for chat requests, keyed by client host.

    Disabled when ``limit`` is 0. In-process only; a multi-worker deployment must
    enforce a shared limit at the gateway instead.
    """

    def __init__(self, app, limit: int, window_seconds: float = 60.0):
        self.app = app
        self.limit = limit
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or self.limit <= 0:
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if scope.get("method") != "POST" or not path.startswith("/api/v1/chat"):
            await self.app(scope, receive, send)
            return

        client = scope.get("client") or ("unknown", 0)
        key = str(client[0])
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            cutoff = now - self.window_seconds
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.limit:
                retry_after = max(1, int(self.window_seconds - (now - hits[0])))
                allowed = False
            else:
                hits.append(now)
                retry_after = 0
                allowed = True

        if not allowed:
            request_id = _request_id(scope)
            response = _error(
                "Bạn đã gửi quá nhiều yêu cầu. Vui lòng thử lại sau.",
                "rate_limited",
                429,
                request_id,
                headers={"Retry-After": str(retry_after)},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def client_host(request: Request) -> str:
    return str(getattr(request.client, "host", "") or "")


def install(app, settings: Settings) -> None:
    """Attach the middlewares in the intended order.

    ``add_middleware`` makes the last-added middleware outermost, so request
    context wraps the others and CORS remains outermost (it must decorate error
    responses too).
    """
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        RateLimitMiddleware,
        limit=settings.rate_limit_per_minute,
    )
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_bytes=settings.max_request_bytes,
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
    )
