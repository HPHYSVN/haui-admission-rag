# API contract

Base URL in development: `http://localhost:8001`. All endpoints are versioned
under `/api/v1`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | cheap liveness; always HTTP 200 |
| `GET` | `/api/v1/ready` | deep readiness; 200 or 503 |
| `POST` | `/api/v1/chat` | ask one question (JSON) |
| `POST` | `/api/v1/chat/stream` | ask one question (SSE token stream) |

OpenAPI is served at `/docs` and `/openapi.json`.

## `POST /api/v1/chat`

Request:

```json
{
  "question": "HaUI có những phương thức xét tuyển nào năm 2026?",
  "history": [
    { "role": "user", "content": "Tôi muốn hỏi về tuyển sinh HaUI." },
    { "role": "assistant", "content": "Bạn muốn hỏi mục nào?" }
  ]
}
```

- `question`: 1–2000 characters (`APP_MAX_QUESTION_LENGTH`), whitespace-only is rejected.
- `history`: at most 8 messages (`APP_MAX_HISTORY_MESSAGES`), each 1–4000 characters.
  Roles are `user` or `assistant`. The browser owns history; the backend persists
  nothing.

Response `200`:

```json
{
  "answer": "Câu trả lời dựa trên tài liệu đã truy xuất.",
  "sources": [
    {
      "source_id": "01_admission_001",
      "chunk_id": "01_admission_001__section-001__chunk-001",
      "title": "Thông tin tuyển sinh",
      "url": "https://tuyensinh.haui.edu.vn/...",
      "year": 2026,
      "category": "01_admission",
      "score": 0.91
    }
  ],
  "request_id": "9f4c2e0e-..."
}
```

Source fields are stable and are built by the backend from retrieved chunks, not
by the model. `score` is the fused retrieval score, not a probability.

If retrieval finds nothing relevant, generation is skipped and `sources` is an
empty array.

### curl

```bash
curl -s http://localhost:8001/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"Học phí năm 2026 là bao nhiêu?","history":[]}'
```

With the optional key gate enabled:

```bash
curl -s http://localhost:8001/api/v1/chat \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $APP_API_KEY" \
  -d '{"question":"Học phí năm 2026 là bao nhiêu?"}'
```

### Browser fetch

```js
const response = await fetch("/api/v1/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question, history: history.slice(-8) }),
});
const body = await response.json();
if (!response.ok) throw new Error(body.detail);
```

Render `body.answer` as text (`{answer}` in React). Do not inject it as HTML.

## Streaming — `POST /api/v1/chat/stream`

Same request body, limits, authentication, and retrieval policy as
`POST /api/v1/chat`. The response is `text/event-stream`; the endpoint does not
buffer the full answer before forwarding tokens.

Event order is fixed:

1. `meta` — `request_id`, `provider`, `model`, resolved `year`.
2. `token` — zero or more events, each with a text fragment in `text`.
3. `sources` — backend-built source array (same shape as the JSON endpoint).
4. `done` — `request_id`, `finish_reason` (`stop` or `abstain`).

```text
event: meta
data: {"request_id":"9f4c2e0e-…","provider":"vllm","model":"…","year":2026}

event: token
data: {"text":"HaUI"}

event: token
data: {"text":" có 8.300 chỉ tiêu"}

event: sources
data: {"sources":[{…}]}

event: done
data: {"request_id":"9f4c2e0e-…","finish_reason":"stop"}
```

An abstention is sent as one `token` event containing the safe Vietnamese
message, an empty `sources` array, then `done` with `finish_reason: "abstain"`.
No provider call is made in that case.

### Errors after the stream starts

Validation, authentication, body-size, rate-limit, and corpus-not-ready failures
still happen before headers are sent and use the normal JSON error envelope with
the corresponding HTTP status. Once the stream is open, a provider failure is a
sanitized terminal event instead of an HTTP status change:

```text
event: error
data: {"code":"provider_timeout","message":"Không thể tạo câu trả lời từ mô hình lúc này.","request_id":"…","error_type":"ProviderTimeout"}
```

A client must treat the stream as incomplete if it ends without `done` or
`error`. Closing the connection cancels the request, closes the provider
iterator, and releases its upstream HTTP connection.

### curl

```bash
curl -N http://localhost:8001/api/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"question":"Học phí năm 2026 là bao nhiêu?","history":[]}'
```

Use `fetch` + `ReadableStream` in browsers (the current frontend does), not
`EventSource`, because the request body and `X-API-Key` header are required.

Reverse proxies must disable response buffering for this path. The bundled Nginx
configuration has an exact `/api/v1/chat/stream` location with
`proxy_buffering off`, `proxy_cache off`, and a 300-second read timeout; the app
also sends `X-Accel-Buffering: no`.

## Error envelope

Every error body has the same three fields, plus `error_type` on generation
failures:

```json
{
  "detail": "Thông báo tiếng Việt an toàn để hiển thị.",
  "code": "stable_machine_readable_code",
  "request_id": "9f4c2e0e-...",
  "error_type": "ProviderTimeout"
}
```

| HTTP | `code` | Meaning |
|---|---|---|
| 401 | `unauthorized` | `APP_API_KEY` is set and `X-API-Key` is missing/wrong |
| 413 | `payload_too_large` | body exceeds `APP_MAX_REQUEST_BYTES` (checked before parsing) |
| 422 | `invalid_request` | pydantic shape/type/empty-question failure |
| 422 | `question_too_long` | question exceeds `APP_MAX_QUESTION_LENGTH` |
| 422 | `history_too_long` | more than `APP_MAX_HISTORY_MESSAGES` |
| 429 | `rate_limited` | `APP_RATE_LIMIT_PER_MINUTE` exceeded; see `Retry-After` |
| 502 | `provider_*` | generation failed; `error_type` keeps the old field working |
| 503 | `service_unavailable` | corpus/index not loaded at startup |
| 500 | `internal_error` | unexpected failure; internals are never returned |

`error_type` is retained for backward compatibility; the documented stable field
is `code`.

Every response also carries `X-Request-ID`, matching `request_id`.

## `GET /api/v1/health`

Always HTTP 200, even when degraded, so the existing frontend can display a
status without treating it as a network failure. It does not load models or call
the LLM.

```json
{
  "status": "degraded",
  "rag_loaded": true,
  "model_configured": true,
  "index_ready": false,
  "chunk_count": 1224,
  "provider": "openai_compatible",
  "model": "JunHowie/Qwen3-4B-Instruct-2507-GPTQ-Int4",
  "app_version": "1.1.0"
}
```

- `status` is `ready` only when chunks are loaded and the vector index exists.
- `index_ready` is a file-existence check on `RAG_INDEX_DIR`.
- Additive fields are ignored by the current frontend validator.

## `GET /api/v1/ready`

Deep check for Docker healthchecks and orchestrators. Returns `200` only when all
of the following pass; otherwise `503`.

```json
{
  "status": "ready",
  "checks": {
    "chunks_loaded": true,
    "index_ready": true,
    "provider_reachable": true
  },
  "chunk_count": 1224,
  "provider": "ollama",
  "model": "qwen3.5:4b",
  "app_version": "1.1.0",
  "request_id": "..."
}
```

`provider_reachable` is a short `GET <base_url>/models` with a 2-second timeout.

## Limits and authentication

| Variable | Default | Effect |
|---|---|---|
| `APP_MAX_QUESTION_LENGTH` | 2000 | per-question characters |
| `APP_MAX_HISTORY_MESSAGES` | 8 | history entries |
| `APP_MAX_REQUEST_BYTES` | 65536 | raw body cap, checked before JSON parsing |
| `APP_RATE_LIMIT_PER_MINUTE` | 0 (off) | in-process per-client limit for both chat endpoints |
| `APP_API_KEY` | empty (off) | when set, both chat endpoints require `X-API-Key` |

The rate limiter is in-process and per worker. For multi-worker or public
deployments, enforce a shared limit at the reverse proxy/gateway. The API key is
a seam, not a full authorization system.

## Deliberately not exposed

- Filesystem paths, index contents, or model cache locations.
- LLM endpoint URL, API key, or stack traces.
- Raw provider responses or exception messages.
- Question text in logs. Logs contain request id, method, path, status, duration,
  provider name, model, and error class only.
- Administrative ingestion. Review, chunking, and indexing remain CLI-only.
