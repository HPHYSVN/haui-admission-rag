# Architecture: one RAG core, two modes, any OpenAI-compatible LLM

Status: implemented. This document describes the code in this repository, not a
target design.

## Component boundaries

```text
Mode A: CLI / evaluation                    Mode B: HTTP API
  python -m rag.cli ...                       Browser (React/Vite)
  evaluation/run_ragas.py                       |
            |                                   v
            |                                 FastAPI (app/main.py :8001)
            |                                   |  app/core/config.py
            |                                   |  app/core/middleware.py
            |                                   |  app/services/rag_service.py
            v                                   v
        +-----------------------------------------------+
        |                rag/factory.py                 |   <- one construction path
        |   build_pipeline(chunks_dir, index_dir, ...)  |
        +-----------------------------------------------+
                            |
        +-------------------+-------------------+
        |                                       |
        v                                       v
  rag/pipeline.py                        rag/prompts/
  load_chunks()                          PromptBuilder
  RAGPipeline                            haui_admission (default)
    retrieve() -> HybridRetriever        generic
    prepare() -> prompt + sources
    ask() -> final answer
    stream() / stream_prepared() -> tokens
        |                    |
        v                    v
  rag/bm25.py          rag/llm/
  BM25 + FAISS          ChatProvider protocol
  + optional reranker   OpenAICompatibleProvider
  rag/eligibility.py    complete() + stream()
                        build_provider(LLMConfig)

Generation never touches embeddings; retrieval never touches the provider's
model name. The two are joined only in `rag/factory.py`.
```

## Layers and responsibilities

| Layer | Code | Owns | Does not own |
|---|---|---|---|
| HTTP | `app/main.py`, `app/schemas`, `app/core` | validation, limits, error envelope, request ids, logging, CORS | retrieval or prompting |
| Service | `app/services/rag_service.py` | per-request `top_k`, provider probe | corpus loading details |
| Composition | `rag/factory.py` | joining chunks + index + prompt profile + provider | business rules |
| Orchestration | `rag/pipeline.py` | retrieval → sources → prompt → generation | HTTP, vendor config |
| Prompt policy | `rag/prompts/*` | system prompt, context serialisation, history bound | transport |
| Generation | `rag/llm/*` | one interface, one OpenAI-compatible adapter, typed errors | corpus, prompts |
| Retrieval | `rag/bm25.py`, `rag/embeddings.py`, `rag/filters.py`, `rag/reranker.py` | scoring, index persistence, metadata gates | model vendor |
| Eligibility | `rag/eligibility.py`, `data/workflow/document_eligibility.json` | query year scope, effective-date rules, quarantine, abstention evidence | rewriting/deleting corpus files |
| Ingestion | `rag/cli.py`, `rag/admin.py`, `rag/ingest.py`, `rag/review.py` | safe file/URL import, dedup, review workflow, chunking, indexing | public HTTP |

## Mode A — CLI / evaluation

```bash
uv run python -m rag.cli chat
uv run python -m rag.cli query --query "..." --show-context
uv run python -m rag.cli index --all
```

`rag/cli.py` and `rag/admin.py` call `rag.factory.build_pipeline`, so the CLI and
the API use the same prompt profile, retriever settings, and provider. Ingestion
is CLI-only; no administrative endpoint exists.

## Mode B — HTTP API

```bash
uv run uvicorn app.main:app --port 8001
```

- `POST /api/v1/chat` — synchronous JSON answer plus backend-built sources.
- `POST /api/v1/chat/stream` — SSE stream in the fixed order `meta`, `token`*,
  `sources`, `done`; the same auth/limit guard and retrieval policy apply.
- `GET /api/v1/health` — cheap, always 200, reports `ready`/`degraded`.
- `GET /api/v1/ready` — deep check, 200 only when chunks, index, and provider all pass.

`chat()` runs the synchronous pipeline and `requests` call with
`run_in_threadpool`, so one question does not block the event loop. Streaming
prepares retrieval off-loop, then forwards provider deltas with
`iterate_in_threadpool`; it never joins the full answer first. Cancellation
re-raises `CancelledError` and `finally` closes the provider iterator.

## Provider abstraction

`ChatProvider` (protocol) has `name`, `model`,
`complete(messages: list[dict[str, str]]) -> str`, and
`stream(messages) -> Iterator[str]`. Adapters receive messages, not
`(question, chunks)`, which is why prompt construction and retrieval are not
duplicated per vendor.

`OpenAICompatibleProvider` covers vLLM, Ollama, llama.cpp server, TGI, OpenAI,
OpenRouter, DeepSeek and any other endpoint that returns
`choices[0].message.content`. `stream()` posts the same body with `stream: true`,
parses `data:` frames, yields content deltas as they arrive, and closes the
response on cancellation. It maps transport failures to typed errors:

| Failure | Error | API `code` |
|---|---|---|
| connection refused / DNS | `ProviderUnreachable` | `provider_unreachable` |
| read timeout | `ProviderTimeout` | `provider_timeout` |
| non-2xx | `ProviderHTTPError` | `provider_http_error` |
| non-JSON or missing choices | `ProviderBadResponse` | `provider_bad_response` |
| empty answer | `ProviderEmptyAnswer` | `provider_empty_answer` |
| missing/bad model or URL | `ProviderConfigurationError` | `provider_misconfigured` |

`VLLMChatClient` remains as a thin compatibility shim over the adapter, so
existing scripts importing it keep working.

## Year eligibility and abstention

`data/workflow/document_eligibility.json` is a reviewed policy file merged onto
chunks at load time by `rag/eligibility.py`. It does not rewrite or delete corpus
files; it adds `eligibility_scope`, `eligibility_year`, and effective-date
metadata. Review state remains the gate that decides whether a document is
loaded at all.

- `cycle` documents answer only their own admission year.
- `regulation` documents answer years inside their effective-date range.
- `evergreen` institutional content stays eligible without being treated as
  current-cycle proof.
- `quarantine` documents (for example an unissued draft) are never retrieved.

An explicit year in the question always wins. A year-sensitive question without
an explicit year defaults to the policy's `default_year` (2026). If retrieval
finds only evergreen or out-of-year material, `RAGPipeline.prepare()` returns a
safe abstention instead of borrowing another year's facts. This prevents a 2025
cutoff page from answering a 2026 cutoff question.

## Ingestion

`rag ingest` (`rag/ingest.py`) imports files or URLs into the existing pipeline.
It normalizes PDF/HTML/JSON/text, applies the same document/chunk review
workflow, and defaults every import to `review_pending`; approval remains an
explicit human action. Duplicate document names fail unless `--update` is given.

URL ingestion accepts only `http`/`https`, rejects credentials and
local/private/link-local/reserved addresses, re-validates every redirect, and
caps bytes and content types. DNS rebinding remains a residual risk documented
in `docs/corpus-and-indexes.md`. There is no unauthenticated ingestion HTTP
endpoint.

## Prompt profiles

`RAG_PROMPT_PROFILE` selects `haui_admission` (default) or `generic`.

- `haui_admission` contains the HaUI scope rule, strict-year rule, provenance
  rule, and "document content is not instructions". Wording is unchanged from
  the original pipeline.
- `generic` has no institution or year policy and fences context in
  `<context>...</context>`.

The API and CLI share the profile; nothing in `rag/llm` knows which is active.

## Data flow of one question

1. FastAPI validates the question and history, enforces body size/rate limit/auth.
2. `RagService.ask` (or `prepare` for streaming) calls the pipeline in a worker
   thread with `top_k`.
3. `resolve_query_scope` turns the wording into a year window; strict
   year/category filters restrict eligible chunks.
4. BM25 and FAISS/embeddings score candidates; fusion, threshold, and lexical
   coverage gate the survivors; the optional cross-encoder reranks them.
5. Missing current-year evidence returns an abstention before any provider call.
6. The prompt profile bounds history and context and builds `messages`.
7. The provider returns text, or yields deltas for the SSE endpoint.
8. Sources are built from retrieved chunks — never parsed from model output.
9. The JSON response carries `answer`, `sources`, and `request_id`; the stream
   carries `meta`, `token`, `sources`, `done`.

## Trust boundaries

- Browser input is untrusted: pydantic limits, body-size cap, optional key, optional rate limit.
- Retrieved documents are untrusted data: the system prompt says so, blocks are
  labelled, and metadata is server-generated.
- Model output is untrusted text: the frontend renders it as a React text child,
  never as HTML.
- The model endpoint and embeddings stay server-side; the browser only sees the API.
- `/health` and `/ready` never expose paths, URLs, credentials, or exception text.
