# LLM providers

Generation is vendor-neutral. Every supported backend must expose the OpenAI
`POST /v1/chat/completions` shape and return
`choices[0].message.content`. One adapter (`rag/llm/openai_compatible.py`) serves
all of them; `LLM_PROVIDER` only selects a default base URL and a name for
observability.

## Variables

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `openai_compatible` | `openai_compatible`, `vllm`, `ollama`, `openai` |
| `LLM_BASE_URL` | provider default | OpenAI-compatible root, e.g. `http://localhost:8000/v1` |
| `LLM_MODEL` | `JunHowie/Qwen3-4B-Instruct-2507-GPTQ-Int4` | model identifier sent in the body |
| `LLM_API_KEY` | empty | bearer token; empty means no `Authorization` header |
| `LLM_TIMEOUT` | `120` | per-request timeout in seconds (> 0) |
| `LLM_TEMPERATURE` | `0.1` | sampling temperature, 0–2 |
| `LLM_MAX_TOKENS` | `800` | output token budget |
| `LLM_TOP_P` | unset | optional nucleus sampling; omitted when unset |
| `LLM_EXTRA_BODY` | `{}` | JSON object of provider-specific body fields |

Provider default base URLs:

| `LLM_PROVIDER` | Default `LLM_BASE_URL` |
|---|---|
| `openai_compatible` | `http://localhost:8000/v1` |
| `vllm` | `http://localhost:8000/v1` |
| `ollama` | `http://localhost:11434/v1` |
| `openai` | `https://api.openai.com/v1` |

`LLM_EXTRA_BODY` is merged into the request body, but `model`, `messages`, and
`stream` are reserved and can never be overridden there. Example:

```dotenv
LLM_EXTRA_BODY={"repetition_penalty": 1.05, "seed": 7}
```

## Streaming

The same adapter implements both `complete(messages)` and
`stream(messages) -> Iterator[str]`. Streaming posts the same
`/chat/completions` body with `stream: true`, reads `data:` frames, yields each
`choices[0].delta.content` fragment as it arrives, and stops at `[DONE]`.
Fragments are never joined in the provider layer. A malformed frame, an upstream
non-2xx response, or an empty stream raises the same typed errors as the
non-streaming path. Closing the iterator closes the underlying HTTP response.

The API exposes this through `POST /api/v1/chat/stream`; the JSON endpoint keeps
using `complete()`. Providers that do not implement streaming can still serve the
JSON endpoint. The SSE contract is documented in `docs/api.md`.

## Legacy `VLLM_*` aliases

Existing deployments keep working. The `LLM_*` name wins when both are set; an
empty value counts as unset.

| Primary | Legacy alias |
|---|---|
| `LLM_BASE_URL` | `VLLM_BASE_URL` |
| `LLM_MODEL` | `VLLM_MODEL` |
| `LLM_API_KEY` | `VLLM_API_KEY` |
| `LLM_TIMEOUT` | `VLLM_TIMEOUT` |

`LLM_PROVIDER`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_TOP_P`, and
`LLM_EXTRA_BODY` have no legacy equivalent. The old hard-coded values were
temperature `0.1` and max tokens `800`, which are now the defaults.

## Local vLLM

```dotenv
LLM_PROVIDER=vllm
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=JunHowie/Qwen3-4B-Instruct-2507-GPTQ-Int4
LLM_API_KEY=
```

```bash
curl http://localhost:8000/v1/models
uv run python -m rag.cli chat
```

## Local Ollama

Ollama exposes an OpenAI-compatible endpoint at `/v1`:

```dotenv
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen3.5:4b
LLM_API_KEY=
```

```bash
ollama list
curl http://localhost:11434/v1/models
uv run python -m rag.cli query --query "Học phí năm 2026?"
```

Inside Docker, the host endpoint is `http://host.docker.internal:11434/v1`, but
Ollama must not be bound to loopback only: start it with
`OLLAMA_HOST=0.0.0.0:11434`, or run the backend container with
`network_mode: host`. Otherwise `/api/v1/ready` reports
`provider_reachable: false` while a `GET /models` from the host still works.

Reasoning-style models can spend the whole `LLM_MAX_TOKENS` budget on their
private reasoning and return an empty `content`. The adapter reports that as
`provider_empty_answer`, and the log/error message suggests raising
`LLM_MAX_TOKENS` or disabling thinking. Use a non-thinking model, or a model
whose chat template supports disabling reasoning, when a low token budget is
required. `LLM_EXTRA_BODY` can forward provider-specific switches, but support
varies: Ollama's OpenAI-compatible endpoint ignores the `think` field.

## External API

```dotenv
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-...
LLM_TIMEOUT=60
LLM_MAX_TOKENS=500
```

Any other OpenAI-compatible host works the same way (OpenRouter, DeepSeek,
Together, a TGI or llama.cpp server). Set `LLM_PROVIDER=openai_compatible` and
provide `LLM_BASE_URL` explicitly.

Never commit a real key. `.env` is gitignored; `.env.example` keeps the key empty.

## Switching models safely

- Chat model: change `LLM_MODEL` (and `LLM_BASE_URL` if the server moves). No
  index rebuild is needed.
- Embedding/reranker model: change `RAG_EMBEDDING_MODEL` or
  `RAG_RERANKER_MODEL`. Changing the embedding model **does** require rebuilding
  the vector index; `rag/embeddings.py` refuses to reuse an index built with a
  different embedding model.
- The two are kept apart on purpose: `rag/llm/config.py` contains only generation
  settings, and `rag/config.py` contains corpus/retrieval settings.

## Failure codes

The API maps provider failures to stable codes (`provider_unreachable`,
`provider_timeout`, `provider_http_error`, `provider_bad_response`,
`provider_empty_answer`, `provider_misconfigured`). The message never contains
the endpoint URL, the API key, or a stack trace.
