# Docker deployment

Two images, no LLM container and no broker:

- `backend` — `python:3.12-slim`, FastAPI + `rag/`, non-root, port `8001`.
- `frontend` — multi-stage `node:22-alpine` build served by `nginx:alpine`, port `80`.

The backend image contains no model weights and no vector index. Both live in
named volumes so they survive `docker compose down` and container recreation.

## Prerequisites

- Docker with Compose v2.
- A vLLM or Ollama server reachable from the host (for example
  `http://localhost:8000/v1` or `http://localhost:11434/v1`).
- `.env` created from `.env.example`.

## Build and start

```bash
cp .env.example .env   # if you do not already have one
docker compose build
docker compose up -d
```

With the default `.env`, the backend reaches a host LLM through
`host.docker.internal:8000`. Override it without editing the compose file:

```bash
DOCKER_LLM_BASE_URL=http://host.docker.internal:11434/v1 \
  LLM_PROVIDER=ollama LLM_MODEL=qwen3.5:4b \
  docker compose up -d
```

The frontend is at <http://localhost:8080> and proxies `/api` to the backend.

### Ollama on the host

Ollama listens on loopback (`127.0.0.1:11434`) by default, which `host.docker.internal`
cannot reach from a bridge network. Start it bound to all interfaces before using
it from a container:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
# or, for a systemd user service:
systemctl --user edit ollama
# [Service]
# Environment="OLLAMA_HOST=0.0.0.0:11434"
```

Alternatively run the backend with `network_mode: host` and
`LLM_BASE_URL=http://127.0.0.1:11434/v1`. Do not expose Ollama to an untrusted
network without authentication.

## Build the vector index once

The first `/readiness` check fails until two things exist: the embedding model
cache and the index. Build the index inside the running stack (it downloads the
embedding model into the `hf_cache` volume on first use):

```bash
docker compose run --rm backend python -m rag.cli index --all
```

Check status:

```bash
curl -s http://localhost:8001/api/v1/health | python -m json.tool
curl -s http://localhost:8001/api/v1/ready  | python -m json.tool
```

`/ready` returns HTTP 200 only when chunks, the on-disk index, and the LLM
endpoint all check out.

## Logs, stop, restart

```bash
docker compose logs -f backend
docker compose ps
docker compose restart backend
docker compose down            # keeps indexes and the model cache
docker compose down -v         # deletes both volumes (index + weights)
```

## Verify volume persistence

```bash
docker compose up -d
docker compose restart backend
curl -s http://localhost:8001/api/v1/ready
```

The index and model cache are still present after the restart because they are
named volumes, not container layers.

## Chat through the stack

```bash
curl -s http://localhost:8001/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"Học phí năm 2026 là bao nhiêu?"}'

# SSE through the bundled Nginx proxy (no buffering)
curl -N http://localhost:8080/api/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"question":"Học phí năm 2026 là bao nhiêu?"}'

curl -s http://localhost:8080/api/v1/health
```

The frontend image's `nginx.conf` gives `/api/v1/chat/stream` its own exact
location with `proxy_buffering off`, `proxy_cache off`, and a 300-second read
timeout. If you put another proxy in front, disable response buffering there too.

## Ingestion

Ingestion stays CLI-only. The backend image bakes `data/documents`,
`data/chunks/final`, and `data/workflow` and does not mount them back to the
host, so run ingestion in the repository checkout on the host; container-local
writes would be lost when the container is recreated:

```bash
uv run python -m rag.cli ingest url \
  https://tuyensinh.haui.edu.vn/... --category 01_admission --dry-run
uv run python -m rag.cli ingest file path/to/document.pdf --category 01_admission
uv run python -m rag.cli review
uv run python -m rag.cli index --all
```

Then rebuild the backend image so the approved corpus and workflow state are
included, or add a deliberate bind mount for `data/` before running ingestion
inside the stack.

## Troubleshooting

| Symptom | Check |
|---|---|
| `degraded`, `index_ready: false` | run the index command above; confirm `RAG_INDEX_DIR=/app/data/indexes/vector` |
| `provider_reachable: false` | the LLM host is not reachable from the container; use `host.docker.internal` and `extra_hosts` |
| first query very slow | embedding/reranker weights are downloading into `hf_cache`; the first build is the slow one |
| `docker compose build` cannot resolve `ghcr.io/astral-sh/uv` | the build host has no access to GHCR; mirror the image or preinstall `uv` |
| frontend loads but `/api` 502 | backend is not healthy; check `docker compose logs backend` |
| `permission denied` on the index volume | the image creates `/app/data/indexes` owned by uid `10001`; do not bind-mount a root-owned host directory |

## What is not in the images

- `data/indexes/` — generated, volume only.
- `data/raw/`, `data/review/`, proposals — ingestion artifacts, not needed to serve chat.
- `.env` — passed through `env_file`, never baked into a layer.
- Model weights — downloaded to the `hf_cache` volume.
- Tests, docs, crawler, evaluation — excluded by `.dockerignore`.
