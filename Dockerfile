# Backend image: FastAPI + the framework-free RAG package.
#
# Deliberately does NOT contain model weights or a prebuilt vector index.
#   * model weights are cached in the `hf_cache` volume on first use;
#   * `data/indexes/` is a volume, built with an explicit command.
# See docs/docker.md.

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/cache/huggingface \
    FASTEMBED_CACHE_PATH=/cache/huggingface/fastembed \
    HF_HUB_DISABLE_SYMLINKS=1

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

WORKDIR /app

# Install locked runtime dependencies before the source so this layer is cached.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY app ./app
COPY rag ./rag
COPY schemas ./schemas
COPY data/documents ./data/documents
COPY data/chunks/final ./data/chunks/final
COPY data/workflow ./data/workflow

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/indexes/vector /cache/huggingface \
    && chown -R appuser:appuser /app /cache/huggingface

USER appuser

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8001

# Deep readiness: chunks + on-disk index + provider probe. A container that
# answers HTTP 200 on /health but cannot actually answer questions stays unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8001/api/v1/ready', timeout=3).status == 200 else 1)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
