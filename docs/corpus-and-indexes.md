# Corpus and indexes

## Committed sources vs generated artifacts

| Path | Committed? | Meaning |
|---|---|---|
| `data/documents/*.jsonl` | yes | normalized source documents, source of truth |
| `data/workflow/documents.json` | yes | per-document review state |
| `data/workflow/document_eligibility.json` | yes | reviewed year/effective-date/quarantine policy merged at load time |
| `data/chunks/proposals/*.jsonl` | no | generated chunk proposals |
| `data/chunks/reviewed/*.jsonl` | no | reviewer decisions and text overrides |
| `data/chunks/final/*.jsonl` | yes | approved chunks used for retrieval |
| `data/indexes/vector/` | no (gitignored) | generated embeddings, FAISS file, metadata |
| `data/raw/`, `data/review/` | no | crawler output and review Markdown |

`load_chunks()` reads only `data/chunks/final/*.jsonl` records whose chunk
`status` is `final` **and** whose document workflow state is `final` or
`indexed`. Changing a document back to review makes its old final chunks
ineligible until new chunks are finalized.

## What is in the index directory

`RAG_INDEX_DIR` (default `data/indexes/vector`):

- `vectors.jsonl` — one record per chunk: chunk id, content hash, vector.
- `vectors.faiss` — `IndexFlatIP` over L2-normalized float32 vectors.
- `metadata.json` — embedding model name and chunk count.

`VectorIndex.ensure_current()` re-embeds only documents whose `content_hash`
changed and refuses to run when `metadata.json` names a different embedding
model. That guard is intentionally independent of the generation model: switching
`LLM_MODEL` never invalidates the index.

## Why `data/indexes/` is never baked into the image

The index is derived data and is large. Baking it in would:

- couple the image to one embedding model and one corpus snapshot;
- force a rebuild for every corpus change;
- duplicate data between the image and the runtime volume.

`Dockerfile` creates an empty `/app/data/indexes/vector`; `docker-compose.yml`
mounts the `indexes` volume over it. `docs/docker.md` shows the explicit build
command.

## Year eligibility policy

`data/workflow/document_eligibility.json` keeps historical and evergreen
material without letting it answer the wrong cycle. It is merged onto chunks by
`rag/eligibility.py` at load time; source files and review state are unchanged.
Per-category defaults are overridden per document when needed.

| Scope | Use | Retrieval effect |
|---|---|---|
| `cycle` | admission notices, cutoffs, tuition/scholarships | only the matching admission year |
| `regulation` | ministry/HaUI regulations | years inside `effective_from`/`effective_to` |
| `evergreen` | campus, student life, support, career context | available without becoming current-cycle proof |
| `quarantine` | unissued drafts, unsafe duplicates | never retrieved |

Explicit years in a question win. Year-sensitive questions without an explicit
year default to `default_year` (2026). If no retrieved chunk is positive
evidence for that year, the pipeline abstains instead of citing an older cycle.
Curated overrides also record known metadata anomalies (for example a
content-2026 admission plan imported with a 2023 publication date) so the
policy, not a guessed date, decides eligibility.

## Ingestion

```bash
uv run python -m rag.cli ingest --help
uv run python -m rag.cli ingest file path/to/document.pdf --category 01_admission
uv run python -m rag.cli ingest url https://tuyensinh.haui.edu.vn/... --category 01_admission --dry-run
```

`rag/ingest.py` reuses the existing review workflow. Imports default to
`review_pending`; `approve` and `exclude` are explicit human decisions.
Duplicate document names fail by default unless `duplicate="update"` (`--update`
on the CLI). `--dry-run` reports what would change without writing.

URL imports accept only `http`/`https`, reject embedded credentials, block
local/private/link-local/reserved addresses, re-validate every redirect, and cap
response bytes and content types. DNS rebinding between the validation lookup
and the actual request remains a residual risk; deploy ingestion behind a
controlled network boundary or a resolving proxy when that threat matters.

## Explicit rebuild

```bash
# All documents
uv run python -m rag.cli index --all

# One document only
uv run python -m rag.cli index --document 05_tuition_scholarship_001
```

The first semantic query builds a missing index automatically, but an explicit
build before serving is the recommended warm-up. Inside Docker:

```bash
docker compose run --rm backend python -m rag.cli index --all
```

Also pre-download the reranker, or disable it for a lighter deployment:

```dotenv
RAG_RERANKER_ENABLED=false
```

## Ingestion and index updates (out of band from chat)

Ingestion remains CLI-only and does not run inside a chat request. After a
document is approved and its chunks are final, rebuild the affected part of the
index:

```python
from pathlib import Path
from rag.embeddings import VectorIndex
from rag.pipeline import load_chunks

chunks = load_chunks(Path("data/chunks/final"))
VectorIndex(Path("data/indexes/vector")).build(chunks, document_id="<document-id>")
```

Because `build(document_id=...)` only re-embeds that document, an operator can run
it per document. Chat workers keep reading the previous index file until the new
one is written; the write is atomic per file. A production deployment that runs
ingestion while serving should add a lock or write to a new directory and swap
`RAG_INDEX_DIR`, which is outside this pass.

## Retrieval settings

All retrieval configuration stays in `rag/config.py` and environment variables:
`RAG_ALPHA`, `RAG_SCORE_THRESHOLD`, `RAG_MIN_TERM_COVERAGE`,
`RAG_TOP_K`, `RAG_RERANKER_*`, `RAG_EMBEDDING_MODEL`, `RAG_EMBED_BATCH_SIZE`. None
of these belong to the generation provider.

`RAG_EMBED_BATCH_SIZE` defaults to `32`. fastembed's own default is `256`, which
measured roughly 8 GB peak RSS for `multilingual-e5-large` on this stack (ONNX
activation memory); `32` measured roughly 3 GB. Raise it only if the host has
spare memory and the model files are already cached.

## Known ONNX cache requirement

`intfloat/multilingual-e5-large` is an ONNX model with an external
`model.onnx_data` file. onnxruntime refuses that file when it is reached through
Hugging Face cache symlinks, so the deployment must materialise real files:

```dotenv
HF_HUB_DISABLE_SYMLINKS=1
FASTEMBED_CACHE_PATH=/path/to/persistent/fastembed-cache
```

The Docker image and compose file set both, with the cache inside the
`hf_cache` volume. Without them the model download appears to succeed and then
fails at load with "External data path escapes model directory".
