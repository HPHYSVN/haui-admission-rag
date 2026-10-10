# Dữ liệu corpus và artefact RAG

`documents/*.jsonl` là source of truth máy đọc được. Review Markdown chỉ là
giao diện sửa trường `content`; trạng thái review nằm riêng trong
`workflow/documents.json` để không vi phạm schema tài liệu.

```text
documents/*.jsonl
  -> review/<category>/<document-id>.md
  -> chunks/proposals/*.jsonl
  -> chunks/reviewed/*.jsonl
  -> chunks/final/*.jsonl
  -> indexes/vector/ (generated, ignored by git)
```

Review độc lập theo document. Không coi đề xuất chunk tự động là final và
không index proposal chưa được duyệt.

## Document review

```bash
uv run python -m rag.cli render --document <document-id>
# Chỉnh phần nội dung giữa hai marker trong file Markdown
uv run python scripts/apply_review.py \
  data/documents/<category>.jsonl data/review/<category>
uv run python -m rag.cli document-review approve --document <document-id>
```

Các trạng thái workflow không thay thế `status` active/archived trong corpus.
Nếu tài liệu trở lại review, final chunks cũ của tài liệu đó không còn được
phép tham gia retrieval.

## Chunk proposal, review và final

```bash
uv run python -m rag.cli chunk --document <approved-document-id>
uv run python -m rag.cli chunk-review approve \
  --input data/chunks/proposals/<category>.jsonl --chunk-id <chunk-id>
uv run python -m rag.cli finalize-chunks \
  --input data/chunks/proposals/<category>.jsonl --document <document-id>
```

Chunk review có thể approve, edit qua `--text-file`, split, merge hoặc reject.
Chỉ approved/edited chunks đi vào `chunks/final/`; text override không sửa
document nguồn. `structured_content`, content type, heading path và source
position giữ khả năng truy nguyên cấu trúc/provenance.

Vector index là dữ liệu sinh, không commit:

```bash
uv run python -m rag.cli index --document <document-id>
uv run python -m rag.cli index --all
```

Xem toàn bộ flow, query, evaluation và vLLM tại
[rag/README.md](../rag/README.md).
