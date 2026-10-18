# Retrieval and generation evaluation

Add one JSON object per line to `questions.jsonl`. The checked-in suite contains
24 questions, split evenly across corpus groups `01`, `02`, and `03`. Each case may specify
expected document/chunk IDs, an expected year, and reviewer notes. Retrieval
results and generation checks are separate so a correct source lookup is not
confused with a fluent answer.

Team: để thêm câu hỏi, chỉ cần chèn JSON line vào `evaluation/questions.jsonl`
và chạy lại evaluation. Cấu trúc mỗi dòng:

```json
{"id":"04-01","dataset":"01","question":"Năm 2026 HaUI tuyển sinh đại học chính quy bằng những phương thức nào?","expected_document_ids":["01_admission_004"],"expected_year":2026}
```

## Evaluation chuẩn RAGAS

Chạy full pipeline end-to-end (retrieval + generation + judge LLM):

```bash
# Cấu hình mặc định (sau khi thay model retrieval)
RAG_EMBEDDING_MODEL=intfloat/multilingual-e5-large \
RAG_RERANKER_MODEL=jinaai/jina-reranker-v2-base-multilingual \
uv run python -m evaluation.run_ragas \
  --reranker on \
  --references evaluation/ragas/references.json \
  --output evaluation/ragas/reranked-new.json

# Cấu hình baseline (no reranker)
uv run python -m evaluation.run_ragas \
  --reranker off \
  --references evaluation/ragas/references.json \
  --output evaluation/ragas/baseline-new.json
```

Sử dụng cùng `--references` file cho cả hai lượt để đảm bảo so sánh công bằng.

## Bảng so sánh before/after (quy trình tự động)

Script `evaluation/run_ragas_benchmark.py` hỗ trợ chạy before và after cùng lúc và
so sánh metric:

```bash
# Chạy cả before và after, lưu vào benchmark.json
uv run python -m evaluation.run_ragas_benchmark --output evaluation/ragas/benchmark-new.json

# Chạy riêng từng cấu hình
uv run python -m evaluation.run_ragas_benchmark --config before --output evaluation/ragas/before.json
uv run python -m evaluation.run_ragas_benchmark --config after  --output evaluation/ragas/after.json
```

Kết quả xuất ra `benchmark.json` chứa:
- `before`: kết quả cấu hình cũ (embedding MiniLM)
- `after`: kết quả cấu hình mới (embedding e5-large)
- `summary`: thông tin model cấu hình

## Artifact và tái chạy

```bash
# Làm mới tất cả kết quả evaluation
rm -f evaluation/ragas/*.json
uv run python -m evaluation.run_ragas_benchmark --output evaluation/ragas/benchmark-new.json
```

## Metrics và tầm quan trọng

| Metric | Ý nghĩa |
|---|---|
| Faithfulness | câu trả lời có được hỗ trợ bởi ngữ cảnh hay không |
| Answer relevancy | câu trả lời có trực tiếp trả lời câu hỏi |
| Context precision | context liên quan có xếp cao đúng thứ tự |
| Context recall | ngữ cảnh có bao phủ thông tin cần trả lời |

Bảo toàn hai lớp đánh giá:
1. **Retrieval hit/Recall@K** – phát hiện mất nguồn (từ `rag.evaluate`)
2. **RAGAS** + đánh giá thủ công – phát hiện ngữ cảnh nhiễu, câu trả lời thiếu căn cứ

## Kiểm thử

```bash
uv run python -m unittest discover -s tests -v
```
