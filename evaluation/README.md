# Retrieval and generation evaluation

Add one JSON object per line to `questions.jsonl`. The checked-in suite contains
24 questions, split evenly across corpus groups `01`, `02`, and `03`. Each case may specify
expected document/chunk IDs, an expected year, and reviewer notes. Retrieval
results and generation checks are separate so a correct source lookup is not
confused with a fluent answer.

Run the baseline and reranked retrieval evaluations with the same suite:

```bash
uv run python -m rag.evaluate --reranker off --output evaluation/retrieval/baseline.json
uv run python -m rag.evaluate --reranker on --output evaluation/retrieval/reranked.json
```

`--reranker off` is the baseline. `--reranker on` scores the hybrid candidates
with the configured fastembed cross-encoder and then returns the top-k chunks.
Use `retrieval/` for reviewed retrieval run outputs and `generation/` for
manually assessed answer outputs. Keep expected answers out of prompts.

## RAGAS end-to-end evaluation

RAGAS evaluates the complete retrieval-to-generation pipeline with
faithfulness, answer relevancy, context precision, and context recall. It
requires the local vLLM endpoint configured in `.env`:

```bash
uv run python -m evaluation.run_ragas \
  --reranker off \
  --output evaluation/ragas/baseline-v2.json

uv run python -m evaluation.run_ragas \
  --reranker on \
  --reranker-model Xenova/ms-marco-MiniLM-L-6-v2 \
  --references evaluation/ragas/references.json \
  --output evaluation/ragas/reranked-v2.json
```

Use the same `--references` file for both runs. The detailed comparison is in
`evaluation/ragas/report.md`.
