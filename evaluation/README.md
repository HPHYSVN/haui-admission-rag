# Retrieval and generation evaluation

Add one JSON object per line to `questions.jsonl`. Each case may specify
expected document/chunk IDs, an expected year, and reviewer notes. Retrieval
results and generation checks are separate so a correct source lookup is not
confused with a fluent answer.

Run retrieval evaluation after approving and finalizing the documents needed by
the cases:

```bash
uv run python -m rag.evaluate
```

Use `retrieval/` for reviewed retrieval run outputs and `generation/` for
manually assessed answer outputs. Keep expected answers out of prompts.
