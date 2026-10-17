"""Run end-to-end RAGAS evaluation for the HaUI RAG pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

from rag.bm25 import BM25Retriever
from rag.config import DEFAULT_TOP_K, DEFAULT_VECTOR_INDEX_DIR
from rag.pipeline import RAGPipeline, VLLMChatClient, load_chunks
from rag.embeddings import LocalEmbedder

ROOT_DIR = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT_DIR / "evaluation" / "questions.jsonl"
CHUNKS_DIR = ROOT_DIR / "data" / "chunks" / "final"


class LocalFastEmbedEmbeddings(Embeddings):
    """LangChain adapter using the same local embedding model as retrieval."""

    def __init__(self) -> None:
        self.embedder = LocalEmbedder()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.embed([text])[0]


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
        if not isinstance(case, dict) or not isinstance(case.get("question"), str):
            raise ValueError(f"{path}:{line_no}: invalid evaluation case")
        cases.append(case)
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    return cases


def reference_contexts(
    question: str,
    expected_ids: list[str],
    chunks: list[dict[str, Any]],
    top_k: int,
) -> list[str]:
    expected_chunks = [chunk for chunk in chunks if chunk.get("document_id") in expected_ids]
    if not expected_chunks:
        raise ValueError(f"No reference chunks found for {expected_ids!r}")
    ranked = BM25Retriever(expected_chunks).search(question, top_k=max(top_k, 4))
    return [chunk["text"] for _, chunk in ranked[:top_k]]


def build_judge() -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    load_dotenv(ROOT_DIR / ".env", override=False)
    llm = ChatOpenAI(
        model=os.getenv("VLLM_MODEL", "JunHowie/Qwen3-4B-Instruct-2507-GPTQ-Int4"),
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("VLLM_API_KEY") or "local",
        temperature=0.0,
        max_tokens=1600,
        timeout=180,
        max_retries=1,
    )
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(LocalFastEmbedEmbeddings())


def generate_reference(
    question: str,
    contexts: list[str],
    client: VLLMChatClient,
) -> str:
    chunks = [
        {
            "chunk_id": f"reference-{index}",
            "document_id": "reference",
            "title": "Gold reference context",
            "source_url": "",
            "text": text,
        }
        for index, text in enumerate(contexts)
    ]
    return client.answer(question, [(1.0, chunk) for chunk in chunks])


def run(
    *,
    use_reranker: bool,
    reranker_model: str | None,
    questions_path: Path,
    output_path: Path,
    references_path: Path,
    top_k: int,
) -> None:
    cases = load_cases(questions_path)
    chunks = load_chunks(CHUNKS_DIR)
    pipeline = RAGPipeline(
        chunks,
        VLLMChatClient(),
        index_dir=DEFAULT_VECTOR_INDEX_DIR,
        use_reranker=use_reranker,
    )
    if use_reranker and reranker_model:
        from rag.reranker import CrossEncoderReranker

        pipeline.retriever.reranker = CrossEncoderReranker(reranker_model)
    reference_client = VLLMChatClient()
    if references_path.exists():
        references = json.loads(references_path.read_text(encoding="utf-8"))
        if not isinstance(references, dict):
            raise ValueError(f"Reference file must contain an object: {references_path}")
    else:
        references = {}
    samples: list[SingleTurnSample] = []
    records: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        question = case["question"]
        expected_ids = case.get("expected_document_ids", [])
        gold_contexts = reference_contexts(question, expected_ids, chunks, top_k)
        case_id = case.get("id")
        if not isinstance(case_id, str):
            raise ValueError(f"Evaluation case has no string id: {case!r}")
        reference = references.get(case_id)
        if not isinstance(reference, str) or not reference.strip():
            reference = generate_reference(question, gold_contexts, reference_client)
            references[case_id] = reference
            references_path.parent.mkdir(parents=True, exist_ok=True)
            references_path.write_text(
                json.dumps(references, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        response = pipeline.ask(question, top_k=top_k)
        retrieved_contexts = [chunk["text"] for _, chunk in response.retrieved]
        samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=retrieved_contexts,
                reference_contexts=gold_contexts,
                response=response.answer,
                reference=reference,
            )
        )
        records.append(
            {
                "id": case_id,
                "dataset": case.get("dataset"),
                "question": question,
                "reference": reference,
                "answer": response.answer,
                "retrieved_document_ids": [chunk["document_id"] for _, chunk in response.retrieved],
                "retrieved_chunk_ids": [chunk["chunk_id"] for _, chunk in response.retrieved],
                "reference_context_count": len(gold_contexts),
            }
        )
        print(f"[{index}/{len(cases)}] {question}")

    llm, embeddings = build_judge()
    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        show_progress=True,
    )
    scores = result.to_pandas().to_dict(orient="records")
    for record, score in zip(records, scores, strict=True):
        record["scores"] = {
            name: (None if value != value else float(value))
            for name, value in score.items()
            if name in {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "config": {
                    "reranker": use_reranker,
                    "reranker_model": reranker_model,
                    "top_k": top_k,
                    "judge_model": os.getenv("VLLM_MODEL"),
                },
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reranker", choices=("off", "on"), required=True)
    parser.add_argument("--reranker-model")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--references",
        type=Path,
        default=ROOT_DIR / "evaluation" / "ragas" / "references.json",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    run(
        use_reranker=args.reranker == "on",
        reranker_model=args.reranker_model,
        questions_path=args.questions,
        output_path=args.output,
        references_path=args.references,
        top_k=args.top_k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
