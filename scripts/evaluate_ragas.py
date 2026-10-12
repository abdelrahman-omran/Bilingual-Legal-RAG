"""RAGAS evaluation over the eval set. Logs all 4 metrics to MLflow and
attaches per-question faithfulness to each Langfuse trace."""

import json
import os
import math
from datetime import datetime, timezone
from pathlib import Path

# --- FIX FOR UPSTREAM RAGAS VERTEXAI IMPORT BUG ---
import sys
import types
try:
    from langchain_community.chat_models.vertexai import ChatVertexAI
except ImportError:
    dummy_mod = types.ModuleType("langchain_community.chat_models.vertexai")
    dummy_mod.ChatVertexAI = type("ChatVertexAI", (object,), {})
    sys.modules["langchain_community.chat_models.vertexai"] = dummy_mod
# --------------------------------------------------

import mlflow
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from legal_rag.core.llm_backends import build_backend_from_env
from legal_rag.core.rag import LegalRAGPipeline

EVAL_PATH = Path("tests/eval/eval_20_questions.json")


def main() -> None:
    eval_data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))["questions"]
    pipeline = LegalRAGPipeline(llm_backend=build_backend_from_env())

    rows = []
    for item in eval_data:
        result = pipeline.answer(item["question"], top_k=2)
        rows.append({
            "question": item["question"],
            "answer": result["answer"],
            "contexts": [c["text_en"] for c in result["retrieved_contexts"]],
            "ground_truth": item["ground_truth"],
            "trace_id": result.get("trace_id"),
        })

    judge_llm = LangchainLLMWrapper(ChatOpenAI(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:8080/v1"),
        api_key="not-needed",
        model=os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
        temperature=0.0,
    ))
    judge_emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="BAAI/bge-m3"))

    # Throttled with max_workers=1 to prevent overloading local LLM server
    scores = evaluate(
        Dataset.from_list([{k: v for k, v in r.items() if k != "trace_id"} for r in rows]),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_emb,
        run_config=RunConfig(max_workers=1, timeout=120),
    )

    print(scores)

    # Convert EvaluationResult to pandas DataFrame for robust metrics access
    df = scores.to_pandas()

    # 1) MLflow logging
    mlflow.set_experiment("ragas_evaluation")
    with mlflow.start_run(run_name=f"eval-{datetime.now(timezone.utc):%Y%m%d-%H%M}"):
        mlflow.log_param("llm_model", os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"))
        mlflow.log_param("embedding_model", "BAAI/bge-m3")
        mlflow.log_param("n_questions", len(rows))
        mlflow.log_param("eval_set", EVAL_PATH.name)
        
        for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if m in df.columns:
                valid_vals = [
                    float(v) for v in df[m].dropna().tolist()
                    if not math.isnan(float(v))
                ]
                if valid_vals:
                    mlflow.log_metric(f"mean_{m}", sum(valid_vals) / len(valid_vals))

    # 2) Attach faithfulness to each Langfuse trace
    from langfuse import get_client
    langfuse_client = get_client()

    faith_scores = df["faithfulness"].tolist() if "faithfulness" in df.columns else [None] * len(rows)
    for row, f in zip(rows, faith_scores):
        if row["trace_id"] and f is not None:
            try:
                val = float(f)
                if not math.isnan(val):
                    langfuse_client.create_score(
                        trace_id=row["trace_id"],
                        name="ragas_faithfulness",
                        value=val,
                    )
            except (ValueError, TypeError):
                pass
                
    langfuse_client.flush()


if __name__ == "__main__":
    main()