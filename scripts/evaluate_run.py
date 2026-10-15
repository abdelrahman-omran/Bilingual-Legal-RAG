"""RAGAS evaluation script running against a live BentoML service endpoint."""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import requests

# --- FIX FOR UPSTREAM RAGAS VERTEXAI IMPORT BUG ---
import sys
import types

try:
    from langchain_community.chat_models.vertexai import ChatVertexAI  # noqa: F401
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
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

EVAL_PATH = Path("tests/eval/eval_4_questions.json")
BENTO_URL = os.getenv("BENTO_URL", "http://localhost:3000/ask")

def main() -> None:
    eval_data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))["questions"]

    rows = []
    print(f"Querying BentoML service at {BENTO_URL}...")
    
    for item in eval_data:
        response = requests.post(
            BENTO_URL, 
            json={"question": item["question"]},
            timeout=60
        )
        response.raise_for_status()
        data = response.json()

        rows.append({
            "question": item["question"],
            "answer": data["answer"],
            "contexts": [c["text_en"] for c in data.get("retrieved_contexts", [])],
            "ground_truth": item["ground_truth"],
        })

    print("Finished gathering responses from BentoML. Running RAGAS judge metrics...")

    judge_llm = LangchainLLMWrapper(ChatOpenAI(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:8080/v1"),
        api_key="not-needed",
        model=os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
        temperature=0.0,
    ))
    judge_emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="BAAI/bge-m3"))

    # Evaluate dataset using RAGAS
    scores = evaluate(
        Dataset.from_list(rows),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_emb,
        run_config=RunConfig(max_workers=1, timeout=300),
    )

    print(scores)
    df = scores.to_pandas()

    # Log metrics to MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")  
    mlflow.set_experiment("ragas_evaluation_bentoml")
    with mlflow.start_run(run_name=f"eval-bento-{datetime.now(timezone.utc):%Y%m%d-%H%M}"):
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

    print("Evaluation complete and results logged to MLflow!")

if __name__ == "__main__":
    main()