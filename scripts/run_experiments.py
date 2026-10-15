"""Compare five chunking configs with a compact custom faithfulness judge.

Faithfulness here is a custom LLM-as-judge score, not the RAGAS metric.
Run from the repo root: uv run python scripts/run_experiments.py
Chunk sizes and overlaps are measured in characters.
"""
import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

QDRANT_URL = os.getenv("MLFLOW_EXPERIMENT_QDRANT_URL", os.getenv("QDRANT_URL", "")).strip()
os.environ["QDRANT_URL"] = QDRANT_URL

import mlflow
from mlflow.tracking import MlflowClient
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from legal_rag.core.llm_backends import build_backend_from_env
from legal_rag.core.rag import pipeline as rag
from legal_rag.ingestion.indexer import CorpusIndexer

TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
EXPERIMENT = "egyptian-legal-rag-chunking"
REGISTRY_NAME = "EgyptianLegalRAGBestChunkingConfig"
CORPUS = ROOT / "data/processed/corpus.json"
QUESTIONS_FILE = ROOT / "tests/eval/eval_4_questions.json"
STORES = ROOT / ".mlflow_chunking_qdrant"
REPORTS = ROOT / "reports"
TOP_K = 3
JUDGE_MAX_TOKENS = 32
CONFIGS = [
    {"chunk_size": 256, "overlap": 32, "embedding_model": "BAAI/bge-m3"},
    {"chunk_size": 384, "overlap": 48, "embedding_model": "BAAI/bge-m3"},
    {"chunk_size": 512, "overlap": 64, "embedding_model": "BAAI/bge-m3"},
    {"chunk_size": 512, "overlap": 0, "embedding_model": "BAAI/bge-m3"},
    {"chunk_size": 256, "overlap": 64, "embedding_model": "BAAI/bge-m3"},
]


def load_questions():
    data = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    items = data.get("questions", data) if isinstance(data, dict) else data
    questions = [
        str(item if isinstance(item, str) else item.get("question", item.get("query", ""))).strip()
        for item in items
        if isinstance(item, (str, dict))
    ]
    questions = [question for question in questions if question]
    if not questions:
        raise ValueError(f"No questions found in {QUESTIONS_FILE}")
    return questions


def make_judge():
    url = os.getenv("FAITHFULNESS_JUDGE_BASE_URL") or os.getenv("LLM_BASE_URL") or "http://localhost:8080/v1"
    model = os.getenv("FAITHFULNESS_JUDGE_MODEL") or os.getenv("LLM_MODEL_NAME") or "Qwen/Qwen2.5-1.5B-Instruct"
    key = os.getenv("FAITHFULNESS_JUDGE_API_KEY") or os.getenv("LLM_API_KEY") or "not-needed"
    print(f"Compact faithfulness judge: {url} | {model} | max_tokens={JUDGE_MAX_TOKENS}")
    return OpenAI(base_url=url, api_key=key, timeout=120, max_retries=1), model


def faithfulness_score(question, judge, judge_model):
    """One short judge request per question; fail clearly if no valid score is returned."""
    result = rag.answer(question, top_k=TOP_K)
    answer = str(result.get("answer") or "").strip()[:800]
    docs = result.get("retrieved_contexts") or []
    if not answer or not docs:
        raise RuntimeError(f"Missing generated answer or retrieved context for: {question}")

    context_parts = []
    for doc in docs[:TOP_K]:
        citation = doc.get("citation", f"Article {doc.get('article_number', '?')}")
        arabic = str(doc.get("text_ar") or "")[:250]
        english = str(doc.get("text_en") or "")[:250]
        context_parts.append(f"{citation}\nAR: {arabic}\nEN: {english}")
    context = "\n\n".join(context_parts)[:1700]
    prompt = f'''Judge whether the answer's factual/legal claims are supported by the context.
1.0 = fully supported; 0.5 = partly supported; 0.0 = unsupported.
Use only the context. Return ONLY JSON such as {{"score": 0.8}}.

Question: {question[:300]}
Context:
{context}
Answer:
{answer}
'''
    response = judge.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": "You are a strict legal answer grounding evaluator. Output only the requested JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=JUDGE_MAX_TOKENS,
    )
    text = (response.choices[0].message.content or "").strip()
    match = re.search(r'"score"\s*:\s*(0(?:\.\d+)?|1(?:\.0+)?)', text)
    if not match:
        match = re.fullmatch(r"\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*", text)
    if not match:
        raise RuntimeError(f"Judge did not return a numeric score: {text!r}")
    score = float(match.group(1))
    if not 0.0 <= score <= 1.0:
        raise RuntimeError(f"Invalid faithfulness score: {score}")
    return score


class BestConfigModel(mlflow.pyfunc.PythonModel):
    """Small Registry model that exposes the selected configuration only."""
    def __init__(self, config):
        self.config = config

    def predict(self, context, model_input, params=None):
        return self.config


def register_best(config, best_run_id):
    client = MlflowClient(tracking_uri=TRACKING_URI, registry_uri=TRACKING_URI)
    client.set_tag(best_run_id, "selection_status", "best_chunking_config")
    config_path = REPORTS / "best_chunking_config.json"
    client.log_artifact(best_run_id, str(config_path), artifact_path="selection")

    with mlflow.start_run(run_name="register-best-chunking-config") as run:
        mlflow.set_tag("source_best_run_id", best_run_id)
        mlflow.log_params({key: config[key] for key in ("chunk_size", "overlap", "embedding_model")})
        mlflow.log_metric("faithfulness", config["faithfulness"])
        mlflow.log_dict(config, "best_chunking_config.json")
        mlflow.pyfunc.log_model(
            artifact_path="chunking_config",
            python_model=BestConfigModel(config),
            registered_model_name=REGISTRY_NAME,
        )
        registry_run_id = run.info.run_id

    versions = client.search_model_versions(f"name='{REGISTRY_NAME}'")
    version = next((v for v in versions if v.run_id == registry_run_id), None)
    if version is None:
        raise RuntimeError(f"Could not find registered version for {REGISTRY_NAME}")
    try:
        client.set_registered_model_alias(REGISTRY_NAME, "best", version.version)
    except Exception as exc:
        print(f"Model version {version.version} registered; alias was not set: {exc}")
    print(f"Registered {REGISTRY_NAME} version {version.version}")


def main():
    for path in (CORPUS, QUESTIONS_FILE):
        if not path.exists():
            raise FileNotFoundError(path)
    if len(CONFIGS) < 5:
        raise ValueError("The assignment requires at least five configs.")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    questions = load_questions()
    judge, judge_model = make_judge()
    rag.llm_backend = build_backend_from_env()
    rag.qdrant_client.close()  # release default local Qdrant opened on import

    encoder = rag.encoder  # reuse the loaded embedding model across all configs
    results = []
    REPORTS.mkdir(parents=True, exist_ok=True)

    for number, config in enumerate(CONFIGS, 1):
        name = f"chunk_{config['chunk_size']}_overlap_{config['overlap']}"
        store = STORES / name
        if not QDRANT_URL:
            shutil.rmtree(store, ignore_errors=True)
        collection = f"exp_{name}"
        print(f"\n[{number}/{len(CONFIGS)}] {name}")

        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({**config, "chunk_unit": "characters", "top_k": TOP_K,
                               "faithfulness_evaluator": "compact_llm_judge"})
            indexer = CorpusIndexer(
                collection_name=collection,
                model_name=config["embedding_model"],
                db_path=str(store),
                chunk_size=config["chunk_size"],
                overlap=config["overlap"],
                encoder=encoder,
            )
            try:
                indexed_count = indexer.index_corpus(CORPUS)
            finally:
                indexer.client.close()

            rag.encoder, rag.collection_name = encoder, collection
            rag.qdrant_client = QdrantClient(url=QDRANT_URL) if QDRANT_URL else QdrantClient(path=str(store))
            try:
                scores = []
                for q_num, question in enumerate(questions, 1):
                    print(f"  question {q_num}/{len(questions)}")
                    scores.append(faithfulness_score(question, judge, judge_model))
                score = mean(scores)
                mlflow.log_metric("faithfulness", score)
                mlflow.log_metric("chunks_indexed", indexed_count)
                results.append({**config, "faithfulness": score, "run_id": run.info.run_id,
                                "run_name": name, "chunks_indexed": indexed_count})
                print(f"  mean faithfulness={score:.3f}; chunks={indexed_count}")
            finally:
                rag.qdrant_client.close()
                rag.qdrant_client = None

    best = max(results, key=lambda row: row["faithfulness"])
    best_config = {key: best[key] for key in ("chunk_size", "overlap", "embedding_model", "faithfulness", "run_id")}
    best_path = REPORTS / "best_chunking_config.json"
    best_path.write_text(json.dumps(best_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    results_path = REPORTS / "mlflow_chunking_runs.csv"
    with results_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    register_best(best_config, best["run_id"])
    print(f"\nBest config: {best_config}")
    print(f"Comparison CSV: {results_path}")
    print("Take a screenshot of the MLflow comparison UI and save it as reports/mlflow_chunking_experiment.png")


if __name__ == "__main__":
    main()
