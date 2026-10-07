import json
import sys
from pathlib import Path

from legal_rag.core.rag import LegalRAGPipeline


def run_quality_gate():
    """CI Quality Gate: Evaluates retrieval Hit@3 against ground-truth evaluation data.

    Bypasses LLM generation to ensure fast, deterministic execution in CI.
    """
    print("Starting CI Quality Gate Benchmark (Retrieval Hit@3)...")

    eval_path = Path("tests/eval/eval_20_questions.json")
    if not eval_path.exists():
        print(f"ERROR: Evaluation dataset not found at '{eval_path}'. Quality gate cannot run.")
        sys.exit(1)  # Fail CI explicitly if test data is missing

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    if not eval_data:
        print("ERROR: Evaluation dataset is empty.")
        sys.exit(1)  # Fail CI explicitly if test dataset is empty

    pipeline = LegalRAGPipeline()

    hits = 0
    total = len(eval_data)

    for item in eval_data:
        query = item["question"]
        expected_article = item.get("expected_article") or item.get("article_number")

        # Evaluate retrieval directly (uses bge-m3 embedding model only)
        retrieved_docs = pipeline.retrieve(query, top_k=3)
        retrieved_articles = [doc.get("article_number") for doc in retrieved_docs]

        if expected_article in retrieved_articles:
            hits += 1

    hit_rate = hits / total if total > 0 else 0.0
    print(f"Retrieval Hit@3 Score: {hit_rate:.2f} ({hits}/{total})")

    threshold = 0.75
    if hit_rate < threshold:
        print(
            f"QUALITY GATE FAILED: Hit@3 score ({hit_rate:.2f}) is below threshold ({threshold})."
        )
        sys.exit(1)

    print(f"QUALITY GATE PASSED: Hit@3 score ({hit_rate:.2f}) meets threshold ({threshold}).")
    sys.exit(0)


if __name__ == "__main__":
    run_quality_gate()