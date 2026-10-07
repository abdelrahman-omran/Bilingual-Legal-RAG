import json
import sys
from pathlib import Path

from legal_rag.core.rag import LegalRAGPipeline


def run_quality_gate():
    """CI Quality Gate: Evaluates retrieval Hit@3 against ground-truth evaluation data."""
    print("Starting CI Quality Gate Benchmark (Retrieval Hit@3)...")

    eval_path = Path("tests/eval/eval_20_questions.json")
    if not eval_path.exists():
        print(f"ERROR: Evaluation dataset not found at '{eval_path}'. Quality gate cannot run.")
        sys.exit(1)

    with open(eval_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Extract questions list
    if isinstance(raw_data, dict) and "questions" in raw_data:
        eval_data = raw_data["questions"]
    elif isinstance(raw_data, list):
        eval_data = raw_data
    else:
        print("ERROR: Unexpected evaluation dataset format.")
        sys.exit(1)

    if not eval_data:
        print("ERROR: Evaluation dataset is empty.")
        sys.exit(1)

    pipeline = LegalRAGPipeline()

    hits = 0
    total = len(eval_data)

    for item in eval_data:
        query = item.get("question")
        if not query:
            continue

        # Extract expected articles list safely
        expected = (
            item.get("expected_articles")
            or item.get("expected_article")
            or item.get("article_number")
        )

        if isinstance(expected, int):
            expected_list = [expected]
        elif isinstance(expected, list):
            expected_list = expected
        elif isinstance(expected, str):
            try:
                expected_list = [int(expected)]
            except ValueError:
                expected_list = []
        else:
            expected_list = []

        # Convert expected articles to integers and strings for type-safe comparison
        expected_set = {int(a) for a in expected_list if str(a).isdigit()} | {
            str(a) for a in expected_list
        }

        # Retrieve documents from vector store
        retrieved_docs = pipeline.retrieve(query, top_k=3)
        retrieved_articles = {
            doc.get("article_number") for doc in retrieved_docs if doc.get("article_number") is not None
        }

        # Hit if any expected article is present in retrieved results
        if bool(expected_set & retrieved_articles):
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