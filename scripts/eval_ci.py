import json
import sys
from pathlib import Path

from legal_rag.core.rag import LegalRAGPipeline


def find_eval_dataset() -> Path:
    """Finds the evaluation dataset across standard candidate locations."""
    candidate_paths = [
        Path("tests/eval/eval_20_questions.json"),
    ]
    for path in candidate_paths:
        if path.exists():
            return path

    print(
        f"Error: Evaluation dataset missing! Searched: {[str(p) for p in candidate_paths]}"
    )
    sys.exit(1)


def run_quality_gate(min_faithfulness_threshold: float = 0.75) -> None:
    """Runs quality evaluation against LegalRAGPipeline on local CPU."""
    eval_path = find_eval_dataset()
    print(f"Loading evaluation dataset from: {eval_path}")

    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", data) if isinstance(data, dict) else data
    print(f"Loaded {len(questions)} evaluation questions.")

    # Initialize local CPU RAG pipeline
    rag_pipeline = LegalRAGPipeline()

    hits = 0
    total = len(questions)

    for idx, item in enumerate(questions, 1):
        q_text = item["question"]
        expected_articles = set(item.get("expected_articles", []))

        # Call pipeline .answer() instead of .query()
        response = rag_pipeline.answer(q_text)
        retrieved_contexts = response.get("retrieved_contexts", [])

        # Extract article numbers from retrieved contexts
        retrieved_articles = {
            doc["article_number"]
            for doc in retrieved_contexts
            if doc.get("article_number") is not None
        }

        # Validate retrieval hit
        hit = len(expected_articles.intersection(retrieved_articles)) > 0
        if hit:
            hits += 1

        print(
            f"[{idx}/{total}] Question: {q_text[:35]}... | "
            f"Expected: {list(expected_articles)} | Retrieved: {list(retrieved_articles)} | Hit: {hit}"
        )

    accuracy = round(hits / total, 4) if total > 0 else 0.0

    print("\n" + "=" * 50)
    print(f"CI Quality Gate Retrieval Score: {accuracy:.2%}")
    print(f"Required Threshold: {min_faithfulness_threshold:.2%}")
    print("=" * 50 + "\n")

    if accuracy < min_faithfulness_threshold:
        print(
            f"❌ CI FAILED: Score ({accuracy:.2%}) is below threshold ({min_faithfulness_threshold:.2%})."
        )
        sys.exit(1)

    print("✅ CI PASSED: Quality gate threshold met.")


if __name__ == "__main__":
    run_quality_gate()
