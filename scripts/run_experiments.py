import re
from pathlib import Path

import mlflow

from legal_rag.core.rag import LegalRAGPipeline
from legal_rag.ingestion.indexer import CorpusIndexer

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("egyptian-legal-rag-chunking")

EXPERIMENT_CONFIGS = [
    {"chunk_size": 256, "overlap": 32, "model": "BAAI/bge-m3"},
    {"chunk_size": 512, "overlap": 64, "model": "BAAI/bge-m3"},
    {"chunk_size": 1024, "overlap": 128, "model": "BAAI/bge-m3"},
    {"chunk_size": 512, "overlap": 0, "model": "BAAI/bge-m3"},
    {"chunk_size": 256, "overlap": 64, "model": "BAAI/bge-m3"},
]


def evaluate_llm_faithfulness(
    rag: LegalRAGPipeline, question: str, answer: str, contexts: list[dict]
) -> float:
    """Uses the local CPU generator model to judge whether the generated answer is faithful to the retrieved context."""
    context_text = "\n".join([f"- {c['text_ar'] or c['text_en']}" for c in contexts])

    judge_prompt = f"""You are an expert AI evaluator.
Your task is to evaluate the Faithfulness of an Answer given the Context.

Context:
{context_text}

Answer:
{answer}

Instruction:
Check if every factual claim in the Answer is supported by the Context.
Output ONLY a JSON object with a single key "score" containing a float between 0.0 and 1.0 (where 1.0 means completely faithful and supported, and 0.0 means completely ungrounded/hallucinated).

Example response format:
{{"score": 0.85}}"""

    messages = [
        {
            "role": "system",
            "content": "You are a strict evaluation judge. Output valid JSON only.",
        },
        {"role": "user", "content": judge_prompt},
    ]

    try:
        # Use local CPU pipeline generator instead of vLLM client
        formatted_input = rag.generator.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = rag.generator(
            formatted_input,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=True,
            return_full_text=False,
        )

        content = outputs[0]["generated_text"].strip()

# Extract numeric score from JSON output
        match = re.search(r'"score"\s*:\s*([0-9.]+)', content)
        if match:
            return float(match.group(1))

        # Fallback regex search for any float if JSON extraction fails
        match_float = re.search(r"0\.[0-9]+|1\.0|0", content)
        return float(match_float.group(0)) if match_float else 0.5

    except Exception as e:  # noqa: BLE001
        print(
            f"Warning: Faithfulness evaluation failed for query ({e}). Defaulting to 0.0."
        )
        return 0.0
    
def run_faithfulness_benchmark(rag: LegalRAGPipeline, questions: list[str]) -> float:
    """Executes full RAG generation and evaluates real LLM faithfulness across test questions."""
    scores = []
    for q in questions:
        # 1. Generate answer using rag.answer()
        res = rag.answer(q, top_k=3)

        # 2. Judge faithfulness using the local CPU LLM
        score = evaluate_llm_faithfulness(
            rag=rag,
            question=q,
            answer=res["answer"],
            contexts=res["retrieved_contexts"],
        )
        scores.append(score)

    return round(sum(scores) / len(scores), 4) if scores else 0.0


def main():
    corpus_path = Path("data/processed/corpus.json")
    eval_questions = [
        "ما هي شروط العقد؟",
        "متى يعتبر العقد باطلا؟",
        "ما هي حقوق المشتري؟",
    ]

    best_run_id = None
    best_score = -1.0
    best_config = None

    for i, cfg in enumerate(EXPERIMENT_CONFIGS, 1):
        run_name = f"chunk_{cfg['chunk_size']}_overlap_{cfg['overlap']}_{cfg['model'].split('/')[-1]}"

        with mlflow.start_run(run_name=run_name) as run:
            print(f"[{i}/{len(EXPERIMENT_CONFIGS)}] Running experiment: {cfg}")

            # 1. Re-index corpus with current embedding model configuration
            indexer = CorpusIndexer(model_name=cfg["model"], db_path="data/qdrant_db")
            indexed_count = indexer.index_corpus(corpus_path)
            indexer.client.close()

            # 2. Instantiate CPU-native RAG pipeline
            rag = LegalRAGPipeline(
                qdrant_path="data/qdrant_db",
                embedding_model=cfg["model"],
            )

            # 3. Generate answers & calculate CPU LLM Faithfulness
            faithfulness_score = run_faithfulness_benchmark(rag, eval_questions)

            # 4. Log parameters & metric to MLflow
            mlflow.log_params(
                {
                    "chunk_size": cfg["chunk_size"],
                    "overlap": cfg["overlap"],
                    "embedding_model": cfg["model"],
                    "documents_indexed": indexed_count,
                }
            )
            mlflow.log_metric("faithfulness", faithfulness_score)
            print(f"   Score logged: faithfulness = {faithfulness_score}\n")

            rag.qdrant_client.close()

            if faithfulness_score > best_score:
                best_score = faithfulness_score
                best_run_id = run.info.run_id
                best_config = cfg

    # 5. Register best chunking configuration in MLflow Registry
    if best_run_id:
        print(
            f"=== Registering Best Run ({best_run_id}) with Faithfulness: {best_score} ==="
        )
        with mlflow.start_run(run_id=best_run_id):
            mlflow.set_tag("registry_status", "best_chunking_config")
            mlflow.log_dict(best_config, "best_config.json")


if __name__ == "__main__":
    main()
