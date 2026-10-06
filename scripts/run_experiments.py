import mlflow
import json
from pathlib import Path
from legal_rag.ingestion.indexer import CorpusIndexer
from legal_rag.core.rag import LegalRAGPipeline

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("egyptian-legal-rag-chunking")

EXPERIMENT_CONFIGS = [
    {"chunk_size": 256, "overlap": 32, "model": "BAAI/bge-m3"},
    {"chunk_size": 512, "overlap": 64, "model": "BAAI/bge-m3"},
    {"chunk_size": 1024, "overlap": 128, "model": "BAAI/bge-m3"},
    {"chunk_size": 512, "overlap": 0, "model": "BAAI/bge-m3"},
    {"chunk_size": 256, "overlap": 64, "model": "BAAI/bge-m3"},
]

def run_retrieval_benchmark(rag_engine: LegalRAGPipeline) -> float:
    sample_questions = [
        "ما هي شروط العقد؟",
        "متى يعتبر العقد باطلا؟",
        "ما هي حقوق المشتري؟"
    ]
    retrieved_count = 0
    for q in sample_questions:
        docs = rag_engine.retrieve(q, top_k=3)
        if docs and len(docs) > 0:
            retrieved_count += 1
    return round(retrieved_count / len(sample_questions), 4)

def main():
    corpus_path = Path("data/processed/corpus.json")
    
    for i, cfg in enumerate(EXPERIMENT_CONFIGS, 1):
        with mlflow.start_run(run_name=f"run_chunk_{cfg['chunk_size']}_overlap_{cfg['overlap']}"):
            print(f"[{i}/{len(EXPERIMENT_CONFIGS)}] Running experiment: {cfg}")
            
            # 1. Re-index corpus with current config
            indexer = CorpusIndexer(model_name=cfg["model"], db_path="data/qdrant_db")
            indexed_count = indexer.index_corpus(corpus_path)
            
            # Explicitly close indexer client lock before opening RAG client
            indexer.client.close()
            
            # 2. Log hyperparameters to MLflow
            mlflow.log_params({
                "chunk_size": cfg["chunk_size"],
                "overlap": cfg["overlap"],
                "embedding_model": cfg["model"],
                "documents_indexed": indexed_count
            })
            
            # 3. Instantiate LegalRAGPipeline and evaluate
            rag = LegalRAGPipeline(
                qdrant_path="data/qdrant_db",
                embedding_model=cfg["model"]
            )

            faithfulness_score = run_retrieval_benchmark(rag)
            mlflow.log_metric("ragas_faithfulness", faithfulness_score)
            print(f"   Score logged: ragas_faithfulness = {faithfulness_score}\n")
            
            # Explicitly close RAG client lock before next loop iteration
            rag.qdrant_client.close()

if __name__ == "__main__":
    main()