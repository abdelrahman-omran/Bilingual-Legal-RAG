import argparse
from pathlib import Path

from legal_rag.ingestion.indexer import CorpusIndexer


def main() -> None:
    parser = argparse.ArgumentParser(description="Index legal corpus into Qdrant.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/corpus.json",
        help="Path to JSON corpus",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/qdrant_db",
        help="Path to Qdrant storage directory",
    )
    parser.add_argument(
        "--model", type=str, default="BAAI/bge-m3", help="Embedding model name"
    )

    args = parser.parse_args()

    corpus_path = Path(args.input)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"{corpus_path} does not exist. Run PDF extraction stage first."
        )

    indexer = CorpusIndexer(model_name=args.model, db_path=args.db_path)
    indexed_count = indexer.index_corpus(corpus_path)
    print(
        f"Successfully indexed {indexed_count} articles into Qdrant at {args.db_path}"
    )


if __name__ == "__main__":
    main()
