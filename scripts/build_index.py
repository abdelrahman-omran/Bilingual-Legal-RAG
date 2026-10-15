"""Build the legal-corpus Qdrant index, optionally applying an MLflow winner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from legal_rag.ingestion.indexer import CorpusIndexer


def _load_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    if not config_path.is_file():
        raise FileNotFoundError(f"Chunking config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected a JSON object in {config_path}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Index legal corpus into Qdrant.")
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/corpus.json"),
        help="Path to JSON corpus",
    )
    parser.add_argument(
        "--db-path", type=str, default="data/qdrant_db",
        help="Path to local Qdrant storage directory (ignored if QDRANT_URL is set)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Embedding model name; overrides the model in --config",
    )
    parser.add_argument(
        "--collection-name", type=str, default="egyptian_civil_code",
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Optional chunk size in characters; omit to keep one point per article unless --config is supplied",
    )
    parser.add_argument(
        "--overlap", type=int, default=None,
        help="Optional chunk overlap in characters; must be smaller than chunk size",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Optional reports/best_chunking_config.json from run_experiments.py",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    corpus_path: Path = args.input
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"{corpus_path} does not exist. Run PDF extraction stage first."
        )

    model_name = args.model or str(config.get("embedding_model", "BAAI/bge-m3"))
    configured_chunk_size = config.get("chunk_size")
    configured_overlap = config.get("overlap", 0)
    chunk_size = args.chunk_size if args.chunk_size is not None else configured_chunk_size
    overlap = args.overlap if args.overlap is not None else configured_overlap
    chunk_size = int(chunk_size) if chunk_size is not None else None
    overlap = int(overlap or 0)

    indexer = CorpusIndexer(
        collection_name=args.collection_name,
        model_name=model_name,
        db_path=args.db_path,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    try:
        indexed_count = indexer.index_corpus(corpus_path)
        articles_count = len(indexer.load_corpus(corpus_path))
    finally:
        indexer.client.close()

    # Preserve the existing DVC tracking marker behavior from the original script.
    # This remains useful when Qdrant itself is served remotely over HTTP.
    db_path = Path(args.db_path)
    db_path.mkdir(parents=True, exist_ok=True)
    marker_file = db_path / ".dvc_keep"
    marker_file.write_text("indexed", encoding="utf-8")

    mode = (
        f"chunk_size={chunk_size}, overlap={overlap} characters"
        if chunk_size is not None
        else "one point per article (chunking disabled)"
    )
    print(
        f"Successfully indexed {articles_count} articles as {indexed_count} Qdrant points "
        f"at {args.db_path}; embedding_model={model_name}; {mode}; "
        f"collection={args.collection_name}."
    )


if __name__ == "__main__":
    main()
