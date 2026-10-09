"""Batch re-indexing: upsert new/updated articles into the live Qdrant index."""

import argparse
import json
from pathlib import Path

from legal_rag.ingestion.indexer import CorpusIndexer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upsert articles (corpus schema) into the existing index."
    )
    parser.add_argument("json_path", type=Path, help="Path to a JSON file: one article object or a list of them")
    args = parser.parse_args()

    data = json.loads(args.json_path.read_text(encoding="utf-8"))
    articles = data if isinstance(data, list) else [data]

    indexer = CorpusIndexer()  # honors QDRANT_URL
    n = indexer.upsert_articles(articles)
    print(f"Upserted {n} article(s).")


if __name__ == "__main__":
    main()