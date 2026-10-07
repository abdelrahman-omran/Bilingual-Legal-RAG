import json
from pathlib import Path

import pytest
from sentence_transformers import SentenceTransformer


@pytest.fixture(scope="module")
def setup_index(tmp_path_factory):
    """Fixture that runs indexer once on a temporary test DB."""
    db_dir = tmp_path_factory.mktemp("qdrant_test_db")
    corpus_path = Path("data/processed/corpus.json")

    if not corpus_path.exists():
        pytest.skip("data/processed/corpus.json not found.")

    from legal_rag.ingestion.indexer import CorpusIndexer

    indexer = CorpusIndexer(
        collection_name="test_civil_code",
        db_path=str(db_dir),
    )
    indexed_count = indexer.index_corpus(corpus_path)

    return indexer.client, corpus_path, indexed_count, str(db_dir)


def test_qdrant_point_count(setup_index):
    client, corpus_path, indexed_count, _ = setup_index

    with open(corpus_path, "r", encoding="utf-8") as f:
        raw_corpus = json.load(f)

    collection_info = client.get_collection("test_civil_code")

    assert indexed_count == len(raw_corpus)
    assert collection_info.points_count == len(raw_corpus)


def test_vector_retrieval_payload(setup_index):
    client, _, _, _ = setup_index
    encoder = SentenceTransformer("BAAI/bge-m3")

    # Search query in Arabic
    query_vector = encoder.encode("العقد شريعة المتعاقدين").tolist()

    search_result = client.query_points(
        collection_name="test_civil_code",
        query=query_vector,
        limit=1,
    )
    search_results = search_result.points

    assert len(search_results) > 0
    top_hit = search_results[0]

    # Assert expected payload metadata fields exist
    assert "citation" in top_hit.payload
    assert "text_ar" in top_hit.payload
    assert top_hit.payload["article_number"] == 147  # Should match Article 147
