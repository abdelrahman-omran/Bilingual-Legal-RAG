import json
import os
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


def _make_qdrant_client(db_path: str) -> QdrantClient:
    """Server mode when QDRANT_URL is set; local file mode otherwise."""
    qdrant_url = os.getenv("QDRANT_URL")
    if qdrant_url:
        return QdrantClient(url=qdrant_url)
    return QdrantClient(path=db_path)


class CorpusIndexer:
    """Handles article-based document embedding and Qdrant vector indexing."""

    def __init__(
        self,
        collection_name: str = "egyptian_civil_code",
        model_name: str = "BAAI/bge-m3",
        db_path: str = "./data/qdrant_db",
    ):
        self.collection_name = collection_name
        self.encoder = SentenceTransformer(model_name, device="cpu")
        self.vector_size = self.encoder.get_sentence_embedding_dimension()
        self.client = _make_qdrant_client(db_path)

    def init_collection(self) -> None:
        """Create or recreate the Qdrant collection."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    def load_corpus(self, json_path: Path) -> list[dict[str, Any]]:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_point(self, art: dict[str, Any]) -> PointStruct:
        """Embed one article and build its Qdrant point (id = article_number)."""
        text_to_embed = f"Article {art['article_number']}\nArabic: {art['text_ar']}\nEnglish: {art['text_en']}"
        vector = self.encoder.encode(text_to_embed).tolist()

        payload = {
            "article_number": art["article_number"],
            "book": art["book"],
            "chapter": art["chapter"],
            "section": art["section"],
            "topic": art["topic"],
            "text_ar": art["text_ar"],
            "text_en": art["text_en"],
            "is_repealed": art["is_repealed"],
            "citation": art["citation"],
        }

        return PointStruct(id=art["article_number"], vector=vector, payload=payload)

    def upsert_articles(self, articles: list[dict[str, Any]]) -> int:
        """Insert or update articles in the existing collection (no rebuild)."""
        points = [self._build_point(art) for art in articles]
        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def index_corpus(self, json_path: Path) -> int:
        self.init_collection()
        articles = self.load_corpus(json_path)
        return self.upsert_articles(articles)