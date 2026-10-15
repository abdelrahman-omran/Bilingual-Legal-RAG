"""Corpus indexing with optional character-based chunking.

Default behavior stays article-based for compatibility with existing tests and
normal indexing. Pass chunk_size and overlap to run a chunking experiment.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


def _make_qdrant_client(db_path: str) -> QdrantClient:
    """Use Qdrant server mode when QDRANT_URL is non-empty; otherwise local mode."""
    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    if qdrant_url:
        return QdrantClient(url=qdrant_url)
    return QdrantClient(path=db_path)


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping character windows, preferring natural breaks.

    ``chunk_size`` and ``overlap`` are measured in Unicode characters, not tokens.
    The boundary heuristic prefers punctuation/whitespace within the last 30% of
    each window while keeping the chunk length at or below ``chunk_size``.
    """
    text = (text or "").strip()
    if not text:
        return [""]
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    if len(text) <= chunk_size:
        return [text]

    separators = "\n\r .!؟?؛;:,،۔"
    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        end = hard_end

        if hard_end < len(text):
            minimum_boundary = start + int(chunk_size * 0.70)
            last_boundary = -1
            for separator in separators:
                last_boundary = max(
                    last_boundary,
                    text.rfind(separator, minimum_boundary, hard_end),
                )
            if last_boundary >= minimum_boundary:
                end = last_boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks or [text]


class CorpusIndexer:
    """Embed and index Egyptian Civil Code articles/chunks in Qdrant."""

    def __init__(
        self,
        collection_name: str = "egyptian_civil_code",
        model_name: str = "BAAI/bge-m3",
        db_path: str = "./data/qdrant_db",
        chunk_size: int | None = None,
        overlap: int = 0,
        encoder: SentenceTransformer | None = None,
    ) -> None:
        if chunk_size is None and overlap != 0:
            raise ValueError("overlap must be 0 when chunk_size is None")
        if chunk_size is not None:
            if chunk_size <= 0:
                raise ValueError("chunk_size must be positive")
            if overlap < 0 or overlap >= chunk_size:
                raise ValueError("overlap must be >= 0 and smaller than chunk_size")

        self.collection_name = collection_name
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Optional injection lets a benchmark share the same encoder for indexing
        # and query embedding, and avoids repeatedly loading the same model.
        self.encoder = encoder or SentenceTransformer(model_name, device="cpu")
        self.vector_size = self.encoder.get_sentence_embedding_dimension()
        self.client = _make_qdrant_client(db_path)

    def init_collection(self) -> None:
        """Create or recreate the configured Qdrant collection."""
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
        with json_path.open("r", encoding="utf-8") as file:
            corpus = json.load(file)
        if not isinstance(corpus, list):
            raise TypeError(
                f"Expected {json_path} to contain a JSON list of article records; "
                f"got {type(corpus).__name__}."
            )
        return corpus

    def _prepare_point(
        self,
        article: dict[str, Any],
        *,
        text_ar: str | None = None,
        text_en: str | None = None,
        chunk_index: int | None = None,
    ) -> tuple[int | str, str, dict[str, Any]]:
        article_number = article["article_number"]
        text_ar = article.get("text_ar", "") if text_ar is None else text_ar
        text_en = article.get("text_en", "") if text_en is None else text_en
        text_to_embed = (
            f"Article {article_number}\nArabic: {text_ar}\nEnglish: {text_en}"
        )
        payload: dict[str, Any] = {
            "article_number": article_number,
            "book": article.get("book", ""),
            "chapter": article.get("chapter", ""),
            "section": article.get("section", ""),
            "topic": article.get("topic", ""),
            "text_ar": text_ar,
            "text_en": text_en,
            "is_repealed": bool(article.get("is_repealed", False)),
            "citation": article.get(
                "citation", f"Egyptian Civil Code, Article {article_number}"
            ),
        }

        if chunk_index is None:
            # Preserve existing article-level IDs and behavior when chunking is off.
            point_id: int | str = article_number
        else:
            # One article may now map to multiple points; IDs must remain unique.
            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"bilingual-legal-rag:{article_number}:chunk:{chunk_index}",
                )
            )
            payload.update(
                {
                    "chunk_index": chunk_index,
                    "chunk_size": self.chunk_size,
                    "overlap": self.overlap,
                    "chunk_unit": "characters",
                }
            )
        return point_id, text_to_embed, payload

    def _build_point(
        self,
        article: dict[str, Any],
        *,
        text_ar: str | None = None,
        text_en: str | None = None,
        chunk_index: int | None = None,
    ) -> PointStruct:
        """Build one point; retained for compatibility with direct callers/tests."""
        point_id, text_to_embed, payload = self._prepare_point(
            article, text_ar=text_ar, text_en=text_en, chunk_index=chunk_index
        )
        vector = self.encoder.encode(text_to_embed).tolist()
        return PointStruct(id=point_id, vector=vector, payload=payload)

    def _prepared_points_for_article(
        self, article: dict[str, Any]
    ) -> list[tuple[int | str, str, dict[str, Any]]]:
        if self.chunk_size is None:
            return [self._prepare_point(article)]

        arabic_chunks = split_text(
            str(article.get("text_ar", "")), self.chunk_size, self.overlap
        )
        english_chunks = split_text(
            str(article.get("text_en", "")), self.chunk_size, self.overlap
        )
        count = max(len(arabic_chunks), len(english_chunks))
        prepared: list[tuple[int | str, str, dict[str, Any]]] = []
        for chunk_index in range(count):
            text_ar = (
                arabic_chunks[chunk_index]
                if chunk_index < len(arabic_chunks)
                else ""
            )
            text_en = (
                english_chunks[chunk_index]
                if chunk_index < len(english_chunks)
                else ""
            )
            prepared.append(
                self._prepare_point(
                    article,
                    text_ar=text_ar,
                    text_en=text_en,
                    chunk_index=chunk_index,
                )
            )
        return prepared

    def upsert_articles(self, articles: list[dict[str, Any]]) -> int:
        """Insert indexed article/chunk points into the existing collection.

        Embeddings are calculated in batches; encoding one short string per model
        invocation would add substantial overhead for fine-grained chunk configs.
        """
        prepared = [
            record
            for article in articles
            for record in self._prepared_points_for_article(article)
        ]
        if not prepared:
            return 0

        texts_to_embed = [record[1] for record in prepared]
        vectors = self.encoder.encode(
            texts_to_embed,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        points = [
            PointStruct(id=point_id, vector=vector.tolist(), payload=payload)
            for (point_id, _text, payload), vector in zip(prepared, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def index_corpus(self, json_path: Path) -> int:
        """Recreate collection, index corpus, and return indexed point count."""
        self.init_collection()
        articles = self.load_corpus(json_path)
        return self.upsert_articles(articles)
