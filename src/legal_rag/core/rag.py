import os
from collections.abc import Iterator
from typing import Any

from langfuse.decorators import langfuse_context, observe
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from legal_rag.core.llm_backends import LLMBackend, TransformersBackend


def _make_qdrant_client(qdrant_path: str) -> QdrantClient:
    """Server mode when QDRANT_URL is set (no file lock, multi-process safe);
    local file mode otherwise (single-process, used by tests)."""
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    if qdrant_url:
        return QdrantClient(url=qdrant_url)
    return QdrantClient(path=qdrant_path)


class LegalRAGPipeline:
    """Core RAG pipeline for the Egyptian Civil Code running locally on CPU."""

    def __init__(
        self,
        collection_name: str = "egyptian_civil_code",
        embedding_model: str = "BAAI/bge-m3",
        qdrant_path: str = "./data/qdrant_db",
        llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        llm_backend: LLMBackend | None = None,
    ):
        # 1. Retrieval Components
        self.collection_name = collection_name
        self.encoder = SentenceTransformer(embedding_model, device="cpu")
        self.qdrant_client = _make_qdrant_client(qdrant_path)

        # 2. Generation (pluggable backend; defaults to local transformers)
        self.llm_backend = llm_backend or TransformersBackend(model_name=llm_model_name)
    @observe(name="retrieve")
    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Encodes user query and retrieves top_k relevant articles from Qdrant."""
        query_vector = self.encoder.encode(query).tolist()

        search_result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        )

        retrieved_docs = []
        for hit in search_result.points:
            retrieved_docs.append({
                "score": hit.score,
                "article_number": hit.payload.get("article_number"),
                "text_ar": hit.payload.get("text_ar"),
                "text_en": hit.payload.get("text_en"),
                "citation": hit.payload.get("citation"),
            })
        return retrieved_docs

    def format_prompt(self, query: str, contexts: list[dict[str, Any]]) -> str:
        """Constructs a strict system/user prompt forcing citation usage."""
        formatted_context = ""
        for idx, doc in enumerate(contexts, 1):
            formatted_context += (
                f"--- Context {idx} [{doc['citation']}] ---\n"
                f"Arabic: {doc['text_ar']}\n"
                f"English: {doc['text_en']}\n\n"
            )

        return f"""You are an expert legal assistant specializing in the Egyptian Civil Code.
Answer the user's question accurately using ONLY the provided legal contexts below.
For every claim or statement in your answer, you MUST cite the corresponding article (e.g., Egyptian Civil Code, Article 147).
If the context does not contain enough information to answer, state clearly that the answer is not found in the provided articles.
If the question is in English, answer in English. If the question is in Arabic, answer in Arabic.
Contexts:
{formatted_context}

User Question: {query}

Answer:"""

    def _build_messages(self, query: str, contexts: list[dict[str, Any]]) -> list[dict[str, str]]:
        prompt = self.format_prompt(query, contexts)
        return [
            {"role": "system", "content": "You are a precise bilingual legal AI assistant."},
            {"role": "user", "content": prompt},
        ]
    @observe(name="rag_answer")
    def answer(self, query: str, top_k: int = 3) -> dict[str, Any]:
        """End-to-end RAG execution: Retrieve -> Format -> Generate."""
        contexts = self.retrieve(query, top_k=top_k)
        messages = self._build_messages(query, contexts)

        generated_answer = self.llm_backend.generate(
            messages, max_tokens=512, temperature=0.1
        )
        sources = [doc["citation"] for doc in contexts]
        
        return {
            "question": query,
            "answer": generated_answer,
            "sources": sources,
            "retrieved_contexts": contexts,
            "trace_id": langfuse_context.get_current_trace_id(), 
        }

    def answer_stream(self, query: str, top_k: int = 3) -> Iterator[dict[str, Any]]:
        """Streams the RAG answer.

        Yields:
            {"type": "sources", "sources": [...]}   once, before generation
            {"type": "token", "token": "..."}       per generated token
        """
        contexts = self.retrieve(query, top_k=top_k)
        yield {"type": "sources", "sources": [doc["citation"] for doc in contexts]}

        messages = self._build_messages(query, contexts)
        for token in self.llm_backend.generate_stream(
            messages, max_tokens=512, temperature=0.1
        ):
            yield {"type": "token", "token": token}
pipeline = LegalRAGPipeline()