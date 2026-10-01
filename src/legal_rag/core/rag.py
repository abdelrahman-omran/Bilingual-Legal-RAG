import os
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


class LegalRAGPipeline:
    """Core RAG pipeline for the Egyptian Civil Code."""

    def __init__(
        self,
        collection_name: str = "egyptian_civil_code",
        embedding_model: str = "BAAI/bge-m3",
        qdrant_path: str = "./data/qdrant_db",
        vllm_base_url: str = "http://localhost:8000/v1",
        llm_model_name: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
    ):
        # 1. Retrieval Components
        self.collection_name = collection_name
        self.encoder = SentenceTransformer(embedding_model)
        self.qdrant_client = QdrantClient(path=qdrant_path)

        # 2. Local vLLM Generator Client
        self.llm_model_name = llm_model_name
        self.llm_client = OpenAI(
            base_url=vllm_base_url,
            api_key="none",  # Local vLLM doesn't require an API key
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Encodes user query and retrieves top_k relevant articles from Qdrant using query_points."""
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

        prompt = f"""You are an expert legal assistant specializing in the Egyptian Civil Code.
Answer the user's question accurately using ONLY the provided legal contexts below.
For every claim or statement in your answer, you MUST cite the corresponding article (e.g., Egyptian Civil Code, Article 147).
If the context does not contain enough information to answer, state clearly that the answer is not found in the provided articles.

Contexts:
{formatted_context}

User Question: {query}

Answer:"""
        return prompt

    def answer(self, query: str, top_k: int = 3) -> dict[str, Any]:
        """End-to-end RAG execution: Retrieve -> Format -> Generate."""
        # Step 1: Retrieval
        contexts = self.retrieve(query, top_k=top_k)

        # Step 2: Prompt Engineering
        prompt = self.format_prompt(query, contexts)

        # Step 3: LLM Inference via local vLLM
        response = self.llm_client.chat.completions.create(
            model=self.llm_model_name,
            messages=[
                {"role": "system", "content": "You are a precise bilingual legal AI assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,  # Low temperature for precise legal QA
        )

        generated_answer = response.choices[0].message.content
        sources = [doc["citation"] for doc in contexts]

        return {
            "question": query,
            "answer": generated_answer,
            "sources": sources,
            "retrieved_contexts": contexts,
        }


if __name__ == "__main__":
    # Quick sanity check for retrieval part before vLLM server is running
    rag = LegalRAGPipeline()
    sample_query = "ما هي شروط العقد والتزاماته؟"
    print(f"Testing Query Retrieval: {sample_query}\n")
    
    retrieved = rag.retrieve(sample_query, top_k=2)
    for doc in retrieved:
        print(f"Retrieved: {doc['citation']} | Score: {doc['score']:.4f}")