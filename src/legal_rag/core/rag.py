"""Bilingual Egyptian Civil Code RAG pipeline with optional per-request RAGAS.

Enable request-time evaluation with RAGAS_ENABLED=true. Successful faithfulness
scores are returned to the caller, sent to Langfuse, and exposed directly as
Prometheus metrics at the BentoML service's /metrics endpoint.
"""
from __future__ import annotations

import logging
import math
import os
import sys
import threading
import types
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langfuse.decorators import langfuse_context, observe
from prometheus_client import Gauge
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from legal_rag.core.llm_backends import LLMBackend, build_backend_from_env

logger = logging.getLogger(__name__)
_RAGAS_LOCK = threading.Lock()
_RAGAS_JUDGE: Any | None = None
_RAGAS_EVALUATE: Any | None = None
_RAGAS_FAITHFULNESS: Any | None = None

# Export the latest successful per-question RAGAS score directly to Prometheus.
# Both score names are kept as aliases for compatibility with existing dashboards.
FAITHFULNESS_SCORE = Gauge(
    "faithfulness",
    "Latest successfully calculated RAGAS faithfulness score",
)
RAGAS_FAITHFULNESS_SCORE = Gauge(
    "ragas_faithfulness",
    "Latest successfully calculated RAGAS faithfulness score",
)
FAITHFULNESS_AVAILABLE = Gauge(
    "ragas_faithfulness_available",
    "Whether a valid RAGAS faithfulness score has been calculated",
)
FAITHFULNESS_SCORE.set(float("nan"))
RAGAS_FAITHFULNESS_SCORE.set(float("nan"))
FAITHFULNESS_AVAILABLE.set(0)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _make_qdrant_client(qdrant_path: str) -> QdrantClient:
    """Use Qdrant server mode when QDRANT_URL is set; local mode otherwise."""
    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    if qdrant_url:
        return QdrantClient(url=qdrant_url)
    return QdrantClient(path=qdrant_path)


def _load_ragas_components() -> tuple[Any, Any, Any]:
    """Lazy-load the same real RAGAS faithfulness stack used by evaluate_ragas.py.

    ChatOpenAI points at any OpenAI-compatible endpoint (normally local
    llama.cpp). Vertex AI is not used. The small module shim matches the
    workaround already used in scripts/evaluate_ragas.py for RAGAS releases
    that import an optional, unavailable LangChain VertexAI module.
    """
    global _RAGAS_JUDGE, _RAGAS_EVALUATE, _RAGAS_FAITHFULNESS
    if _RAGAS_JUDGE is not None:
        return _RAGAS_EVALUATE, _RAGAS_FAITHFULNESS, _RAGAS_JUDGE

    try:
        from langchain_community.chat_models.vertexai import ChatVertexAI  # noqa: F401
    except ImportError:
        module_name = "langchain_community.chat_models.vertexai"
        dummy_module = types.ModuleType(module_name)
        dummy_module.ChatVertexAI = type("ChatVertexAI", (object,), {})
        sys.modules[module_name] = dummy_module

    # These are the same RAGAS imports/wrappers used in scripts/evaluate_ragas.py.
    from langchain_openai import ChatOpenAI
    from ragas import evaluate as ragas_evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import faithfulness

    base_url = (
        os.getenv("RAGAS_LLM_BASE_URL")
        or os.getenv("FAITHFULNESS_JUDGE_BASE_URL")
        or os.getenv("LLM_BASE_URL")
        or "http://localhost:8080/v1"
    )
    model = (
        os.getenv("RAGAS_LLM_MODEL")
        or os.getenv("FAITHFULNESS_JUDGE_MODEL")
        or os.getenv("LLM_MODEL_NAME")
        or "Qwen/Qwen2.5-1.5B-Instruct"
    )
    api_key = (
        os.getenv("RAGAS_LLM_API_KEY")
        or os.getenv("FAITHFULNESS_JUDGE_API_KEY")
        or os.getenv("LLM_API_KEY")
        or "not-needed"
    )
    timeout = float(os.getenv("RAGAS_TIMEOUT_SECONDS", "300"))

    # Deliberately do not set max_tokens here: evaluate_ragas.py leaves this
    # unset, and the previous 256-token override caused LLMDidNotFinishException
    # when RAGAS requested structured verdicts from the local model.
    chat_model = ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=0.0,
        timeout=timeout,
        max_retries=1,
    )
    _RAGAS_JUDGE = LangchainLLMWrapper(chat_model)
    _RAGAS_EVALUATE = ragas_evaluate
    _RAGAS_FAITHFULNESS = faithfulness
    logger.info(
        "RAGAS Faithfulness enabled; judge endpoint=%s, model=%s, timeout=%ss",
        base_url,
        model,
        timeout,
    )
    return _RAGAS_EVALUATE, _RAGAS_FAITHFULNESS, _RAGAS_JUDGE


def _contexts_for_ragas(
    contexts: list[dict[str, Any]], max_chars: int
) -> tuple[list[str], bool]:
    """Format retrieved bilingual context, bounded to fit smaller local LLMs."""
    if max_chars <= 0:
        raise ValueError("RAGAS_MAX_CONTEXT_CHARS must be a positive integer")

    formatted: list[str] = []
    remaining = max_chars
    truncated = False
    for doc in contexts:
        citation = str(doc.get("citation") or f"Article {doc.get('article_number', '?')}")
        ar = str(doc.get("text_ar") or "")
        en = str(doc.get("text_en") or "")
        content = f"{citation}\nArabic: {ar}\nEnglish: {en}".strip()
        if not content:
            continue
        if len(content) > remaining:
            content = content[:remaining]
            truncated = True
        if content:
            formatted.append(content)
            remaining -= len(content)
        if remaining <= 0:
            if doc is not contexts[-1]:
                truncated = True
            break
    return formatted, truncated


def _calculate_ragas_faithfulness(
    question: str, answer: str, contexts: list[dict[str, Any]]
) -> tuple[float, bool]:
    """Compute the real RAGAS Faithfulness metric for one Q&A pair.

    This follows scripts/evaluate_ragas.py: Dataset.from_list +
    ragas.evaluate(metrics=[faithfulness], llm=LangchainLLMWrapper(ChatOpenAI),
    and RunConfig(max_workers=1, timeout=300). The synchronous RAGAS call is
    run in a separate thread so RAGAS/nest_asyncio cannot patch or disturb the
    BentoML request's event loop.
    """
    from datasets import Dataset
    from ragas.run_config import RunConfig

    max_context_chars = int(os.getenv("RAGAS_MAX_CONTEXT_CHARS", "2400"))
    ragas_contexts, was_truncated = _contexts_for_ragas(contexts, max_context_chars)
    if not answer.strip():
        raise ValueError("Cannot evaluate RAGAS faithfulness for an empty answer")
    if not ragas_contexts:
        raise ValueError("Cannot evaluate RAGAS faithfulness without retrieved context")

    dataset = Dataset.from_list(
        [
            {
                "question": question,
                "answer": answer,
                "contexts": ragas_contexts,
            }
        ]
    )

    evaluate, metric, judge = _load_ragas_components()
    timeout_seconds = int(float(os.getenv("RAGAS_TIMEOUT_SECONDS", "300")))
    run_config = RunConfig(max_workers=1, timeout=timeout_seconds)

    # Ragas' synchronous evaluate() uses asyncio internally. Isolating it in a
    # worker thread avoids patching the serving server's active event loop.
    with _RAGAS_LOCK, ThreadPoolExecutor(max_workers=1, thread_name_prefix="ragas-faithfulness") as pool:
            future = pool.submit(
                evaluate,
                dataset,
                metrics=[metric],
                llm=judge,
                run_config=run_config,
                raise_exceptions=True,
            )
            result = future.result(timeout=timeout_seconds + 30)

    value = float(result["faithfulness"][0])
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"RAGAS returned an invalid faithfulness score: {value!r}")
    return value, was_truncated


class LegalRAGPipeline:
    """Core RAG pipeline for the Egyptian Civil Code, with optional RAGAS scoring."""

    def __init__(
        self,
        collection_name: str = "egyptian_civil_code",
        embedding_model: str = "BAAI/bge-m3",
        qdrant_path: str = "./data/qdrant_db",
        llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        llm_backend: LLMBackend | None = None,
    ):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.encoder = SentenceTransformer(embedding_model, device="cpu")
        self.qdrant_client = _make_qdrant_client(qdrant_path)
        self.llm_backend = (
            llm_backend
            if llm_backend is not None
            else build_backend_from_env(default_model=llm_model_name)
        )
        print(f"Selected LLM backend: {type(self.llm_backend).__name__}")

    @observe(name="retrieve")
    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Embed the query and retrieve the top-k relevant articles/chunks."""
        query_vector = self.encoder.encode(query).tolist()
        search_result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        )
        retrieved_docs = []
        for hit in search_result.points:
            payload = hit.payload or {}
            retrieved_docs.append(
                {
                    "score": hit.score,
                    "article_number": payload.get("article_number"),
                    "text_ar": payload.get("text_ar"),
                    "text_en": payload.get("text_en"),
                    "citation": payload.get("citation"),
                }
            )
        return retrieved_docs

    def format_prompt(self, query: str, contexts: list[dict[str, Any]]) -> str:
        """Construct a strict prompt requiring article citations."""
        formatted_context = ""
        for idx, doc in enumerate(contexts, 1):
            formatted_context += (
                f"--- Context {idx} [{doc.get('citation', 'Unknown article')}] ---\n"
                f"Arabic: {doc.get('text_ar', '')}\n"
                f"English: {doc.get('text_en', '')}\n\n"
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

    def _build_messages(
        self, query: str, contexts: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are a precise bilingual legal AI assistant.",
            },
            {"role": "user", "content": self.format_prompt(query, contexts)},
        ]

    def _should_use_ragas(self, use_ragas: bool | None) -> bool:
        # Read the environment at call time so the switch is centralized and
        # testable. Set RAGAS_ENABLED before launching the serving process.
        return _env_bool("RAGAS_ENABLED", default=False) if use_ragas is None else use_ragas

    def _evaluate_and_record(
        self,
        query: str,
        answer: str,
        contexts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        score: float | None = None
        error_message: str | None = None
        context_truncated = False
        try:
            score, context_truncated = _calculate_ragas_faithfulness(
                question=query, answer=answer, contexts=contexts
            )
            logger.info("RAGAS faithfulness for question: %.4f", score)
            # Update Prometheus only after a valid score has been calculated.
            FAITHFULNESS_SCORE.set(score)
            RAGAS_FAITHFULNESS_SCORE.set(score)
            FAITHFULNESS_AVAILABLE.set(1)
            try:
                # The answer() method is Langfuse-observed, so associate this
                # numeric evaluation with that same request trace when tracing is on.
                langfuse_context.score_current_trace(
                    name="faithfulness",
                    value=score,
                    comment="RAGAS Faithfulness score for this question",
                )
            except Exception:
                # Langfuse is optional; scoring must not break /ask.
                logger.debug("Could not attach RAGAS score to Langfuse trace", exc_info=True)
        except Exception as exc:  # don't fail the user's answer if evaluation fails
            error_message = f"{type(exc).__name__}: {exc}"[:500]
            logger.exception("RAGAS faithfulness calculation failed")

        result: dict[str, Any] = {
            "ragas_enabled": True,
            "faithfulness": score,
            "faithfulness_context_truncated": context_truncated,
        }
        if error_message:
            result["faithfulness_error"] = error_message
        return result

    @observe(name="rag_answer")
    def answer(
        self, query: str, top_k: int = 3, use_ragas: bool | None = None
    ) -> dict[str, Any]:
        """Retrieve and answer; optionally calculate RAGAS faithfulness per ask.

        Environment switch: RAGAS_ENABLED=true|false. A caller may override it
        for one call with use_ragas=True/False.
        """
        contexts = self.retrieve(query, top_k=top_k)
        messages = self._build_messages(query, contexts)
        generated_answer = self.llm_backend.generate(
            messages, max_tokens=512, temperature=0.1
        )
        result: dict[str, Any] = {
            "question": query,
            "answer": generated_answer,
            "sources": [doc["citation"] for doc in contexts],
            "retrieved_contexts": contexts,
            "trace_id": langfuse_context.get_current_trace_id(),
            "ragas_enabled": self._should_use_ragas(use_ragas),
            "faithfulness": None,
        }
        if result["ragas_enabled"]:
            result.update(self._evaluate_and_record(query, generated_answer, contexts))
        return result

    def answer_stream(
        self, query: str, top_k: int = 3, use_ragas: bool | None = None
    ) -> Iterator[dict[str, Any]]:
        """Stream answer tokens; when enabled, emit a final faithfulness event."""
        contexts = self.retrieve(query, top_k=top_k)
        yield {"type": "sources", "sources": [doc["citation"] for doc in contexts]}
        messages = self._build_messages(query, contexts)
        answer_parts: list[str] = []
        for token in self.llm_backend.generate_stream(
            messages, max_tokens=512, temperature=0.1
        ):
            answer_parts.append(token)
            yield {"type": "token", "token": token}

        if self._should_use_ragas(use_ragas):
            evaluation = self._evaluate_and_record(
                query, "".join(answer_parts), contexts
            )
            yield {"type": "faithfulness", **evaluation}


