"""BentoML service wrapping the RAG pipeline.

Retrieval + prompt formatting run here; generation is delegated to an LLM
backend (llama.cpp server via OpenAI-compatible API, or local transformers).
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path
from typing import Annotated, Any

import bentoml
from pydantic import BaseModel, Field

from legal_rag.core.llm_backends import build_backend_from_env
from legal_rag.core.rag import LegalRAGPipeline


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


def _default_qdrant_path() -> str:
    """Resolve the Qdrant dir both in a repo checkout and inside a built Bento."""
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "data" / "qdrant_db",  # repo: src/legal_rag/serving/...
        here.parents[2] / "data" / "qdrant_db",  # inside bento (files land in bento src/)
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(candidates[0])


_SENTINEL = object()


def _next_or_stop(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return _SENTINEL


@bentoml.service(traffic={"timeout": 300})
class LegalRAGService:
    """BentoML wrapper: retrieval runs here, LLM generation via backend."""

    def __init__(self) -> None:
        if os.getenv("LLM_BACKEND", "transformers") == "llamacpp":
            base_url = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
            # fail fast with a clear message instead of silently falling back

            try:
                urllib.request.urlopen(base_url.replace("/v1", "/health"), timeout=3)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
                raise RuntimeError(
                    f"llama.cpp server not reachable at {base_url} — run `docker compose up -d llm`"
                )
        self.pipeline = LegalRAGPipeline(
            qdrant_path=os.getenv("QDRANT_PATH", _default_qdrant_path()),
            llm_backend=build_backend_from_env(),
        )

    @bentoml.api
    def health(self) -> dict[str, Any]:
        info = self.pipeline.qdrant_client.get_collection(self.pipeline.collection_name)
        return {"status": "healthy", "documents_indexed": info.points_count}

    @bentoml.api
    async def ask(
        self, question: Annotated[str, Field(min_length=1)]
    ) -> AskResponse:
        result = self.pipeline.answer(question, top_k=3)
        return AskResponse(answer=result["answer"], sources=result["sources"])

    @bentoml.api
    async def ask_stream(
        self, question: Annotated[str, Field(min_length=1)]
    ) -> AsyncGenerator[str, None]:
        """Streams NDJSON: first line = sources, then one JSON object per token."""
        events: Iterator[dict[str, Any]] = self.pipeline.answer_stream(question, top_k=3)
        loop = asyncio.get_running_loop()
        while True:
            event = await loop.run_in_executor(None, _next_or_stop, events)
            if event is _SENTINEL:
                break
            yield json.dumps(event, ensure_ascii=False) + "\n"