"""Pluggable LLM generation backends for the RAG pipeline.

Backends:
- TransformersBackend: local Hugging Face pipeline (default, unchanged behavior).
- LlamaServerBackend: OpenAI-compatible client for a llama.cpp server (llama-server),
  which streams tokens efficiently on CPU.

Selection via env:
    LLM_BACKEND=transformers|llamacpp
    LLM_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
    LLM_BASE_URL=http://llm:8080/v1   (llamacpp only)
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Protocol

from langfuse.decorators import observe


class LLMBackend(Protocol):
    """Minimal contract: chat-style generation with optional streaming."""
    @observe(name="llm_generate")
    def generate(
        self, messages: list[dict[str, str]], max_tokens: int = 512, temperature: float = 0.1
    ) -> str: ...

    def generate_stream(
        self, messages: list[dict[str, str]], max_tokens: int = 512, temperature: float = 0.1
    ) -> Iterator[str]: ...


class TransformersBackend:
    """CPU generation via Hugging Face pipeline (lazy-loaded)."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct") -> None:
        self.model_name = model_name
        self._generator = None

    @property
    def generator(self):
        if self._generator is None:
            from transformers import pipeline

            print(f"Loading local CPU model ({self.model_name})...")
            self._generator = pipeline(
                "text-generation",
                model=self.model_name,
                device="cpu",
                torch_dtype="auto",
            )
        return self._generator
    @observe(name="llm_generate")
    def generate(self, messages, max_tokens=512, temperature=0.1) -> str:
        formatted = self.generator.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = self.generator(
            formatted,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            return_full_text=False,
        )
        return outputs[0]["generated_text"].strip()

    def generate_stream(self, messages, max_tokens=512, temperature=0.1) -> Iterator[str]:
        from threading import Thread

        from transformers import TextIteratorStreamer

        formatted = self.generator.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        streamer = TextIteratorStreamer(
            self.generator.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        thread = Thread(
            target=self.generator,
            kwargs={
                "text": formatted,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0,
                "streamer": streamer,
                "return_full_text": False,
            },
        )
        thread.start()
        for chunk in streamer:
            if chunk:
                yield chunk
        thread.join()



class LlamaServerBackend:
    """Generation through an OpenAI-compatible server:
    llama.cpp locally or vLLM remotely.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        model: str = "local-llm",
        api_key: str = "not-needed",
        default_headers: dict[str, str] | None = None,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers or {},
        )
        self.model = model

    @observe(name="llm_generate")
    def generate(self, messages, max_tokens=512, temperature=0.1) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    def generate_stream(self, messages, max_tokens=512, temperature=0.1):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def build_backend_from_env(
    default_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> LLMBackend:
    """Build a generation backend from environment variables."""

    backend = os.getenv("LLM_BACKEND", "transformers").strip().lower()
    model = os.getenv("LLM_MODEL_NAME", default_model)

    if backend == "transformers":
        return TransformersBackend(model_name=model)

    if backend in {"llamacpp", "vllm"}:
        is_vllm = backend == "vllm"

        default_url = (
            None if is_vllm else "http://localhost:8080/v1"
        )
        base_url = os.getenv("LLM_BASE_URL", default_url)

        if not base_url:
            raise ValueError(
                "LLM_BASE_URL must be set when LLM_BACKEND='vllm'."
            )

        api_key = os.getenv("LLM_API_KEY", "not-needed")

        if is_vllm and not os.getenv("LLM_API_KEY"):
            raise ValueError(
                "LLM_API_KEY must match the --api-key value "
                "configured in the vLLM server."
            )

        headers = (
            {"ngrok-skip-browser-warning": "true"}
            if is_vllm
            else {}
        )

        return LlamaServerBackend(
            base_url=base_url,
            model=model,
            api_key=api_key,
            default_headers=headers,
        )

    raise ValueError(
        f"Unknown LLM_BACKEND: {backend!r} "
        "(expected 'transformers', 'llamacpp', or 'vllm')"
    )