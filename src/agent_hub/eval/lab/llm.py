"""A small, inspectable Ollama client for the generation half of the Eval Lab.

Why not ChatOllama / LangChain here: the lab must record *exactly* what each
judge was asked and what it answered, so a score can be traced back to a
prompt. A thin client over `/api/chat` keeps the request visible, lets every
call pin `temperature` and `seed`, and needs nothing beyond httpx — the same
choice `generate.py` makes for synthetic questions.

Everything the lab asks an LLM goes through the `LLM` protocol, which is also
the seam tests replace with a rule-based fake. No document text leaves the
network: the only endpoint is the stack's own Ollama.
"""

from __future__ import annotations

import json
import math
import os
import threading
from typing import Protocol

import httpx

__all__ = [
    "LLM",
    "OllamaLLM",
    "LLMError",
    "DEFAULT_MODEL",
    "EMBED_MODEL",
    "default_model",
    "cosine",
]

DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"
EMBED_MODEL = "nomic-embed-text"


def default_model() -> str:
    return os.environ.get("EVAL_LAB_JUDGE_MODEL") or os.environ.get(
        "EVAL_LAB_GEN_MODEL", DEFAULT_MODEL
    )


class LLMError(RuntimeError):
    """The model could not produce a usable answer (transport or parse error)."""


class LLM(Protocol):
    def chat_json(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        seed: int = 7,
        num_predict: int = 600,
    ) -> dict: ...

    def chat_text(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        seed: int = 7,
        num_predict: int = 700,
    ) -> str: ...

    def embed(self, texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]: ...

    def models(self) -> list[str]: ...


class OllamaLLM:
    def __init__(self, base_url: str | None = None, timeout: float = 240.0) -> None:
        self.base_url = (
            base_url or os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
        ).rstrip("/")
        self.timeout = timeout
        self._local = threading.local()

    def _client(self) -> httpx.Client:
        # One client per thread: jobs run on their own threads.
        client = getattr(self._local, "client", None)
        if client is None:
            client = httpx.Client(timeout=self.timeout)
            self._local.client = client
        return client

    def _chat(
        self,
        system: str,
        user: str,
        model: str | None,
        temperature: float,
        seed: int,
        num_predict: int,
        json_mode: bool,
    ) -> str:
        body = {
            "model": model or default_model(),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            # temperature AND seed: at temperature 0 the seed is irrelevant; above 0
            # it is what makes a "noisy" judge reproducible run to run.
            "options": {"temperature": temperature, "seed": seed, "num_predict": num_predict},
        }
        if json_mode:
            body["format"] = "json"
        try:
            response = self._client().post(f"{self.base_url}/api/chat", json=body)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama request failed: {exc}") from exc
        return str(response.json().get("message", {}).get("content", ""))

    def chat_json(self, system, user, *, model=None, temperature=0.0, seed=7, num_predict=600):
        last = ""
        # One retry with a different seed: an 8B model occasionally emits a
        # truncated object. Two failures are reported, never guessed around.
        for attempt in range(2):
            last = self._chat(system, user, model, temperature, seed + attempt, num_predict, True)
            try:
                parsed = json.loads(last)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        raise LLMError(f"model did not return a JSON object: {last[:200]!r}")

    def chat_text(self, system, user, *, model=None, temperature=0.0, seed=7, num_predict=700):
        return self._chat(system, user, model, temperature, seed, num_predict, False).strip()

    def embed(self, texts, model=EMBED_MODEL):
        try:
            response = self._client().post(
                f"{self.base_url}/api/embed", json={"model": model, "input": texts}
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama embed failed: {exc}") from exc
        return response.json().get("embeddings", [])

    def models(self) -> list[str]:
        try:
            response = self._client().get(f"{self.base_url}/api/tags", timeout=5.0)
            response.raise_for_status()
        except httpx.HTTPError:
            return []
        return sorted(m.get("name", "") for m in response.json().get("models", []) if m.get("name"))


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
