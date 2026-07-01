"""Cross-encoder reranker adapter (stage-2 retrieval).

Backed by a sentence-transformers ``CrossEncoder`` (e.g.
``cross-encoder/ms-marco-MiniLM-L-6-v2``). The model is HEAVY (pulls torch) and
optional, so:

  * it is imported lazily on first use, not at module import;
  * the dependency lives in the optional ``rerank`` extra;
  * every failure (missing package, model download blocked, inference error)
    FAILS OPEN — we log and return the stage-1 order truncated to ``top_n`` — so a
    reranker problem degrades precision but never breaks a chat turn.

Inference is synchronous CPU work, so it runs in a thread via ``asyncio.to_thread``
to keep the event loop responsive.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

from agent_hub.domain.models import RetrievedChunk
from agent_hub.observability.logging import get_logger

log = get_logger("reranker")


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None
        self._unavailable = False  # sticky: don't retry a known-broken load per call

    def _ensure_model(self):
        if self._model is not None or self._unavailable:
            return self._model
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
            log.info("reranker.loaded", model=self._model_name)
        except Exception as exc:  # package missing / download blocked
            self._unavailable = True
            log.warning("reranker.unavailable", model=self._model_name, error=str(exc))
        return self._model

    def _score(self, query: str, chunks: list[RetrievedChunk]) -> list[float]:
        model = self._model
        assert model is not None
        pairs = [[query, c.text] for c in chunks]
        return [float(s) for s in model.predict(pairs)]

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        model = await asyncio.to_thread(self._ensure_model)
        if model is None:  # fail open — keep stage-1 order
            return chunks[:top_n]
        try:
            scores = await asyncio.to_thread(self._score, query, chunks)
        except Exception as exc:
            log.warning("reranker.predict_failed", error=str(exc))
            return chunks[:top_n]
        # Overwrite each chunk's score with the cross-encoder score so downstream
        # citations + the UI reflect the reranked confidence, then sort desc.
        rescored = [replace(c, score=s) for c, s in zip(chunks, scores, strict=True)]
        rescored.sort(key=lambda c: c.score, reverse=True)
        return rescored[:top_n]


class NullReranker:
    """No-op used when reranking is off or deps are absent — keeps CRAG wiring uniform."""

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        return chunks[:top_n]
