"""Reasoning ports — the LLM-backed judgement steps.

Kept semantic (grade / rewrite_query / generate / check_grounded / classify)
rather than a generic `complete()` so the prompt design lives in the adapter and
nodes stay thin and declarative. Each method that needs determinism runs the
model at temperature 0; generation runs slightly warmer (§4.4 principal's note).
"""

from collections.abc import AsyncIterator
from typing import Protocol

from agent_hub.domain.models import DocClass, Grade, Groundedness, RetrievedChunk


class CragReasoner(Protocol):
    """The four LLM judgements of the CRAG graph (§6.4)."""

    async def grade(self, question: str, chunks: list[RetrievedChunk]) -> Grade:
        """Judge whether the chunks actually answer the question (temp 0, structured)."""
        ...

    async def rewrite_query(
        self, question: str, current_query: str, chunks: list[RetrievedChunk]
    ) -> str:
        """Rewrite the query to be more retrievable (temp 0)."""

    async def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Compose a grounded, cited answer from accepted chunks (temp ~0.2)."""

    async def stream_generate(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> AsyncIterator[str]:
        """Token stream of generate() — drives the SSE `token` events (M5)."""
        # Protocol stub; concrete adapters implement as an async generator.
        raise NotImplementedError
        yield ""  # pragma: no cover  (marks this an async generator for typing)

    async def check_grounded(self, answer: str, chunks: list[RetrievedChunk]) -> Groundedness:
        """Verify every claim is supported by a chunk (temp 0, structured)."""


class Classifier(Protocol):
    """The ingestion crew's only real reasoning step (§7.3)."""

    async def classify(self, sample_text: str) -> DocClass:
        """Infer {doc_type, topic, suggested_strategy} from a text sample."""
