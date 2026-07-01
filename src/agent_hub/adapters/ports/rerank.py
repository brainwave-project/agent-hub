"""Reranker port — stage-2 of two-stage retrieval.

Stage 1 (pgvector bi-encoder) is fast and recall-oriented but scores the query
and each chunk *independently*. A cross-encoder scores the (query, chunk) pair
*jointly*, which is far more precise — at a cost, so it only ever runs over the
small candidate pool stage 1 returns. Kept behind a port so CRAG depends on the
capability, not on sentence-transformers/torch.
"""

from typing import Protocol

from agent_hub.domain.models import RetrievedChunk


class Reranker(Protocol):
    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        """Re-score chunks against the query and return the best ``top_n``.

        Implementations must fail open: if the model is unavailable, return the
        input order (optionally truncated) rather than raising.
        """
        ...
