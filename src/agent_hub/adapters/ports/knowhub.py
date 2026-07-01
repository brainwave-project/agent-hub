"""know-hub ports — retrieval (read) and the document gateway (delegated writes).

These map onto the DocumentInternal gRPC surface (ADR-AGT-08). Splitting read
from write keeps CRAG (read-only) physically unable to touch the write path —
reinforcing Principle 5 at the type level.
"""

from dataclasses import dataclass
from typing import Protocol

from agent_hub.domain.models import RetrievedChunk, TextChunk


@dataclass(frozen=True)
class Payload:
    raw_text: str
    binary: bytes
    mime_type: str
    found: bool


class Retriever(Protocol):
    """CRAG retrieve node — reads only, scoped to the calling user (Issue 1)."""

    async def search(
        self, query: str, top_k: int, actor_user_id: str, is_admin: bool, min_score: float = 0.0
    ) -> list[RetrievedChunk]: ...


class DocumentGateway(Protocol):
    """Ingestion crew's window onto know-hub. The writes (persist/update) are
    delegated to know-hub, which performs them under its own invariants."""

    async def get_payload(self, document_id: str, actor_user_id: str) -> Payload: ...

    async def persist_chunks(
        self, document_id: str, chunks: list[TextChunk], strategy: str, model: str
    ) -> int: ...

    async def update_metadata(
        self,
        document_id: str,
        doc_type: str = "",
        topic: str = "",
        chunk_strategy: str = "",
        status: str = "",
    ) -> bool: ...
