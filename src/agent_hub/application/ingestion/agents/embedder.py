"""Embedder — deterministic tool (§7.3). Embeds chunks and hands them to know-hub.

The crew never writes the chunk store itself: it embeds, then delegates the
authoritative write to know-hub via PersistChunks + UpdateMetadata (Principle 5 /
ADR-AGT-06). A failed embed/persist marks the document FAILED via know-hub; an
admin reindex re-runs the whole crew with force.
"""

from __future__ import annotations

from agent_hub.adapters.ports.embedding import Embedder
from agent_hub.adapters.ports.knowhub import DocumentGateway
from agent_hub.application.ingestion.state import IngestState
from agent_hub.observability.logging import get_logger

log = get_logger("embedder")

EMBED_MODEL = "nomic-embed-text"


class EmbedderAgent:
    def __init__(self, embedder: Embedder, gateway: DocumentGateway) -> None:
        self._emb = embedder
        self._gw = gateway

    async def __call__(self, state: IngestState) -> dict:
        chunks = state.get("chunks", [])
        if not chunks:
            return {"indexed": 0, "status": "SKIPPED", "history": ["embedder:no_chunks"]}

        try:
            vectors = await self._emb.embed([c.text for c in chunks])
            for c, v in zip(chunks, vectors, strict=True):
                c.vector = v

            persisted = await self._gw.persist_chunks(
                state["document_id"], chunks, state.get("strategy", "fixed"), EMBED_MODEL
            )
            await self._gw.update_metadata(
                state["document_id"],
                doc_type=state.get("doc_type", ""),
                topic=state.get("topic", ""),
                chunk_strategy=state.get("strategy", ""),
                status="INDEXED",
            )
            return {"indexed": persisted, "status": "INDEXED", "history": [f"embedder:indexed:{persisted}"]}
        except Exception as exc:
            log.error("embedder.failed", document_id=state.get("document_id"), error=str(exc))
            # Mark FAILED via know-hub (the single writer); never raise into the graph.
            try:
                await self._gw.update_metadata(state["document_id"], status="FAILED")
            except Exception:
                pass
            return {"indexed": 0, "status": "FAILED", "history": ["embedder:failed"]}
