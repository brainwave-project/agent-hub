"""know-hub DocumentInternal gRPC client (ADR-AGT-08).

Implements both know-hub ports over a single shared async channel (§5.4.2 — one
connection pool for both domains). Cross-language client: TS server, Python
client, the proven authhub pattern.

Identity propagation (review Issue 1): SemanticSearch forwards the calling
user's id + admin flag so know-hub applies its ownership predicate — CRAG can
only ever retrieve the caller's documents.
"""

from __future__ import annotations

import grpc

from agent_hub.adapters.grpc import knowhub_pb2 as pb
from agent_hub.adapters.grpc import knowhub_pb2_grpc as pb_grpc
from agent_hub.adapters.ports.knowhub import Payload
from agent_hub.domain.models import RetrievedChunk, TextChunk
from agent_hub.observability.logging import get_logger

log = get_logger("knowhub-grpc")


class KnowHubGrpcClient:
    """Retriever + DocumentGateway. One channel, lazily opened, reused per process."""

    def __init__(self, host: str, port: int) -> None:
        self._target = f"{host}:{port}"
        self._channel: grpc.aio.Channel | None = None
        self._stub: pb_grpc.DocumentInternalStub | None = None

    def _ensure(self) -> pb_grpc.DocumentInternalStub:
        if self._stub is None:
            self._channel = grpc.aio.insecure_channel(self._target)
            self._stub = pb_grpc.DocumentInternalStub(self._channel)
        return self._stub

    async def close(self) -> None:
        if self._channel is not None:
            await self._channel.close()
            self._channel, self._stub = None, None

    # -- Retriever ----------------------------------------------------------

    async def search(
        self, query: str, top_k: int, actor_user_id: str, is_admin: bool, min_score: float = 0.0
    ) -> list[RetrievedChunk]:
        stub = self._ensure()
        req = pb.SemanticSearchRequest(
            query=query,
            top_k=top_k,
            actor_user_id=actor_user_id,
            is_admin=is_admin,
            min_score=min_score,
        )
        resp = await stub.SemanticSearch(req)
        return [
            RetrievedChunk(
                chunk_id=r.chunk_id,
                text=r.text,
                score=r.score,
                document_id=r.document_id,
                title=r.title,
            )
            for r in resp.results
        ]

    # -- DocumentGateway ----------------------------------------------------

    async def get_payload(self, document_id: str, actor_user_id: str) -> Payload:
        stub = self._ensure()
        resp = await stub.GetPayload(
            pb.GetPayloadRequest(document_id=document_id, actor_user_id=actor_user_id)
        )
        return Payload(
            raw_text=resp.raw_text,
            binary=resp.binary,
            mime_type=resp.mime_type,
            found=resp.found,
        )

    async def persist_chunks(
        self, document_id: str, chunks: list[TextChunk], strategy: str, model: str
    ) -> int:
        stub = self._ensure()
        req = pb.PersistChunksRequest(
            document_id=document_id,
            strategy=strategy,
            model=model,
            chunks=[
                pb.Chunk(text=c.text, vector=c.vector, seq=c.seq, token_count=c.token_count)
                for c in chunks
            ],
        )
        resp = await stub.PersistChunks(req)
        return resp.persisted

    async def update_metadata(
        self,
        document_id: str,
        doc_type: str = "",
        topic: str = "",
        chunk_strategy: str = "",
        status: str = "",
    ) -> bool:
        stub = self._ensure()
        resp = await stub.UpdateMetadata(
            pb.UpdateMetadataRequest(
                document_id=document_id,
                doc_type=doc_type,
                topic=topic,
                chunk_strategy=chunk_strategy,
                status=status,
            )
        )
        return resp.ok
