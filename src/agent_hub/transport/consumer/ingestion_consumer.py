"""Kafka consumer driving the ingestion crew (ADR-AGT-05).

Subscribes to brainwave.document.events as a NEW consumer group, reacting to
DOCUMENT_UPLOADED. This retires the Phase-1 in-process semantic worker: set
SEMANTIC_WORKER_ENABLED=false in know-hub and the crew owns ingestion. Async +
non-blocking — a slow Ollama leaves a document PENDING, never failing the upload
path (§7.7).

Idempotency: the consumer group's committed offset means each event is handled
once; re-ingestion is safe because know-hub's PersistChunks does a delete-then-
insert replace. A forced reindex re-runs the whole crew.

Runs as a background asyncio task in the FastAPI lifespan — one service, two
domains (§5.4); this is the ingestion code path, the HTTP app is CRAG's.
"""

from __future__ import annotations

import asyncio
import json

from aiokafka import AIOKafkaConsumer

from agent_hub.application.ingestion.service import IngestionService
from agent_hub.domain.contracts import (
    ACTION_DOCUMENT_INDEXED,
    ACTION_DOCUMENT_UPLOADED,
    CONSUMER_GROUP_INGESTION_CREW,
    TOPIC_DOCUMENT,
)
from agent_hub.observability.logging import get_logger

log = get_logger("ingestion-consumer")


async def _consume(ctx) -> None:
    settings = ctx.settings
    service = IngestionService(
        gateway=ctx.grpc, classifier=ctx.reasoner, embedder=ctx.embedder, runs_repo=ctx.runs
    )

    consumer = AIOKafkaConsumer(
        TOPIC_DOCUMENT,
        bootstrap_servers=settings.kafka_brokers.split(","),
        group_id=CONSUMER_GROUP_INGESTION_CREW,
        enable_auto_commit=True,
        auto_offset_reset="latest",
        value_deserializer=lambda v: v,
    )
    await consumer.start()
    log.info("ingestion_consumer.started", group=CONSUMER_GROUP_INGESTION_CREW)
    try:
        async for msg in consumer:
            try:
                event = json.loads(msg.value.decode("utf-8"))
            except (json.JSONDecodeError, AttributeError):
                continue
            if event.get("action") != ACTION_DOCUMENT_UPLOADED:
                continue
            doc_id = (event.get("resource") or {}).get("id")
            if not doc_id:
                continue
            actor = (event.get("actor") or {}).get("userId", "")
            correlation_id = event.get("correlationId") or ""

            log.info("ingestion_consumer.handling", document_id=doc_id)
            status = await service.run(doc_id, actor_user_id=actor)

            # Emit the crew-complete event for the audit trail (§10.1).
            await ctx.events.emit(
                ACTION_DOCUMENT_INDEXED,
                actor_user_id=actor,
                resource_id=doc_id,
                correlation_id=correlation_id,
                metadata={"status": status},
            )
    except asyncio.CancelledError:
        log.info("ingestion_consumer.cancelled")
        raise
    finally:
        await consumer.stop()


async def start_ingestion_consumer(ctx) -> asyncio.Task:
    """Launch the consumer as a background task; returns it so lifespan can cancel."""
    return asyncio.create_task(_consume(ctx))
