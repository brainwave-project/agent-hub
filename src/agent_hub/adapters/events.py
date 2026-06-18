"""Kafka producer for the crew's domain events (§10.1).

Emits DOCUMENT_CLASSIFIED (doc typed) and DOCUMENT_INDEXED (crew complete) so
audit-hub records the agentic ingestion trail. Fire-and-forget: an emit failure
is logged and never fails the run (same contract as the TS hubs).
"""

from __future__ import annotations

import json

from aiokafka import AIOKafkaProducer

from agent_hub.domain.contracts import TOPIC_DOCUMENT, build_event
from agent_hub.observability.logging import get_logger

log = get_logger("events")


class EventProducer:
    def __init__(self, brokers: str) -> None:
        self._brokers = brokers
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._brokers.split(","),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
        )
        await self._producer.start()

    async def close(self) -> None:
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None

    async def emit(self, action: str, *, actor_user_id: str, resource_id: str, **kw) -> None:
        if self._producer is None:
            return
        event = build_event(action, actor_user_id=actor_user_id, resource_id=resource_id, **kw)
        try:
            await self._producer.send_and_wait(TOPIC_DOCUMENT, key=event["eventId"], value=event)
        except Exception as exc:  # never block the run on the audit trail
            log.warning("event.emit_failed", action=action, error=str(exc))
