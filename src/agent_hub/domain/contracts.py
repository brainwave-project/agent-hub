"""Kafka/event contracts — the Python mirror of @brainwave-project/core's
events.ts (review note: polyglot duplication).

agent-hub (Python) cannot import the TS core package, so the wire contract is
necessarily re-declared here. core/src/kafka/events.ts is the SOURCE OF TRUTH;
this file must track it. The two new Phase-3 actions (DOCUMENT_CLASSIFIED,
DOCUMENT_INDEXED) and the INGESTION_CREW consumer group are added to BOTH.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

# --- topic registry (mirror of KAFKA_TOPICS) ---
TOPIC_DOCUMENT = "brainwave.document.events"

# --- consumer groups (mirror of CONSUMER_GROUPS) ---
CONSUMER_GROUP_INGESTION_CREW = "brainwave.agent-hub.ingestion-crew"

# --- audit actions (subset we consume/emit) ---
ACTION_DOCUMENT_UPLOADED = "DOCUMENT_UPLOADED"
ACTION_DOCUMENT_CLASSIFIED = "DOCUMENT_CLASSIFIED"  # NEW (Phase 3)
ACTION_DOCUMENT_INDEXED = "DOCUMENT_INDEXED"        # NEW (Phase 3)


def build_event(
    action: str,
    *,
    actor_user_id: str,
    resource_id: str,
    resource_name: str = "",
    outcome: str = "SUCCESS",
    correlation_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a DomainEvent envelope identical in shape to core's interface.

    NOTE: core's `service` union must gain 'agent-hub' (it currently lists only
    auth/user/know-hub) — see the core diff. Audit-hub logs events generically,
    so it tolerates the new service + actions without code changes.
    """
    return {
        "eventId": str(uuid.uuid4()),
        "occurredAt": datetime.now(UTC).isoformat(),
        "service": "agent-hub",
        "action": action,
        "actor": {"userId": actor_user_id},
        "resource": {"type": "Document", "id": resource_id, "name": resource_name},
        "outcome": outcome,
        "correlationId": correlation_id or None,
        "metadata": metadata or {},
    }
