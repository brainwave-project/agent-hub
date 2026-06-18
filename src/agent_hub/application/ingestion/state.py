"""Ingestion crew shared state (§7.5).

Each agent reads what it needs and returns only the keys it produced — a handoff
is just a state update plus the supervisor re-deciding `next`. The `history`
reducer appends every transition, giving a free per-document audit trail that is
persisted to agent.agent_steps.
"""

import operator
from typing import Annotated, TypedDict

from agent_hub.domain.models import TextChunk


class IngestState(TypedDict, total=False):
    # inputs (from the Kafka event)
    document_id: str
    actor_user_id: str
    force: bool

    # produced by Extractor
    raw_text: str
    binary_mime: str
    has_text: bool          # False → image-only / no text layer → SKIPPED

    # produced by Classifier
    doc_type: str
    topic: str
    strategy: str           # clause | slide | turn | fixed

    # produced by Chunker / Embedder
    chunks: list[TextChunk]
    indexed: int

    # control
    next: str               # supervisor's routing decision
    status: str             # INDEXED | SKIPPED | FAILED
    history: Annotated[list[str], operator.add]
