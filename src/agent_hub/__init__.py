"""Brainwave AgentHub — the platform's first reasoning service (Phase 3).

A stateless orchestrator that *reasons* (grades retrievals, classifies
documents, decides the next step) but never performs authoritative writes —
know-hub remains the system of record (Principle 5 / ADR-AGT-06).

Two domains, one process (see TDD §5.4):
  - application.crag       — single-agent Corrective RAG (HTTP/SSE, user-facing)
  - application.ingestion  — multi-agent ingestion crew (Kafka-triggered, background)

They share only the adapters layer; neither imports from the other.
"""

__version__ = "0.1.0"
