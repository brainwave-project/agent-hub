"""Chunker — deterministic tool (§7.3). Executes the strategy the Classifier chose."""

from __future__ import annotations

from agent_hub.application.ingestion.chunking import chunk_text
from agent_hub.application.ingestion.state import IngestState


class ChunkerAgent:
    async def __call__(self, state: IngestState) -> dict:
        strategy = state.get("strategy", "fixed")
        chunks = chunk_text(state.get("raw_text", ""), strategy)
        return {"chunks": chunks, "history": [f"chunker:{strategy}:{len(chunks)}"]}
