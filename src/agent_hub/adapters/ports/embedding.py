"""Embedding port — the ingestion crew's Embedder (deterministic tool, §7.3).

Same nomic-embed-text model as Phase 1, served by the same Ollama box.
"""

from typing import Protocol


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one 768-dim vector per input text (order-preserving)."""
        ...


class StepRecorder(Protocol):
    """Records a node/decision into agent.agent_steps (M4). A no-op recorder is
    injected before the durability milestone so graphs run identically."""

    async def record(
        self,
        node: str,
        decision: str = "",
        input_summary: str = "",
        output_summary: str = "",
        latency_ms: int = 0,
    ) -> None: ...
