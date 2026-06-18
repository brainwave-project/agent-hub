"""CRAG nodes (§6.4).

Each node is `(state) -> partial_state`. Deterministic nodes (retrieve, abstain)
do no reasoning; LLM nodes (grade, transform_query, generate, check_grounded)
delegate the judgement to the CragReasoner port. Dependencies are injected so
the whole graph is exercisable with fakes — no Ollama, no know-hub (review:
verifiable graph logic).

The corrective loop and BOTH guardrails live in the conditional-edge functions
at the bottom: the query-rewrite cap (loops, §6.4) and the grounding-retry cap
(regen_count, review Issue 4).
"""

from __future__ import annotations

import time

from agent_hub.adapters.ports.embedding import StepRecorder
from agent_hub.adapters.ports.knowhub import Retriever
from agent_hub.adapters.ports.reasoning import CragReasoner
from agent_hub.application.crag.state import CragState
from agent_hub.domain.models import Citation, RetrievedChunk
from agent_hub.observability.logging import get_logger

log = get_logger("crag")

ABSTAIN_MESSAGE = (
    "I don't have enough information in the available documents to answer that "
    "confidently."
)


def _citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    """One citation per source document, best score first."""
    best: dict[str, Citation] = {}
    for c in chunks:
        existing = best.get(c.document_id)
        if existing is None or c.score > existing.score:
            best[c.document_id] = Citation(document_id=c.document_id, title=c.title, score=c.score)
    return sorted(best.values(), key=lambda x: x.score, reverse=True)


class CragNodes:
    def __init__(
        self,
        retriever: Retriever,
        reasoner: CragReasoner,
        recorder: StepRecorder,
        *,
        top_k: int = 6,
        min_score: float = 0.0,
    ) -> None:
        self._retriever = retriever
        self._reasoner = reasoner
        self._rec = recorder
        self._top_k = top_k
        self._min_score = min_score

    # -- deterministic ------------------------------------------------------

    async def retrieve(self, state: CragState) -> dict:
        t0 = time.monotonic()
        query = state.get("query") or state["question"]
        chunks = await self._retriever.search(
            query=query,
            top_k=self._top_k,
            actor_user_id=state.get("actor_user_id", ""),
            is_admin=state.get("is_admin", False),
            min_score=self._min_score,
        )
        top = max((c.score for c in chunks), default=0.0)
        await self._rec.record(
            "retrieve",
            input_summary=f"query={query!r}",
            output_summary=f"{len(chunks)} chunks, top={top:.2f}",
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        return {"query": query, "chunks": chunks}

    async def abstain(self, state: CragState) -> dict:
        await self._rec.record("abstain", decision="insufficient_context")
        return {"answer": ABSTAIN_MESSAGE, "abstained": True, "citations": []}

    # -- LLM ----------------------------------------------------------------

    async def grade(self, state: CragState) -> dict:
        t0 = time.monotonic()
        verdict = await self._reasoner.grade(state["question"], state.get("chunks", []))
        await self._rec.record(
            "grade",
            decision=f"relevant={verdict.relevant}",
            output_summary=verdict.reason,
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        return {"relevant": verdict.relevant}

    async def transform_query(self, state: CragState) -> dict:
        t0 = time.monotonic()
        new_query = await self._reasoner.rewrite_query(
            state["question"], state.get("query", ""), state.get("chunks", [])
        )
        await self._rec.record(
            "transform_query",
            input_summary=f"was={state.get('query')!r}",
            output_summary=f"now={new_query!r}",
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        # reducer adds 1 → the corrective-loop counter survives the cycle.
        return {"query": new_query, "loops": 1}

    async def generate(self, state: CragState) -> dict:
        t0 = time.monotonic()
        chunks = state.get("chunks", [])
        answer = await self._reasoner.generate(state["question"], chunks)
        await self._rec.record(
            "generate",
            output_summary=f"{len(answer)} chars",
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        return {"answer": answer, "citations": _citations(chunks)}

    async def check_grounded(self, state: CragState) -> dict:
        t0 = time.monotonic()
        verdict = await self._reasoner.check_grounded(
            state.get("answer", ""), state.get("chunks", [])
        )
        await self._rec.record(
            "check_grounded",
            decision=f"grounded={verdict.grounded}",
            output_summary=verdict.reason,
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        out: dict = {"grounded": verdict.grounded}
        # Increment only when we're about to retry — bounds the regen loop.
        if not verdict.grounded:
            out["regen_count"] = 1
        return out


# ---------------------------------------------------------------------------
# Conditional edges — where the control flow (and the guardrails) live.
# ---------------------------------------------------------------------------


def decide_after_grade(state: CragState, max_loops: int):
    """relevant → generate ; else rewrite & retry until the loop cap, then abstain."""
    if state.get("relevant"):
        return "generate"
    if state.get("loops", 0) >= max_loops:   # guardrail: never loop forever (§6.4)
        return "abstain"
    return "transform_query"


def decide_after_grounded(state: CragState, max_regen: int):
    """grounded → end ; else regenerate ONCE (review Issue 4), then accept & end."""
    if state.get("grounded"):
        return "end"
    if state.get("regen_count", 0) > max_regen:  # bound the grounding retry
        return "end"
    return "generate"
