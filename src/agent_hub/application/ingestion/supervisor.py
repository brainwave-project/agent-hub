"""Supervisor — routes the crew to the next agent (§7.4 / ADR-AGT-07).

REFINEMENT OF §7.4 (principal call): the TDD's table proposes a *router LLM* that
picks the next agent each step. But Principle 3 (§2.3) and the §7.3 note are
explicit that only the Classifier should reason — burning an LLM call per routing
step on a near-linear pipeline is exactly the over-use those principles warn
against (latency + non-determinism for no benefit). So the supervisor routes
DETERMINISTICALLY by inspecting which state fields are filled. It is still
adaptive and bounded in the §7.4 sense:

  - skips Classifier when doc_type is already known (re-ingest / forced type)
  - short-circuits to END (SKIPPED) when the Extractor found no text layer
  - chooses only among the known agent set; it can never invent a step

If open-ended routing is ever needed, swap this function for an LLM router behind
the same signature — the graph topology does not change.
"""

from __future__ import annotations

from agent_hub.application.ingestion.state import IngestState


def route(state: IngestState) -> str:
    # Terminal: a status was set by the embedder (INDEXED/FAILED) or skip (SKIPPED).
    if state.get("status"):
        return "done"
    # 1. Nothing run yet → extract text.
    if not state.get("history"):
        return "extractor"
    # 2. No text layer → nothing to index. Terminal SKIPPED.
    if state.get("has_text") is False:
        return "skip"
    # 3. Classify unless already typed (adaptive skip on re-ingest / known type).
    if not state.get("doc_type"):
        return "classifier"
    # 4. Chunk once we have a strategy.
    if not state.get("chunks"):
        return "chunker"
    # 5. Embed + delegate the write to know-hub.
    return "embedder"
