"""Classifier — the crew's ONLY reasoning agent (§7.3).

Infers {doc_type, topic, suggested_strategy} from a text sample. This single
judgement is what makes chunking adaptive — everything downstream is a
deterministic tool acting on this decision (Principle 3).
"""

from __future__ import annotations

from agent_hub.adapters.ports.reasoning import Classifier
from agent_hub.application.ingestion.state import IngestState

_VALID_STRATEGIES = {"clause", "slide", "turn", "fixed"}


class ClassifierAgent:
    def __init__(self, classifier: Classifier) -> None:
        self._c = classifier

    async def __call__(self, state: IngestState) -> dict:
        # A leading sample is enough to type a document and keeps the prompt small.
        sample = (state.get("raw_text", "") or "")[:4000]
        verdict = await self._c.classify(sample)
        strategy = verdict.suggested_strategy if verdict.suggested_strategy in _VALID_STRATEGIES else "fixed"
        return {
            "doc_type": verdict.doc_type or "other",
            "topic": verdict.topic or "",
            "strategy": strategy,
            "history": [f"classifier:{verdict.doc_type}/{strategy}"],
        }
