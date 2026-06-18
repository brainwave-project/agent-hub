"""CRAG graph state (§6.3).

A TypedDict passed to every node; nodes return partial updates merged by
LangGraph. Two fields carry reducers (operator.add) so they ACCUMULATE across
the corrective cycle instead of being overwritten:

  - loops       : the corrective query-rewrite counter (the §6.4 guardrail).
  - regen_count : the grounding-retry counter — the bound the TDD's diagram
                  implied but its code omitted (review Issue 4).
"""

import operator
from typing import Annotated, TypedDict

from agent_hub.domain.models import Citation, RetrievedChunk


class CragState(TypedDict, total=False):
    # inputs
    question: str                       # original user question (+ condensed history)
    actor_user_id: str                  # the tenant boundary (Issue 1)
    is_admin: bool

    # working state
    query: str                          # current (possibly rewritten) search query
    chunks: list[RetrievedChunk]        # retrieved chunks for the current query
    relevant: bool                      # grader verdict
    grounded: bool                      # groundedness verdict
    loops: Annotated[int, operator.add]        # reducer: corrective retries
    regen_count: Annotated[int, operator.add]  # reducer: grounding regenerations

    # outputs
    answer: str
    citations: list[Citation]
    abstained: bool
