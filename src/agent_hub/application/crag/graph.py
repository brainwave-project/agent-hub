"""CRAG graph assembly (§6.4) — the heart of the single agent.

  retrieve → grade →┬─ relevant ───────────────► generate → check_grounded →┬─ grounded ─► END
                    ├─ weak & loops<MAX ─► transform_query ─► retrieve (cycle) │
                    └─ weak & loops>=MAX ─► abstain ─► END        regenerate ◄──┘ (bounded)

M2 is just this graph with grading that always passes (straight line); M3 adds
the transform_query cycle, the MAX_LOOPS guardrail, and abstain. We build the
full graph once — the milestones differ only in which reasoner is injected.

The compiled app is checkpointer-agnostic: pass a PostgresSaver (M4) or nothing
(M1–M3 in-memory). Nodes and edges are identical either way.
"""

from functools import partial

from langgraph.graph import END, START, StateGraph

from agent_hub.application.crag.nodes import (
    CragNodes,
    decide_after_grade,
    decide_after_grounded,
)
from agent_hub.application.crag.state import CragState


def build_crag_graph(
    nodes: CragNodes,
    *,
    max_loops: int = 2,
    max_regen: int = 1,
    query_rewrite_enabled: bool = True,
    grounding_enabled: bool = True,
    checkpointer=None,
):
    """Assemble the CRAG graph. The two corrective phases are governable:

    * query_rewrite_enabled=False drops the transform_query cycle — a weak grade
      abstains immediately.
    * grounding_enabled=False drops the check_grounded node — generate goes
      straight to END (no post-hoc verification / regeneration).
    """
    g = StateGraph(CragState)

    g.add_node("retrieve", nodes.retrieve)
    g.add_node("grade", nodes.grade)
    g.add_node("generate", nodes.generate)
    g.add_node("abstain", nodes.abstain)
    if query_rewrite_enabled:
        g.add_node("transform_query", nodes.transform_query)
    if grounding_enabled:
        g.add_node("check_grounded", nodes.check_grounded)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")

    grade_targets = {"generate": "generate", "abstain": "abstain"}
    if query_rewrite_enabled:
        grade_targets["transform_query"] = "transform_query"
    g.add_conditional_edges(
        "grade",
        partial(decide_after_grade, max_loops=max_loops, rewrite_enabled=query_rewrite_enabled),
        grade_targets,
    )
    if query_rewrite_enabled:
        g.add_edge("transform_query", "retrieve")      # ◀── the corrective cycle

    if grounding_enabled:
        g.add_edge("generate", "check_grounded")
        g.add_conditional_edges(
            "check_grounded",
            partial(decide_after_grounded, max_regen=max_regen),
            {"end": END, "generate": "generate"},       # bounded grounding retry
        )
    else:
        g.add_edge("generate", END)

    g.add_edge("abstain", END)

    return g.compile(checkpointer=checkpointer)
