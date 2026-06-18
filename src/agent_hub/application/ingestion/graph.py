"""Ingestion crew graph (§7.2) — supervisor hub pattern.

           ┌────────────┐
  START ──▶│ supervisor │──route()──┬─ extractor ─┐
           └─────▲──────┘           ├─ classifier ┤
                 │                  ├─ chunker    │──▶ supervisor (loop)
                 └──────────────────┴─ embedder   ┘
                                    └─ skip ───────▶ supervisor ─ done ─▶ END

Every agent hands control back to the supervisor, which re-decides the next
agent from state (deterministic routing — see supervisor.py). Bounded by a
recursion limit so a routing bug can never run away (§7.7).
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent_hub.application.ingestion.state import IngestState
from agent_hub.application.ingestion.supervisor import route


async def _supervisor(state: IngestState) -> dict:
    return {}  # pure routing hub; the decision lives in the conditional edge


async def _skip(state: IngestState) -> dict:
    return {"status": "SKIPPED", "history": ["skip:no_text_layer"]}


def build_ingestion_graph(*, extractor, classifier, chunker, embedder):
    g = StateGraph(IngestState)

    g.add_node("supervisor", _supervisor)
    g.add_node("extractor", extractor)
    g.add_node("classifier", classifier)
    g.add_node("chunker", chunker)
    g.add_node("embedder", embedder)
    g.add_node("skip", _skip)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        route,
        {
            "extractor": "extractor",
            "classifier": "classifier",
            "chunker": "chunker",
            "embedder": "embedder",
            "skip": "skip",
            "done": END,
        },
    )
    for agent in ("extractor", "classifier", "chunker", "embedder", "skip"):
        g.add_edge(agent, "supervisor")

    return g.compile()
