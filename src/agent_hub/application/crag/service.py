"""CRAG service — orchestrates one user-facing run and streams its trajectory.

Ties the ports together per request: opens a run row, builds the graph with a
run-scoped step recorder + the Postgres checkpointer (thread_id = chat session)
+ the Langfuse handler, then streams LangGraph's node lifecycle as SSE
(§11.1 — we stream the agent's *trajectory*, not just tokens).

Streaming uses astream_events(v2): node starts → node_enter, the generate node's
model tokens → token, grade/check_grounded outputs → decision. Citations + the
terminal done are emitted from the final state.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from agent_hub.adapters.ports.knowhub import Retriever
from agent_hub.adapters.ports.reasoning import CragReasoner
from agent_hub.application.crag.graph import build_crag_graph
from agent_hub.application.crag.nodes import CragNodes
from agent_hub.config import Settings
from agent_hub.observability.langfuse_cb import make_langfuse_handler
from agent_hub.observability.logging import get_logger
from agent_hub.transport.http.sse import sse

log = get_logger("crag-service")

_KNOWN_NODES = {"retrieve", "grade", "transform_query", "generate", "check_grounded", "abstain"}


class CragService:
    def __init__(
        self,
        retriever: Retriever,
        reasoner: CragReasoner,
        runs_repo,          # RunsRepository
        checkpointer,       # AsyncPostgresSaver | None
        settings: Settings,
    ) -> None:
        self._retriever = retriever
        self._reasoner = reasoner
        self._runs = runs_repo
        self._checkpointer = checkpointer
        self._s = settings

    def _build(self, run_id: str):
        recorder = self._runs.step_recorder(run_id)
        nodes = CragNodes(
            self._retriever,
            self._reasoner,
            recorder,
            top_k=self._s.crag_top_k,
            min_score=self._s.crag_min_score,
        )
        return build_crag_graph(
            nodes,
            max_loops=self._s.crag_max_loops,
            max_regen=self._s.crag_max_regen,
            checkpointer=self._checkpointer,
        )

    async def stream(
        self,
        *,
        question: str,
        thread_id: str,
        actor_user_id: str,
        is_admin: bool,
        correlation_id: str = "",
    ) -> AsyncIterator[str]:
        run_id = await self._runs.create_run("CRAG", thread_id, actor_user_id, correlation_id)
        app = self._build(run_id)

        handler = make_langfuse_handler(
            self._s,
            trace_name="crag-run",
            metadata={"thread_id": thread_id, "correlation_id": correlation_id, "run_id": run_id},
        )
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        if handler is not None:
            config["callbacks"] = [handler]

        seq = 0
        final_state: dict[str, Any] = {}
        status = "FAILED"
        try:
            async for ev in app.astream_events(
                {
                    "question": question,
                    "actor_user_id": actor_user_id,
                    "is_admin": is_admin,
                },
                config=config,
                version="v2",
            ):
                kind = ev.get("event")
                node = ev.get("metadata", {}).get("langgraph_node")

                if kind == "on_chain_start" and node in _KNOWN_NODES:
                    seq += 1
                    yield sse("node_enter", {"node": node, "seq": seq})

                elif kind == "on_chat_model_stream" and node == "generate":
                    chunk = ev.get("data", {}).get("chunk")
                    text = getattr(chunk, "content", "") if chunk is not None else ""
                    if text:
                        yield sse("token", {"text": text})

                elif kind == "on_chain_end" and node in {"grade", "check_grounded"}:
                    out = ev.get("data", {}).get("output") or {}
                    if "relevant" in out:
                        choice = "generate" if out["relevant"] else "transform_query/abstain"
                        yield sse("decision", {"node": "grade", "choice": choice})
                    if "grounded" in out:
                        yield sse(
                            "decision",
                            {"node": "check_grounded", "choice": "end" if out["grounded"] else "regenerate"},
                        )

                # Capture the graph's final state from the root chain end.
                if kind == "on_chain_end" and ev.get("name") == "LangGraph":
                    final_state = ev.get("data", {}).get("output") or {}

            for cit in final_state.get("citations", []) or []:
                doc_id = getattr(cit, "document_id", None) or cit.get("document_id")
                score = getattr(cit, "score", None) or cit.get("score", 0)
                yield sse("citation", {"docId": doc_id, "score": score})

            status = "ABSTAINED" if final_state.get("abstained") else "SUCCEEDED"
        except Exception as exc:
            log.error("crag.stream_failed", error=str(exc))
            yield sse("error", {"message": "agent run failed"})
        finally:
            loops = int(final_state.get("loops", 0) or 0)
            await self._runs.finish_run(run_id, status, loops)
            yield sse("done", {"runId": run_id, "status": status, "loops": loops})
