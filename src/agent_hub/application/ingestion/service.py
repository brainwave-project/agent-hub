"""Ingestion service — runs the crew for one document and records its trajectory.

Driven by the Kafka consumer (§5.4.3). Builds the crew from injected adapters,
streams node updates, persists each transition to agent.agent_runs/agent_steps,
and returns the terminal status. know-hub performs all writes; this service only
reasons + delegates (Principle 5).
"""

from __future__ import annotations

from agent_hub.adapters.ports.embedding import Embedder
from agent_hub.adapters.ports.knowhub import DocumentGateway
from agent_hub.adapters.ports.reasoning import Classifier
from agent_hub.application.ingestion.agents.chunker import ChunkerAgent
from agent_hub.application.ingestion.agents.classifier import ClassifierAgent
from agent_hub.application.ingestion.agents.embedder import EmbedderAgent
from agent_hub.application.ingestion.agents.extractor import ExtractorAgent
from agent_hub.application.ingestion.graph import build_ingestion_graph
from agent_hub.observability.logging import get_logger

log = get_logger("ingestion-service")

_RECURSION_LIMIT = 12  # §7.7 — a routing loop can never run away


class IngestionService:
    def __init__(
        self,
        gateway: DocumentGateway,
        classifier: Classifier,
        embedder: Embedder,
        runs_repo,
    ) -> None:
        self._app = build_ingestion_graph(
            extractor=ExtractorAgent(gateway),
            classifier=ClassifierAgent(classifier),
            chunker=ChunkerAgent(),
            embedder=EmbedderAgent(embedder, gateway),
        )
        self._runs = runs_repo

    async def run(self, document_id: str, actor_user_id: str = "", force: bool = False) -> str:
        run_id = await self._runs.create_run("INGESTION", document_id, actor_user_id)
        recorder = self._runs.step_recorder(run_id)
        status = "FAILED"
        steps = 0
        try:
            final = None
            async for update in self._app.astream(
                {"document_id": document_id, "actor_user_id": actor_user_id, "force": force},
                config={"recursion_limit": _RECURSION_LIMIT},
                stream_mode="updates",
            ):
                # update = {node_name: partial_state} for the node that just ran.
                for node, partial in update.items():
                    steps += 1
                    hist = (partial or {}).get("history") or []
                    await recorder.record(
                        node, decision=hist[-1] if hist else "", output_summary=str(hist)[:200]
                    )
                    final = partial
            status = (final or {}).get("status", "SUCCEEDED") if final else "SUCCEEDED"
            # Normalize terminal: crew statuses → run_status enum.
            status = {"INDEXED": "SUCCEEDED", "SKIPPED": "SUCCEEDED", "FAILED": "FAILED"}.get(
                status, "SUCCEEDED"
            )
        except Exception as exc:
            log.error("ingestion.run_failed", document_id=document_id, error=str(exc))
        finally:
            await self._runs.finish_run(run_id, status, steps)
        log.info("ingestion.run_complete", document_id=document_id, status=status, steps=steps)
        return status
