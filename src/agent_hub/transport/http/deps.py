"""Composition root — builds the adapter graph once at startup and holds it.

Wires concrete adapters (Ollama, know-hub gRPC, Postgres checkpointer, runs repo)
into the application services. Everything downstream depends on PORTS, so this is
the only module that knows the concrete types — swap an adapter here and nothing
else changes.

Lifecycle is owned by the FastAPI lifespan via an AsyncExitStack: the
checkpointer (an async context manager), the runs-repo pool, and the gRPC channel
are opened on startup and closed on shutdown.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass

from agent_hub.adapters.checkpointer import open_checkpointer
from agent_hub.adapters.events import EventProducer
from agent_hub.adapters.know_hub_grpc import KnowHubGrpcClient
from agent_hub.adapters.ollama import OllamaEmbedder, OllamaReasoner
from agent_hub.adapters.runs_repo import RunsRepository
from agent_hub.application.crag.service import CragService
from agent_hub.config import Settings
from agent_hub.observability.logging import get_logger

log = get_logger("deps")


@dataclass
class AppContext:
    settings: Settings
    grpc: KnowHubGrpcClient
    reasoner: OllamaReasoner
    embedder: OllamaEmbedder
    runs: RunsRepository
    events: EventProducer
    crag: CragService


async def build_context(settings: Settings, stack: AsyncExitStack) -> AppContext:
    grpc = KnowHubGrpcClient(host="know-hub", port=settings.know_hub_grpc_port)
    stack.push_async_callback(grpc.close)

    runs = RunsRepository(settings.database_url)
    await runs.start()
    stack.push_async_callback(runs.close)

    events = EventProducer(settings.kafka_brokers)
    await events.start()
    stack.push_async_callback(events.close)

    reasoner = OllamaReasoner(
        base_url=settings.ollama_base_url, model=settings.agent_reasoning_model
    )
    embedder = OllamaEmbedder(base_url=settings.ollama_base_url)

    # Checkpointer is an async CM kept open for the app lifetime (ADR-AGT-04).
    checkpointer = await stack.enter_async_context(open_checkpointer(settings.database_url))

    crag = CragService(
        retriever=grpc, reasoner=reasoner, runs_repo=runs, checkpointer=checkpointer, settings=settings
    )

    log.info("context.built", reasoning_model=settings.agent_reasoning_model)
    return AppContext(
        settings=settings, grpc=grpc, reasoner=reasoner, embedder=embedder,
        runs=runs, events=events, crag=crag,
    )
