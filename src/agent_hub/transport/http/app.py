"""FastAPI application — CRAG's HTTP entry point (TDD §5.4.3).

Routes:
  GET  /health/live, /health/ready
  GET  /api/agent/ping            — M1 hello-graph (proves LangGraph wiring)
  POST /api/agent/chat            — M5 CRAG, SSE stream of the reasoning trajectory
  GET  /api/agent/runs/{run_id}   — the durable trajectory of a past run

The ingestion crew does NOT enter here — it is driven by the Kafka consumer
started as a background task in the lifespan (one service, two domains, §5.4).

Identity: Kong validated the JWT and injected X-User-Id / X-User-Name /
X-User-Roles. agent-hub trusts them and forwards the caller's id + admin flag
into know-hub retrieval so CRAG only ever sees the caller's documents (Issue 1).
"""

from __future__ import annotations

import json
import uuid
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent_hub import __version__
from agent_hub.application.hello_graph import hello_app
from agent_hub.config import get_settings
from agent_hub.health.checks import router as health_router
from agent_hub.observability.logging import configure_logging, get_logger

log = get_logger()


def _is_admin(roles_header: str | None) -> bool:
    if not roles_header:
        return False
    try:
        return "ADMIN" in json.loads(roles_header)
    except (json.JSONDecodeError, TypeError):
        return "ADMIN" in roles_header


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    log.info("agent-hub.starting", version=__version__, reasoning_model=settings.agent_reasoning_model)

    async with AsyncExitStack() as stack:
        # Built lazily so M0/M1 reachability checks don't hard-require Postgres;
        # any failure is logged and the agent routes return 503 until deps recover.
        ctx = None
        consumer_task = None
        try:
            from agent_hub.transport.http.deps import build_context

            ctx = await build_context(settings, stack)
            app.state.ctx = ctx

            # Ingestion crew — background Kafka consumer (M6). Imported lazily so
            # the HTTP surface boots even if the consumer deps are unavailable.
            from agent_hub.transport.consumer.ingestion_consumer import start_ingestion_consumer

            consumer_task = await start_ingestion_consumer(ctx)
        except Exception as exc:  # degraded mode — health + hello still serve
            log.error("agent-hub.context_init_failed", error=str(exc))
            app.state.ctx = None

        try:
            yield
        finally:
            if consumer_task is not None:
                consumer_task.cancel()
            log.info("agent-hub.stopping")


def _require_ctx(request: Request):
    ctx = getattr(request.app.state, "ctx", None)
    if ctx is None:
        raise HTTPException(status_code=503, detail="agent-hub dependencies unavailable")
    return ctx


class ChatRequest(BaseModel):
    message: str
    sessionId: str | None = None


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Brainwave AgentHub",
        version=__version__,
        description="Agentic RAG & multi-agent ingestion (Phase 3). Reasoning only — never the system of record.",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.include_router(health_router)

    agent = APIRouter(prefix="/api/agent", tags=["agent"])

    @agent.get("/ping")
    async def ping(
        x_user_id: str | None = Header(default=None),
        x_user_name: str | None = Header(default=None),
    ) -> dict:
        # M1 — drive the hello-graph so /ping proves the LangGraph runtime, not
        # just the web server.
        out = await hello_app.ainvoke({"message": x_user_name or "world", "steps": []})
        return {
            "status": "ok",
            "service": settings.service_name,
            "version": __version__,
            "user_id": x_user_id,
            "graph": {"message": out["message"], "path": out["steps"]},
        }

    @agent.post("/chat")
    async def chat(
        body: ChatRequest,
        request: Request,
        x_user_id: str | None = Header(default=None),
        x_user_roles: str | None = Header(default=None),
        x_correlation_id: str | None = Header(default=None),
    ) -> StreamingResponse:
        ctx = _require_ctx(request)
        thread_id = body.sessionId or f"sess_{uuid.uuid4().hex[:12]}"
        stream = ctx.crag.stream(
            question=body.message,
            thread_id=thread_id,
            actor_user_id=x_user_id or "",
            is_admin=_is_admin(x_user_roles),
            correlation_id=x_correlation_id or "",
        )
        return StreamingResponse(
            stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",   # belt-and-suspenders alongside Kong
            },
        )

    @agent.get("/runs/{run_id}")
    async def get_run(run_id: str, request: Request) -> dict:
        ctx = _require_ctx(request)
        run = await ctx.runs.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return run

    @agent.get("/metrics")
    async def metrics(request: Request) -> dict:
        # M8 — trajectory metrics (correction/abstention/loop-depth/per-node latency, §12.4).
        ctx = _require_ctx(request)
        return await ctx.runs.metrics()

    app.include_router(agent)
    return app


app = create_app()
