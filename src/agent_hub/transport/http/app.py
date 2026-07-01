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
from pydantic import BaseModel, ValidationError

from agent_hub import __version__
from agent_hub.application.hello_graph import hello_app
from agent_hub.config import get_settings
from agent_hub.domain.models import GovernanceSettings
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


class GovernancePatch(BaseModel):
    """Partial update — every field optional so the UI can PATCH one knob at a
    time. Unknown fields are ignored; ranges are enforced by GovernanceSettings."""

    model_config = {"extra": "ignore"}

    top_k: int | None = None
    min_score: float | None = None
    cross_encoder_enabled: bool | None = None
    rerank_candidate_k: int | None = None
    rerank_top_n: int | None = None
    max_loops: int | None = None
    max_regen: int | None = None
    query_rewrite_enabled: bool | None = None
    grounding_enabled: bool | None = None
    grade_temperature: float | None = None
    rewrite_temperature: float | None = None
    generate_temperature: float | None = None
    grounding_temperature: float | None = None


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

    # -- Governance (§13.5) — runtime-tunable RAG/CRAG knobs -----------------

    @agent.get("/governance")
    async def get_governance(request: Request) -> dict:
        """Live settings + shipped defaults + a JSON-schema of bounds so the UI can
        render each control (ranges, types) without duplicating the contract."""
        ctx = _require_ctx(request)
        gov = ctx.governance
        return {
            "settings": gov.current().model_dump(),
            "defaults": gov.defaults.model_dump(),
            "schema": GovernanceSettings.model_json_schema(),
        }

    @agent.put("/governance")
    async def put_governance(
        patch: GovernancePatch,
        request: Request,
        x_user_id: str | None = Header(default=None),
        x_user_roles: str | None = Header(default=None),
    ) -> dict:
        ctx = _require_ctx(request)
        if not _is_admin(x_user_roles):
            raise HTTPException(status_code=403, detail="ADMIN role required")
        try:
            updated = await ctx.governance.update(
                patch.model_dump(exclude_none=True), updated_by=x_user_id or ""
            )
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        return {"settings": updated.model_dump()}

    @agent.post("/governance/reset")
    async def reset_governance(
        request: Request,
        x_user_roles: str | None = Header(default=None),
    ) -> dict:
        ctx = _require_ctx(request)
        if not _is_admin(x_user_roles):
            raise HTTPException(status_code=403, detail="ADMIN role required")
        reset = await ctx.governance.reset()
        return {"settings": reset.model_dump()}

    # -- Model catalog — every LLM/model the platform runs ------------------

    @agent.get("/models")
    async def models(request: Request) -> dict:
        """Catalog of the LLMs/models in use, powering the Observability → LLMs
        section. Reflects live config (reasoning/embedding/rerank tags) rather than
        a hardcoded UI list."""
        ctx = _require_ctx(request)
        s = ctx.settings
        gov = ctx.governance.current()
        return {
            "models": [
                {
                    "id": "reasoning",
                    "name": s.agent_reasoning_model,
                    "provider": "Ollama",
                    "role": "Reasoning · LLM",
                    "purpose": "CRAG grading, rewrite, generation, grounding, classification.",
                    "host": s.ollama_base_url,
                    "params": "8B · Q4_K_M",
                    "modality": "text → text",
                    "enabled": True,
                    "service": "agent-hub",
                },
                {
                    "id": "embedding",
                    "name": "nomic-embed-text",
                    "provider": "Ollama",
                    "role": "Embedding",
                    "purpose": "768-dim document + query embeddings for pgvector semantic search.",
                    "host": s.ollama_base_url,
                    "params": "768-dim",
                    "modality": "text → vector",
                    "enabled": True,
                    "service": "know-hub · agent-hub",
                },
                {
                    "id": "cross-encoder",
                    "name": s.rerank_model,
                    "provider": "sentence-transformers",
                    "role": "Cross-Encoder · Reranker",
                    "purpose": "Stage-2 joint (query, chunk) re-scoring for retrieval precision.",
                    "host": "in-process",
                    "params": "MiniLM-L6",
                    "modality": "(text, text) → score",
                    "enabled": gov.cross_encoder_enabled,
                    "service": "agent-hub",
                },
            ]
        }

    app.include_router(agent)
    return app


app = create_app()
