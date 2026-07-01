"""Health endpoints — identical contract to the TS hubs (/health/live,
/health/ready) so the platform's probes and dashboards stay uniform.

Liveness  : the process is up and the event loop responds. Always 200.
Readiness : downstream dependencies the service needs to do real work.

At M0 the service has no hard runtime dependencies wired yet, so readiness
performs a best-effort Ollama probe and reports it as a component status
WITHOUT failing the gate — the goal of M0 is to prove the empty service is
reachable through Kong. Later milestones (M4 Postgres, M6 Kafka) tighten this
into a real gate.
"""

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from agent_hub.config import get_settings

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def liveness() -> dict:
    return {"status": "ok", "service": get_settings().service_name}


@router.get("/ready")
async def readiness() -> JSONResponse:
    settings = get_settings()
    components: dict[str, str] = {}

    # Best-effort Ollama probe — reported, not gated, at M0.
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            components["ollama"] = "up" if resp.status_code == 200 else "degraded"
    except Exception:
        components["ollama"] = "unreachable"

    return JSONResponse(
        status_code=200,
        content={"status": "ready", "service": settings.service_name, "components": components},
    )
