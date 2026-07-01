# =============================================================================
# agent-hub — production image (GHCR). Dev uses python:3.12-slim with a source
# mount + uv for hot reload (see infra/docker/compose.dev.yml), mirroring how
# the TS hubs run node:20-alpine in dev but ship a built image in prod.
# =============================================================================
FROM python:3.12-slim AS base

# uv: fast, reproducible resolver/installer (the project's Python toolchain).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# --- dependency layer (cached unless pyproject/lock changes) ---
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache .

# --- cross-encoder reranker deps (mirrors the optional `rerank` extra) ---
# Installed as a separate, stable layer so the heavy torch wheel caches
# independently of app code. Pinned to the PyTorch CPU index so we get the small
# CPU wheel, never the multi-GB CUDA build the default index resolves on Linux.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache --index-url https://download.pytorch.org/whl/cpu torch \
    && uv pip install --system --no-cache "sentence-transformers>=3.3"

# --- application layer ---
COPY src ./src
COPY proto ./proto
COPY scripts ./scripts
RUN uv pip install --system --no-cache -e . \
    && python scripts/gen_proto.py   # generate knowhub gRPC stubs from proto/

EXPOSE 3004

# Non-root runtime.
RUN useradd --create-home --uid 10001 appuser
USER appuser

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:3004/health/live').status==200 else 1)"

CMD ["python", "-m", "agent_hub.main"]
