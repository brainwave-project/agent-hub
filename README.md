# agent-hub

Brainwave's first **reasoning service** (Phase 3) — Python · LangGraph · local Ollama.

A stateless orchestrator that *reasons* but never writes: it grades retrievals,
classifies documents, and decides the next step, while **know-hub remains the
system of record** (Principle 5 / ADR-AGT-06). Everything runs air-gapped on the
existing Ollama/Postgres/Kafka infrastructure — zero external calls, zero cost.

Two domains share one process (TDD §5.4):

| Domain | Entry | Shape |
|--------|-------|-------|
| **CRAG** (Corrective RAG) | HTTP `POST /api/agent/chat` (SSE) | user-facing, latency-sensitive |
| **Ingestion crew** | Kafka `DOCUMENT_UPLOADED` | background, throughput-oriented |

They share only `adapters/`; neither imports from the other (`application/crag` ⊥ `application/ingestion`).

## Layout (hexagonal — mirrors the TS hubs)

```
src/agent_hub/
├── domain/         # pure types: CragState, Grade, IngestState (no deps)
├── application/
│   ├── crag/       # §6 single-agent corrective RAG
│   └── ingestion/  # §7 multi-agent crew (supervisor + extractor/classifier/chunker/embedder)
├── adapters/       # know-hub client, Ollama factory, checkpointer, Kafka  (+ ports/)
├── transport/
│   ├── http/       # FastAPI: /api/agent/chat (SSE), /api/agent/runs/:id
│   └── consumer/   # Kafka consumer driving the ingestion crew
├── observability/  # structlog + (M4) Langfuse callback handler
└── health/         # /health/live, /health/ready
```

## Develop

```bash
uv sync                                            # create .venv + install deps
uv run python -m agent_hub.main                    # serve on :3004
uv run uvicorn agent_hub.transport.http.app:app --reload --port 3004
uv run ruff check .
```

In the stack it runs behind Kong with the other hubs:

```bash
cd infra && make dev-services     # brings up agent-hub alongside auth/user/know/audit
make logs s=agent-hub
```

Reachability check through the gateway (M0):

```bash
curl localhost:3004/health/live          # direct
curl localhost:8000/api/agent/ping       # via Kong (with a Bearer token)
```

## Milestones (TDD §15)

`M0` scaffold ✓ · `M1` hello-graph · `M2` CRAG no-loop · `M3` corrective loop ·
`M4` checkpoint+trace · `M5` stream trajectory · `M6` ingestion crew · `M7` supervisor · `M8` eval
