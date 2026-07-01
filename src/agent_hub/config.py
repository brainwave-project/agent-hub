"""Typed configuration for agent-hub.

Mirrors the TS hubs' env contract (literal per-service values in
infra/env/services/agent-hub.dev.env, secrets/interpolation via compose
`environment:`). pydantic-settings validates + coerces at startup so a
misconfigured container fails fast and loud rather than at first request.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    # --- service identity ---
    service_name: str = "agent-hub"
    env: str = "development"
    log_level: str = "info"
    port: int = 3004

    # --- reasoning + embedding model host (local Ollama, air-gapped) ---
    ollama_base_url: str = "http://ollama:11434"
    # Tool-calling model reused from Phase 2 (real pulled tag, not the TDD's
    # shorthand `llama3.1:8b` — see review "Model tag drift").
    agent_reasoning_model: str = "llama3.1:8b-instruct-q4_K_M"

    # --- know-hub (system of record) ---
    # CRAG retrieval reuses the EXISTING HTTP /api/semantic path (review Issue 2:
    # gRPC deferred until the ingestion write path in M6 proves it necessary).
    know_hub_url: str = "http://know-hub:3002"
    know_hub_grpc_port: int = 50052  # reserved for M6+ PersistChunks

    # --- checkpointer + agent schema (wired in M4) ---
    database_url: str = "postgresql://agent_role:agent_dev_password@postgres:5432/brainwave"

    # --- events (wired in M6) ---
    kafka_brokers: str = "kafka:9092"

    # --- CRAG guardrails + retrieval ---
    # These are the SHIPPED DEFAULTS. At runtime the Governance store (§13.5)
    # overlays operator overrides persisted in agent.agent_governance, so the
    # live values can be tuned from the UI without a redeploy.
    crag_max_loops: int = 2          # corrective query-rewrite cap (§6.4)
    crag_max_regen: int = 1          # grounding-retry cap (review Issue 4)
    crag_top_k: int = 6              # chunks fetched per retrieve
    crag_min_score: float = 0.0      # 0 → know-hub applies its own SIMILARITY_THRESHOLD

    # Per-node temperatures (§4.4 principal's note): judgement nodes deterministic
    # (0), generation warmer for fluency. Governable so operators can trade
    # determinism ↔ creativity per node without touching code.
    crag_grade_temperature: float = 0.0
    crag_rewrite_temperature: float = 0.0
    crag_generate_temperature: float = 0.2
    crag_grounding_temperature: float = 0.0

    # Graph-shape toggles — turn a whole corrective phase off without a redeploy.
    crag_query_rewrite_enabled: bool = True
    crag_grounding_enabled: bool = True

    # --- Cross-Encoder reranker (two-stage retrieval) ---
    # Stage 1 (bi-encoder / pgvector) fetches a wide candidate pool; stage 2 (this
    # cross-encoder) re-scores each (query, chunk) pair jointly and keeps the best
    # rerank_top_n. Off by default: enabling it requires the optional `rerank`
    # extra (sentence-transformers + torch). Fail-open — a missing model never
    # breaks a run, it just skips the rerank stage.
    cross_encoder_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidate_k: int = 20     # stage-1 pool size when reranking is on
    rerank_top_n: int = 4            # chunks kept after cross-encoder re-scoring

    # --- prompt management (Langfuse) ---
    # Prompts are authored/versioned in Langfuse and fetched by name; the bundled
    # DEFAULT_PROMPTS are only a fallback when Langfuse is disabled/unreachable.
    prompt_label: str = "production"

    # --- observability (wired in M4) ---
    langfuse_enabled: bool = False
    langfuse_host: str = "http://langfuse:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""


@lru_cache
def get_settings() -> Settings:
    """Process-wide singleton; cached so every module shares one instance."""
    return Settings()
