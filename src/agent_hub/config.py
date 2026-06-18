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
    crag_max_loops: int = 2          # corrective query-rewrite cap (§6.4)
    crag_max_regen: int = 1          # grounding-retry cap (review Issue 4)
    crag_top_k: int = 6              # chunks fetched per retrieve
    crag_min_score: float = 0.0      # 0 → know-hub applies its own SIMILARITY_THRESHOLD

    # --- observability (wired in M4) ---
    langfuse_enabled: bool = False
    langfuse_host: str = "http://langfuse:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""


@lru_cache
def get_settings() -> Settings:
    """Process-wide singleton; cached so every module shares one instance."""
    return Settings()
