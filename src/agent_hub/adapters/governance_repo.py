"""Runtime governance store (§13.5) — one durable, hot-tunable config row.

Holds the live `GovernanceSettings` CRAG reads on every run. Shipped defaults come
from `Settings`; operator overrides are persisted as a single JSONB row in
`agent.agent_governance` and cached in memory so the read path never touches the
DB. Writes are ADMIN-only (enforced at the transport layer) and re-validated
through the pydantic model before they are stored, so a bad value can never reach
a graph.

`defaults()` and the stored row are merged on load: adding a new knob later means
existing rows transparently inherit its default rather than failing validation.
"""

from __future__ import annotations

import json

from psycopg_pool import AsyncConnectionPool

from agent_hub.config import Settings
from agent_hub.domain.models import GovernanceSettings
from agent_hub.observability.logging import get_logger

log = get_logger("governance")

_ROW_ID = 1  # single-row table — one governance profile per environment


def defaults_from_settings(s: Settings) -> GovernanceSettings:
    return GovernanceSettings(
        top_k=s.crag_top_k,
        min_score=s.crag_min_score,
        cross_encoder_enabled=s.cross_encoder_enabled,
        rerank_candidate_k=s.rerank_candidate_k,
        rerank_top_n=s.rerank_top_n,
        max_loops=s.crag_max_loops,
        max_regen=s.crag_max_regen,
        query_rewrite_enabled=s.crag_query_rewrite_enabled,
        grounding_enabled=s.crag_grounding_enabled,
        grade_temperature=s.crag_grade_temperature,
        rewrite_temperature=s.crag_rewrite_temperature,
        generate_temperature=s.crag_generate_temperature,
        grounding_temperature=s.crag_grounding_temperature,
    )


class GovernanceStore:
    def __init__(self, pool: AsyncConnectionPool, settings: Settings) -> None:
        self._pool = pool
        self._settings = settings
        self._defaults = defaults_from_settings(settings)
        self._current = self._defaults

    @property
    def defaults(self) -> GovernanceSettings:
        return self._defaults

    def current(self) -> GovernanceSettings:
        """Synchronous, in-memory read — safe to call on the hot path."""
        return self._current

    async def load(self) -> GovernanceSettings:
        """Hydrate the cache from Postgres at startup (defaults if no row/DB)."""
        try:
            async with self._pool.connection() as conn:
                row = await (
                    await conn.execute(
                        "SELECT data FROM agent_governance WHERE id = %s", (_ROW_ID,)
                    )
                ).fetchone()
            self._current = self._merge(row[0] if row else None)
        except Exception as exc:  # never block startup on a missing table
            log.warning("governance.load_failed", error=str(exc))
            self._current = self._defaults
        return self._current

    async def update(self, patch: dict, updated_by: str = "") -> GovernanceSettings:
        """Merge a partial patch over the current settings, validate, persist."""
        merged = {**self._current.model_dump(), **{k: v for k, v in patch.items() if v is not None}}
        validated = GovernanceSettings(**merged)  # raises on out-of-range → 422
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO agent_governance (id, data, updated_by, updated_at)
                VALUES (%s, %s::jsonb, NULLIF(%s,''), now())
                ON CONFLICT (id) DO UPDATE
                   SET data = EXCLUDED.data,
                       updated_by = EXCLUDED.updated_by,
                       updated_at = now()
                """,
                (_ROW_ID, json.dumps(validated.model_dump()), updated_by),
            )
        self._current = validated
        log.info("governance.updated", updated_by=updated_by, keys=list(patch.keys()))
        return validated

    async def reset(self) -> GovernanceSettings:
        async with self._pool.connection() as conn:
            await conn.execute("DELETE FROM agent_governance WHERE id = %s", (_ROW_ID,))
        self._current = self._defaults
        log.info("governance.reset")
        return self._current

    def _merge(self, stored: dict | None) -> GovernanceSettings:
        base = self._defaults.model_dump()
        if stored:
            base.update(stored)
        try:
            return GovernanceSettings(**base)
        except Exception as exc:  # stored row drifted from schema → defaults
            log.warning("governance.merge_invalid", error=str(exc))
            return self._defaults
