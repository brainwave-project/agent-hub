"""LangGraph Postgres checkpointer (ADR-AGT-04 / §6.5).

Persists graph state after every node, keyed by thread_id, so a crashed run
resumes, you can time-travel during debugging, and human-in-the-loop becomes
possible later — all on the Postgres we already run.

The checkpoint tables live in the `agent` schema: we pin search_path=agent on
the connection so LangGraph's `.setup()` creates them there (not public),
matching the per-service schema convention. agent_role owns the schema, so it
has CREATE (see infra/postgres/init/40-agent.sql).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from agent_hub.observability.logging import get_logger

log = get_logger("checkpointer")


def _with_agent_search_path(database_url: str) -> str:
    sep = "&" if "?" in database_url else "?"
    # psycopg passes `options` straight to the server: pin the schema.
    return f"{database_url}{sep}options=-csearch_path%3Dagent"


@asynccontextmanager
async def open_checkpointer(database_url: str):
    """Yield a set-up AsyncPostgresSaver. Use as `async with open_checkpointer(...)`."""
    conn_str = _with_agent_search_path(database_url)
    async with AsyncPostgresSaver.from_conn_string(conn_str) as saver:
        await saver.setup()  # idempotent — creates checkpoint tables if absent
        log.info("checkpointer.ready", schema="agent")
        yield saver
