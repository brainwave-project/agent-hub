"""Langfuse callback for nested agent traces (§12).

A RAG trace is flat (retrieve→generate); an agent trace is a tree — run → node →
LLM call, with corrective loops appearing as repeated subtrees. Attaching
Langfuse's LangChain callback to the compiled graph produces that tree
automatically. We tag thread_id / loops / correlation_id so traces join logs and
the agent_runs table.

Best-effort and air-gapped (self-hosted Langfuse, §2.2): if Langfuse is disabled
or the package import fails, return None and the graph runs untraced — tracing
never blocks reasoning.
"""

from __future__ import annotations

from typing import Any

from agent_hub.config import Settings
from agent_hub.observability.logging import get_logger

log = get_logger("langfuse")


def make_langfuse_handler(settings: Settings, *, trace_name: str, metadata: dict[str, Any]):
    if not settings.langfuse_enabled:
        return None
    try:
        from langfuse.callback import CallbackHandler

        return CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            trace_name=trace_name,
            metadata=metadata,
        )
    except Exception as exc:  # never block the run on observability
        log.warning("langfuse.init_failed", error=str(exc))
        return None
