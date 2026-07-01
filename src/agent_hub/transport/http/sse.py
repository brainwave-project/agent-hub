"""SSE framing for the agent's reasoning trajectory (§11.2).

Event protocol:
  node_enter {node, seq}        — a graph node started (drives the live stepper)
  decision   {node, choice}     — a conditional edge fired
  token      {text}             — a generation token (generate node only)
  citation   {docId, score}     — a source chunk used in the answer
  done       {runId, status, loops}
  error      {message}
"""

import json
from typing import Any


def sse(event: str, data: dict[str, Any]) -> str:
    """One SSE frame. `event:` lets the browser dispatch by type."""
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"
