"""Serve the Eval Lab API:  python -m agent_hub.eval.lab  (port EVAL_LAB_PORT, default 3005)."""

import os

import uvicorn

from agent_hub.eval import configure_stdout

if __name__ == "__main__":
    configure_stdout()
    # A factory rather than a module-level app, so importing api.py (tests, the
    # REPL) never creates the store directory as a side effect.
    uvicorn.run(
        "agent_hub.eval.lab.api:create_app",
        factory=True,
        host=os.environ.get("EVAL_LAB_HOST", "0.0.0.0"),
        port=int(os.environ.get("EVAL_LAB_PORT", "3005")),
        reload=os.environ.get("EVAL_LAB_RELOAD", "false").lower() == "true",
        reload_dirs=["src"],
    )
