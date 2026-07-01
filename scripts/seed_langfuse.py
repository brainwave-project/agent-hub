"""Seed Langfuse with the platform's prompts (§13.5).

Pushes every prompt in ``DEFAULT_PROMPTS`` into the configured Langfuse project
under its canonical key (e.g. ``crag/grade``) and publishes it with the
``production`` label, so the agent-hub prompt registry resolves managed versions
on the next run. Idempotent: re-running creates a new version only when the text
actually changed (Langfuse dedupes identical bodies).

Usage (from agent-hub/):
    LANGFUSE_ENABLED=true \
    LANGFUSE_HOST=http://localhost:3060 \
    LANGFUSE_PUBLIC_KEY=pk-... LANGFUSE_SECRET_KEY=sk-... \
    uv run python scripts/seed_langfuse.py

Reads the same Settings as the service, so in Docker it just inherits the
agent-hub env. Exits non-zero (with a clear message) if keys are missing.
"""

from __future__ import annotations

import sys

from agent_hub.adapters.prompts import DEFAULT_PROMPTS, PROMPT_DESCRIPTIONS
from agent_hub.config import get_settings


def main() -> int:
    s = get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        print(
            "✗ LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set.\n"
            "  Create a project in the Langfuse UI, copy the keys, and re-run.",
            file=sys.stderr,
        )
        return 1

    try:
        from langfuse import Langfuse
    except ImportError:
        print("✗ langfuse package not installed (run `uv sync`).", file=sys.stderr)
        return 1

    client = Langfuse(
        public_key=s.langfuse_public_key,
        secret_key=s.langfuse_secret_key,
        host=s.langfuse_host,
    )

    print(f"→ Seeding {len(DEFAULT_PROMPTS)} prompts to {s.langfuse_host} (label={s.prompt_label})")
    for key, text in DEFAULT_PROMPTS.items():
        client.create_prompt(
            name=key,
            prompt=text,
            type="text",
            labels=[s.prompt_label],
            config={"description": PROMPT_DESCRIPTIONS.get(key, "")},
        )
        print(f"  ✓ {key}")

    client.flush()
    print("✓ Done. Set LANGFUSE_ENABLED=true on agent-hub to use managed prompts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
