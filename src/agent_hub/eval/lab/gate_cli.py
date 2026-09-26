"""Release gate on the command line — the same evaluation the UI runs, with an exit code for CI.

    # committed snapshot (what CI runs — no stack, no LLM, no network)
    python -m agent_hub.eval.lab.gate_cli \\
        --policy    src/agent_hub/eval/release/policy.json \\
        --baseline  src/agent_hub/eval/release/baseline.json \\
        --candidate src/agent_hub/eval/release/candidate.json \\
        --summary   "$GITHUB_STEP_SUMMARY"

    # straight from the lab store, by run id (inside the eval-api container)
    python -m agent_hub.eval.lab.gate_cli --policy-id kestrel-release \\
        --baseline-id gen_... --candidate-id gen_...

Exit codes: 0 = pass (or pass with warnings), 1 = block, 2 = the gate could not
run (missing file, invalid JSON). CI must treat 2 as a failure too: a gate that
silently skips when its input is missing protects nothing.

Why the snapshot, not a live run in CI: a generation run needs know-hub, an
index and an 8B model. The expensive half runs where the GPU is (a developer
machine or a self-hosted runner) and exports the run; the cheap, deterministic
half — the policy decision — runs on every pull request. The candidate file's
dataset hash and judge fingerprint are checked, so an edited or stale snapshot
is refused, not trusted.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

from agent_hub.eval import configure_stdout
from agent_hub.eval.lab.genmodels import GatePolicy
from agent_hub.eval.lab.release_gate import evaluate_gate, render_markdown


def _load(path: str) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def _from_store(lab_dir: pathlib.Path, run_id: str) -> dict:
    for sub in ("gen_runs", "runs"):
        path = lab_dir / sub / f"{run_id}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"run {run_id} not found under {lab_dir}")


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description="Eval Lab release gate")
    parser.add_argument("--policy", help="policy JSON file")
    parser.add_argument("--policy-id", help="policy id in the lab store")
    parser.add_argument("--baseline", help="baseline run JSON file")
    parser.add_argument("--candidate", help="candidate run JSON file")
    parser.add_argument("--baseline-id")
    parser.add_argument("--candidate-id")
    parser.add_argument("--lab-dir", default=os.environ.get("EVAL_LAB_DIR"))
    parser.add_argument(
        "--summary", help="append a Markdown summary here (e.g. $GITHUB_STEP_SUMMARY)"
    )
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args(argv)

    try:
        lab_dir = (
            pathlib.Path(args.lab_dir)
            if args.lab_dir
            else (pathlib.Path(__file__).resolve().parent.parent / "lab_store")
        )
        if args.policy:
            policy = GatePolicy.model_validate(_load(args.policy))
        elif args.policy_id:
            policy = GatePolicy.model_validate(
                _load(str(lab_dir / "gate_policies" / f"{args.policy_id}.json"))
            )
        else:
            parser.error("give --policy or --policy-id")
        baseline = (
            _load(args.baseline) if args.baseline else _from_store(lab_dir, args.baseline_id or "")
        )
        candidate = (
            _load(args.candidate)
            if args.candidate
            else _from_store(lab_dir, args.candidate_id or "")
        )
    except (OSError, ValueError) as exc:
        print(f"gate could not run: {exc}", file=sys.stderr)
        return 2

    result = evaluate_gate(policy, baseline, candidate)
    markdown = render_markdown(result)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as handle:
            handle.write(markdown)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(markdown)
        for reason in result["reasons"]:
            print(f"  - {reason}")
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
