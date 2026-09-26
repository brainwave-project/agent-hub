# Release gate snapshot

CI (`.github/workflows/eval-gate.yml`) runs the Eval Lab release gate on the three
files in this folder:

| File | What it is | Written by |
|---|---|---|
| `policy.json` | The gate policy: preconditions and rules | Release Gate → Export for CI |
| `baseline.json` | The promoted baseline run (full per-item scores) | Release Gate → Export for CI |
| `candidate.json` | The run of the change under review | Release Gate → Export for CI |

Workflow:

1. Change the pipeline (prompt, governance default, reranker, …).
2. In the Eval Lab, run the candidate on the frozen dataset's dev split.
3. Release Gate → Evaluate against the promoted baseline → **Export for CI**.
4. Commit this folder together with the change. The gate re-checks the dataset
   hash and the judge fingerprint, so a stale or edited snapshot is refused.

Run it locally: `cd infra && make evals-release-gate`, or

```bash
cd agent-hub
uv run python -m agent_hub.eval.lab.gate_cli \
  --policy src/agent_hub/eval/release/policy.json \
  --baseline src/agent_hub/eval/release/baseline.json \
  --candidate src/agent_hub/eval/release/candidate.json
```

Exit codes: `0` pass (or warnings) · `1` block · `2` the gate could not run.
