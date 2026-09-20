"""Eval Lab — the interactive half of the evaluation harness.

Serves the UI's Evaluation section (Golden Sets · Retrieval ·
Experiments · Stats Lab · Learning Path) as a standalone FastAPI app:

    python -m agent_hub.eval.lab            # :3005
    make evals-lab                          # from infra/, with the lean stack

It reuses the offline harness's instruments verbatim — `retrieval_metrics`,
`stats`, `arabic.normalize` — so a number in the UI and a number in
`make evals` come from the same code.

Like `eval-runner`, it talks to know-hub directly with synthetic headers and is
never routed through Kong. Each dataset's corpus is uploaded under its own
synthetic owner id, so know-hub's ownership predicate isolates one dataset's
index from every other document in the database.

Modules:
    models      datasets, golden items, experiments, runs (pydantic)
    store       versioned, hash-addressed file store; frozen versions are immutable
    splits      deterministic stratified, group-aware dev/holdout splitting
    chunking    preview of know-hub's chunker + evidence location
    scoring     turn a ranking into metrics, at document or evidence level
    readiness   Gate checks — "is this dataset fit to evaluate with?"
    runner      search know-hub, score a split, index a corpus
    analysis    experiment reports: CIs, paired deltas, decisions
    jobs        background job registry
    api         the HTTP surface
"""
