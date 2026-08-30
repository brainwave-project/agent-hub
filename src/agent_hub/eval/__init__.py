"""Evaluation harness — retrieval metrics, Arabic normalisation, RAGAS, gating.

Entry point:  python -m agent_hub.eval.run_suite --suite all
Or from infra: make evals

Submodules that stand alone (no stack, no network):
    retrieval_metrics   hit rate, recall@k, precision@k, MRR, NDCG@k, MAP
    stats               bootstrap CIs, paired deltas, sample-size arithmetic
    arabic.normalize    per-transform Arabic orthographic normalisation
    datasets            corpus + versioned golden-set loading

Submodules that need the eval stack:
    seed                ingest the corpus into know-hub, wait for INDEXED
    retrieval           run golden sets through know-hub, score the ranking
    ragas_eval          drive the CRAG graph, judge answers with local Ollama
    report              console rendering + JSON report
    gate                regression gate against a promoted baseline
"""

import sys

from agent_hub.eval import retrieval_metrics, stats
from agent_hub.eval.datasets import GoldenRecord, load_corpus, load_golden

__all__ = [
    "GoldenRecord",
    "configure_stdout",
    "load_corpus",
    "load_golden",
    "retrieval_metrics",
    "stats",
]


def configure_stdout() -> None:
    """Force UTF-8 on stdout/stderr so reports survive a non-UTF-8 console.

    Necessary because the harness prints Arabic — golden queries, document
    titles, error text — and a Windows console defaults to cp1252, which raises
    UnicodeEncodeError mid-report and destroys the run. `errors="replace"` means
    a console that genuinely cannot render a glyph shows a placeholder instead
    of taking the whole eval down.

    Inside the eval-runner container this is a no-op; it matters when someone
    runs a module directly on the host.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
