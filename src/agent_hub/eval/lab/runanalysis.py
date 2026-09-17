"""Run analysis — sweeps and failure analysis, computed offline from a stored run.

A run stores every ranking it produced, so questions like "what if k were 3?"
or "what if we had thresholded at 0.65?" need no new retrieval: re-score the
stored rankings. That makes them cheap enough to explore, and exact — the
rankings are the same ones the run scored.

    k_sweep           every metric at k = 1..K. Recall climbs, precision falls,
                      hit rate saturates: the choice of k is a decision about
                      what the generator consumes, not a free parameter.
    threshold_sweep   metrics after dropping results below a similarity floor.
                      Recreates the CRAG_MIN_SCORE trade-off: too low keeps noise,
                      too high starves retrieval.
    failure_analysis  the queries that failed, with suggested causes. A mean
                      tells you THAT a system fails; error analysis tells you
                      WHY, which is the only thing you can act on.
"""

from __future__ import annotations

import re
import statistics

from agent_hub.eval.lab.chunking import chunk_text, locate_evidence, norm_text
from agent_hub.eval.lab.models import Run
from agent_hub.eval.lab.store import Bundle
from agent_hub.eval.retrieval_metrics import (
    average_precision,
    graded_ndcg_at_k,
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

__all__ = ["TAXONOMY", "k_sweep", "threshold_sweep", "failure_analysis"]

#: Failure categories. `auto` ones can be suggested from data; the rest need a human.
TAXONOMY: dict[str, dict] = {
    "no_results": {
        "auto": True,
        "label": "No results",
        "hint": "Nothing came back — threshold too strict or query empty after normalisation.",
    },
    "not_retrieved": {
        "auto": True,
        "label": "Not retrieved",
        "hint": "A relevant document is absent from the whole fetched list: a recall failure, not a ranking one.",
    },
    "ranked_too_low": {
        "auto": True,
        "label": "Ranked too low",
        "hint": "Retrieved, but below k — a reranker's job.",
    },
    "distractor_outranked": {
        "auto": True,
        "label": "Distractor outranked",
        "hint": "The top result was judged not relevant: a topically close document won.",
    },
    "label_gap": {
        "auto": True,
        "label": "Possible label gap",
        "hint": "The top result was never judged — it may be relevant. Judge it before blaming the retriever.",
    },
    "vocabulary_mismatch": {
        "auto": True,
        "label": "Vocabulary mismatch",
        "hint": "The query shares few content words with the answer document.",
    },
    "chunk_boundary": {
        "auto": True,
        "label": "Chunk boundary",
        "hint": "The evidence straddles a chunk boundary, so no chunk contains it.",
    },
    "orthography_trap": {
        "auto": True,
        "label": "Orthography trap",
        "hint": "The item is a seeded normalisation trap.",
    },
    "partial_multi_doc": {
        "auto": True,
        "label": "Partial multi-document",
        "hint": "Several documents are needed; only some were found.",
    },
    "duplicate_evidence": {
        "auto": True,
        "label": "Duplicate evidence",
        "hint": "Chunks repeated already-credited evidence instead of new evidence.",
    },
    "ambiguous_query": {
        "auto": False,
        "label": "Ambiguous query",
        "hint": "The question admits several readings.",
    },
    "label_error": {
        "auto": False,
        "label": "Label error",
        "hint": "The golden label itself is wrong.",
    },
    "other": {"auto": False, "label": "Other", "hint": "Write a note."},
}

_WORD = re.compile(r"\w+", re.UNICODE)
DEFAULT_THRESHOLDS = [round(i * 0.05, 2) for i in range(20)]


def _usable(run: Run):
    return [r for r in run.results if r.error is None and r.relevant_units]


def k_sweep(run: Run, max_k: int = 20) -> dict:
    results = _usable(run)
    ks = list(range(1, max_k + 1))
    series: dict[str, list[float]] = {
        m: [] for m in ("hit_rate", "recall", "precision", "mrr", "ndcg", "graded_ndcg", "map")
    }
    for k in ks:
        rows = {m: [] for m in series}
        for r in results:
            units = [h.unit_id for h in r.ranked]
            rel = list(r.relevant_units)
            rows["hit_rate"].append(hit_rate(units, rel, k))
            rows["recall"].append(recall_at_k(units, rel, k))
            rows["precision"].append(precision_at_k(units, rel, k))
            rows["mrr"].append(mrr(units, rel, k))
            rows["ndcg"].append(ndcg_at_k(units, rel, k))
            rows["graded_ndcg"].append(graded_ndcg_at_k(units, r.relevant_units, k))
            rows["map"].append(average_precision(units, rel, k))
        for m in series:
            series[m].append(statistics.fmean(rows[m]) if rows[m] else 0.0)
    return {"ks": ks, "series": series, "n": len(results), "run_k": run.k}


def threshold_sweep(run: Run, thresholds: list[float] | None = None) -> dict:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    results = _usable(run)
    k = run.k
    deep = max(k, 10)
    points = []
    for t in thresholds:
        recall, precision, rr, kept, empty = [], [], [], [], 0
        for r in results:
            units = [h.unit_id for h in r.ranked if h.score >= t]
            rel = list(r.relevant_units)
            recall.append(recall_at_k(units, rel, k))
            precision.append(precision_at_k(units, rel, k))
            rr.append(mrr(units, rel, deep))
            kept.append(min(len(units), k))
            empty += not units
        n = len(results) or 1
        points.append(
            {
                "threshold": t,
                "recall": statistics.fmean(recall) if recall else 0.0,
                "precision": statistics.fmean(precision) if precision else 0.0,
                "mrr": statistics.fmean(rr) if rr else 0.0,
                "mean_kept_in_k": statistics.fmean(kept) if kept else 0.0,
                "empty_share": empty / n,
            }
        )
    return {
        "k": k,
        "n": len(results),
        "points": points,
        "score_kind": "best-chunk cosine similarity",
    }


def _content_overlap(query: str, body: str) -> float:
    doc_words = set(_WORD.findall(norm_text(body)))
    words = [w for w in _WORD.findall(norm_text(query)) if len(w) >= 4]
    return sum(1 for w in words if w in doc_words) / len(words) if words else 1.0


def failure_analysis(
    run: Run, bundle: Bundle | None, tags: dict, metric: str, threshold: float
) -> dict:
    items = {i.id: i for i in bundle.items} if bundle else {}
    failures = []
    suggested_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}

    for r in run.results:
        if r.error is not None or metric not in r.metrics or r.metrics[metric] >= threshold:
            continue
        item = items.get(r.item_id)
        relevant = set(item.relevant) if item else set()
        doc_order: list[str] = []
        for h in r.ranked:
            if h.doc_id not in doc_order:
                doc_order.append(h.doc_id)
        suggested: list[str] = []
        if not r.ranked:
            suggested.append("no_results")
        if relevant - set(doc_order):
            suggested.append("not_retrieved")
        ranks = [doc_order.index(d) + 1 for d in relevant if d in doc_order]
        if ranks and min(ranks) > run.k:
            suggested.append("ranked_too_low")
        if doc_order and item and doc_order[0] not in relevant:
            suggested.append(
                "distractor_outranked" if doc_order[0] in item.judged_nonrelevant else "label_gap"
            )
        if item and bundle:
            overlaps = [
                _content_overlap(item.query, bundle.corpus[d].body)
                for d in relevant
                if d in bundle.corpus
            ]
            if overlaps and max(overlaps) < 0.25:
                suggested.append("vocabulary_mismatch")
            if run.variant.level == "chunk":
                for ev in item.evidence:
                    doc = bundle.corpus.get(ev.doc_id)
                    if (
                        doc
                        and locate_evidence(doc.body, ev.text, chunk_text(doc.body))[
                            "straddles_boundary"
                        ]
                    ):
                        suggested.append("chunk_boundary")
                        break
            if item.trap and item.trap != "none":
                suggested.append("orthography_trap")
            recall = r.metrics.get("recall@k", 0.0)
            if len(relevant) > 1 and 0.0 < recall < 1.0:
                suggested.append("partial_multi_doc")
        if any(not h.novel for h in r.ranked[: run.k]):
            suggested.append("duplicate_evidence")

        manual = tags.get(r.item_id, {})
        for s in suggested:
            suggested_counts[s] = suggested_counts.get(s, 0) + 1
        for t in manual.get("tags", []):
            tag_counts[t] = tag_counts.get(t, 0) + 1
        failures.append(
            {
                "item_id": r.item_id,
                "query": r.query,
                "value": r.metrics[metric],
                "difficulty": item.difficulty if item else "",
                "trap": item.trap if item else "",
                "relevant": sorted(relevant),
                "relevant_ranks": {d: doc_order.index(d) + 1 for d in relevant if d in doc_order},
                "top": [
                    {
                        "rank": h.rank,
                        "doc_id": h.doc_id,
                        "grade": h.grade,
                        "judged": bool(item and h.doc_id in item.judged_docs),
                        "score": h.score,
                    }
                    for h in r.ranked[:5]
                ],
                "suggested": suggested,
                "tags": manual.get("tags", []),
                "note": manual.get("note", ""),
            }
        )

    scored = sum(1 for r in run.results if r.error is None and metric in r.metrics)
    return {
        "metric": metric,
        "threshold": threshold,
        "scored": scored,
        "failed": len(failures),
        "failure_rate": len(failures) / scored if scored else 0.0,
        "tagged": sum(1 for f in failures if f["tags"]),
        "suggested_counts": dict(sorted(suggested_counts.items(), key=lambda kv: -kv[1])),
        "tag_counts": dict(sorted(tag_counts.items(), key=lambda kv: -kv[1])),
        "taxonomy": TAXONOMY,
        "failures": sorted(failures, key=lambda f: f["value"]),
    }
