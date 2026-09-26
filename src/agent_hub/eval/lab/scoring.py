"""Turn a ranking into metrics — at document level or evidence level.

The metrics themselves are NOT reimplemented here. This module only decides
what the ranked ids and relevant ids are, then calls `retrieval_metrics`, so a
lab number and a `make evals` number come from the same arithmetic.

**Document level.** Unit = document. Relevant units = the item's graded
relevant documents. Identical to the offline harness.

**Evidence level (chunk results).** Chunk ids are unstable: they change every
time the chunker changes, so a label that says "chunk 8f3a is relevant" expires
on the next reindex. Instead each relevant document contributes one unit per
evidence span (or one unit for the whole document if it has no spans). A
retrieved chunk earns credit for the first *not yet credited* unit it contains:

    recall@k     fraction of evidence spans covered by the top-k chunks
    precision@k  fraction of the top-k chunks that contributed NEW evidence

Two conventions to know:

* A chunk that repeats already-credited evidence is not relevant (novel=False).
  Returning the same paragraph three times does not help a generator.
* One chunk, one credit. A chunk containing two spans credits the first; the
  second can still be credited by a later chunk. This keeps "top-k" meaning
  "top-k chunks", at the cost of slightly pessimistic recall when a single
  chunk contains several spans.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from agent_hub.eval.lab.chunking import norm_text
from agent_hub.eval.lab.models import GoldenItem, RankedHit
from agent_hub.eval.retrieval_metrics import graded_ndcg_at_k, score_all

__all__ = [
    "RawHit",
    "document_units",
    "evidence_units",
    "rank_documents",
    "rank_chunks",
    "compute_metrics",
    "explain",
    "EXCERPT_CHARS",
]

EXCERPT_CHARS = 600


@dataclass(frozen=True)
class RawHit:
    """A search result before relevance is attached."""

    doc_id: str
    document_id: str = ""
    chunk_id: str = ""
    score: float = 0.0
    rrf_score: float = 0.0
    text: str = ""


def document_units(relevant: dict[str, int]) -> dict[str, int]:
    return dict(relevant)


def evidence_units(item: GoldenItem, relevant: dict[str, int]) -> dict[str, tuple[int, str | None]]:
    """unit id -> (grade, normalised span or None for whole-document units)."""
    units: dict[str, tuple[int, str | None]] = {}
    for doc_id, grade in relevant.items():
        spans = [e.text for e in item.evidence if e.doc_id == doc_id]
        if not spans:
            units[doc_id] = (grade, None)
        for n, span in enumerate(spans):
            units[f"{doc_id}#e{n}"] = (grade, norm_text(span))
    return units


def rank_documents(hits: list[RawHit], relevant: dict[str, int]) -> list[RankedHit]:
    ranked: list[RankedHit] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.doc_id in seen:  # know-hub is already DISTINCT per doc; belt and braces
            continue
        seen.add(hit.doc_id)
        ranked.append(
            RankedHit(
                rank=len(ranked) + 1,
                unit_id=hit.doc_id,
                doc_id=hit.doc_id,
                document_id=hit.document_id,
                score=hit.score,
                rrf_score=hit.rrf_score,
                excerpt=hit.text[:EXCERPT_CHARS],
                grade=relevant.get(hit.doc_id, 0),
            )
        )
    return ranked


def rank_chunks(
    hits: list[RawHit], item: GoldenItem, relevant: dict[str, int]
) -> tuple[list[RankedHit], dict[str, int]]:
    units = evidence_units(item, relevant)
    credited: set[str] = set()
    ranked: list[RankedHit] = []

    for position, hit in enumerate(hits, start=1):
        text = norm_text(hit.text)
        candidates = [
            unit
            for unit, (_, span) in units.items()
            if unit.split("#e")[0] == hit.doc_id and (span is None or span in text)
        ]
        fresh = [u for u in candidates if u not in credited]
        if fresh:
            unit = fresh[0]
            credited.add(unit)
            grade, novel = units[unit][0], True
        elif candidates:
            unit, grade, novel = f"dup:{hit.chunk_id or position}", 0, False
        else:
            unit, grade, novel = f"chunk:{hit.chunk_id or position}", 0, True
        ranked.append(
            RankedHit(
                rank=position,
                unit_id=unit,
                doc_id=hit.doc_id,
                document_id=hit.document_id,
                chunk_id=hit.chunk_id,
                score=hit.score,
                excerpt=hit.text[:EXCERPT_CHARS],
                grade=grade,
                novel=novel,
            )
        )
    return ranked, {unit: grade for unit, (grade, _) in units.items()}


def compute_metrics(ranked_units: list[str], grades: dict[str, int], k: int) -> dict[str, float]:
    metrics = score_all(ranked_units, list(grades), k=k)
    metrics["graded_ndcg@k"] = graded_ndcg_at_k(ranked_units, grades, max(k, 10))
    return metrics


def explain(ranked: list[RankedHit], grades: dict[str, int], k: int) -> dict:
    """Every term behind the metrics, rank by rank — the arithmetic, shown.

    Mirrors the formulas in `retrieval_metrics` exactly, including the two
    cutoffs: set metrics at k, rank-aware metrics at max(k, 10).
    """
    deep_k = max(k, 10)
    relevant_count = len(grades)
    rows = []
    hits = 0
    first_rank: int | None = None
    dcg = graded_dcg = ap_sum = 0.0

    for hit in ranked[:deep_k]:
        is_relevant = hit.unit_id in grades
        discount = 1.0 / math.log2(hit.rank + 1)
        precision_here = None
        if is_relevant:
            hits += 1
            precision_here = hits / hit.rank
            ap_sum += precision_here
            dcg += discount
            graded_dcg += (2 ** grades[hit.unit_id] - 1) * discount
            if first_rank is None:
                first_rank = hit.rank
        rows.append(
            {
                "rank": hit.rank,
                "unit_id": hit.unit_id,
                "relevant": is_relevant,
                "grade": grades.get(hit.unit_id, 0),
                "in_k": hit.rank <= k,
                "discount": discount,
                "dcg_gain": discount if is_relevant else 0.0,
                "graded_gain": (2 ** grades[hit.unit_id] - 1) * discount if is_relevant else 0.0,
                "precision_at_rank": precision_here,
                "cumulative_hits": hits,
            }
        )

    ideal_hits = min(relevant_count, deep_k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    ideal_grades = sorted(grades.values(), reverse=True)[:deep_k]
    graded_idcg = sum((2**g - 1) / math.log2(r + 1) for r, g in enumerate(ideal_grades, start=1))
    found_in_k = len({h.unit_id for h in ranked[:k] if h.unit_id in grades})

    return {
        "k": k,
        "deep_k": deep_k,
        "relevant_count": relevant_count,
        "found_in_k": found_in_k,
        "returned_in_k": min(k, len(ranked)),
        "first_relevant_rank": first_rank,
        "dcg": dcg,
        "idcg": idcg,
        "graded_dcg": graded_dcg,
        "graded_idcg": graded_idcg,
        "ap_sum": ap_sum,
        "ap_denominator": min(relevant_count, deep_k),
        "rows": rows,
    }
