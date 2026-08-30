"""Retrieval metrics, implemented by hand.

These are deliberately NOT imported from a library. The point of this module is
that the mechanism is readable: you can see exactly what each metric counts, and
therefore exactly what it hides. Every function's docstring names the specific
situation in which the number lies to you.

Vocabulary used throughout:

    ranked      the retrieval system's output, best-first, as a list of ids
                (document ids or chunk ids — the metrics don't care which,
                as long as `relevant` uses the same id space)
    relevant    the ground-truth set of ids that SHOULD have been retrieved
    k           the cutoff — only the first k results are considered

All functions return a float in [0, 1] and are safe on empty input (they return
0.0 rather than raising), because a retrieval system returning nothing is a real
outcome that should score zero, not crash the harness.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

__all__ = [
    "hit_rate",
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "average_precision",
    "score_all",
    "METRIC_NAMES",
]

METRIC_NAMES = ("hit_rate", "recall@k", "precision@k", "mrr", "ndcg@k", "map")


def _truncate(ranked: Sequence[str], k: int) -> list[str]:
    if k <= 0:
        return []
    return list(ranked[:k])


def hit_rate(ranked: Sequence[str], relevant: Iterable[str], k: int = 5) -> float:
    """1.0 if ANY relevant id appears in the top k, else 0.0.

    MISLEADS WHEN: your corpus is small or your k is generous. Hit rate saturates
    — once most questions find *something*, it pins at ~1.0 and stops moving even
    as ranking quality changes underneath it. A system that puts the right answer
    at rank 1 and a system that puts it at rank 5 score identically here. Use it
    as a floor check ("are we finding anything at all?"), never as the metric you
    tune against.
    """
    rel = set(relevant)
    if not rel:
        return 0.0
    return 1.0 if any(doc in rel for doc in _truncate(ranked, k)) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int = 5) -> float:
    """Fraction of all relevant ids that made it into the top k.

    This is the metric that matters most for RAG, because a generator cannot
    cite what retrieval never returned. Recall failures are unrecoverable
    downstream; precision failures are merely noisy.

    MISLEADS WHEN: k is close to your corpus size — recall trivially approaches
    1.0 because you are returning nearly everything. Always report k alongside
    the number, and keep k at the value the generator actually consumes.
    """
    rel = set(relevant)
    if not rel:
        return 0.0
    found = sum(1 for doc in set(_truncate(ranked, k)) if doc in rel)
    return found / len(rel)


def precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int = 5) -> float:
    """Fraction of the top k that is actually relevant.

    MISLEADS WHEN: the number of relevant documents is smaller than k. If only
    1 document is relevant and k=5, precision@5 caps at 0.20 — the system is
    perfect and the metric reads as a failure. Precision is only interpretable
    when you know the size of the relevant set, which is why this harness
    reports it next to recall rather than alone.
    """
    rel = set(relevant)
    top = _truncate(ranked, k)
    if not top:
        return 0.0
    return sum(1 for doc in top if doc in rel) / len(top)


def mrr(ranked: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    """Reciprocal of the rank of the FIRST relevant result. 1/1, 1/2, 1/3...

    MISLEADS WHEN: there is more than one relevant document. MRR stops looking
    the instant it finds one hit, so it is structurally blind to recall. A system
    that retrieves 1 of 5 relevant docs at rank 1 scores a perfect 1.0 — the same
    as a system that retrieved all 5. If your questions have multi-document
    answers, MRR will tell you everything is fine while the generator starves.
    """
    rel = set(relevant)
    if not rel:
        return 0.0
    for position, doc in enumerate(_truncate(ranked, k), start=1):
        if doc in rel:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    """Normalised Discounted Cumulative Gain with binary relevance.

    DCG sums 1/log2(rank+1) over relevant hits, so earlier hits are worth more.
    IDCG is the same sum for a perfect ranking, which normalises the score into
    [0, 1] and makes questions with different numbers of relevant docs
    comparable to each other.

    MISLEADS WHEN: you care about changes deep in the ranking. The log discount
    is brutal — rank 1 is worth 1.00, rank 10 is worth 0.29, rank 20 is worth
    0.23. Moving a result from rank 20 to rank 11 barely registers, so NDCG will
    report "no change" for reranking work that a user would plainly notice.
    It also assumes graded relevance in general; here it is binary, which throws
    away the distinction between a perfect answer and a merely related one.
    """
    rel = set(relevant)
    if not rel:
        return 0.0

    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, doc in enumerate(_truncate(ranked, k), start=1)
        if doc in rel
    )
    # Ideal: every relevant doc packed into the top positions, up to k.
    ideal_hits = min(len(rel), max(k, 0))
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(ranked: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    """Mean of precision@i taken at each rank i where a relevant doc appears.

    Unlike MRR this rewards finding ALL the relevant documents, and unlike plain
    recall it rewards finding them EARLY. Averaged across questions this is MAP.

    MISLEADS WHEN: relevant-set sizes vary a lot across your question set. AP is
    normalised by the number of relevant docs, so questions with a single
    relevant doc dominate the variance and a handful of them can swing MAP.
    Stratify before trusting it.
    """
    rel = set(relevant)
    if not rel:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for position, doc in enumerate(_truncate(ranked, k), start=1):
        if doc in rel:
            hits += 1
            precision_sum += hits / position
    return precision_sum / min(len(rel), k) if hits else 0.0


def score_all(ranked: Sequence[str], relevant: Iterable[str], k: int = 5) -> dict[str, float]:
    """Every metric for one question, as a flat dict.

    `mrr` and `ndcg@k` use a deeper cutoff (max(k, 10)) than the set-based
    metrics on purpose: rank-aware metrics are uninformative at a shallow k,
    while recall/precision must be measured at exactly the k the generator
    consumes. Reporting both cutoffs from one call keeps that distinction
    explicit instead of hiding it in the caller.
    """
    rel = list(relevant)
    deep_k = max(k, 10)
    return {
        "hit_rate": hit_rate(ranked, rel, k),
        "recall@k": recall_at_k(ranked, rel, k),
        "precision@k": precision_at_k(ranked, rel, k),
        "mrr": mrr(ranked, rel, deep_k),
        "ndcg@k": ndcg_at_k(ranked, rel, deep_k),
        "map": average_precision(ranked, rel, deep_k),
    }
