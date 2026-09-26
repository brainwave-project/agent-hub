"""Label audit — how much should you trust your own relevance judgements?

Every retrieval metric inherits the error rate of its labels. If two careful
people disagree on 30% of (question, document) pairs, a 3-point recall
difference between two systems is below the resolution of the ruler.

The procedure here is the standard one:

1. A second annotator re-judges a random sample of items **blind** — they see
   the question and a shuffled set of candidate documents, never the existing
   labels or which documents were retrieved first.
2. Their grades are compared with the existing labels ("gold") or with another
   annotator.
3. Agreement is reported three ways, because each hides something:

   percent agreement   what everyone reports; inflated by chance
   Cohen's kappa       agreement beyond chance on relevant/not relevant
   weighted kappa      graded (0-3) agreement where a 2-vs-3 disagreement
                       costs less than a 0-vs-3 one (quadratic weights)

The trap worth seeing with your own data — the **kappa paradox**: when almost
every pair is "not relevant", two annotators agree 90% of the time by chance
alone, so high percent agreement coexists with low kappa. Neither number is
wrong; they answer different questions.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from agent_hub.eval.lab.models import GoldenItem, now_iso
from agent_hub.eval.lab.store import Bundle

__all__ = [
    "GOLD",
    "CATEGORIES",
    "percent_agreement",
    "cohens_kappa",
    "weighted_kappa",
    "confusion_matrix",
    "interpret_kappa",
    "gold_grade",
    "build_task",
    "agreement_report",
    "simulate_annotations",
]

GOLD = "gold"
CATEGORIES = (0, 1, 2, 3)
EXCERPT_CHARS = 900


def percent_agreement(a: Sequence[int], b: Sequence[int]) -> float:
    return sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a) if a else 0.0


def cohens_kappa(a: Sequence[int], b: Sequence[int]) -> float | None:
    """(p_observed - p_expected) / (1 - p_expected), nominal categories.

    p_expected is the agreement two annotators would reach by labelling at
    random with their own label frequencies. Returns None when p_expected is 1
    (both raters used a single identical label): kappa is undefined there, and
    reporting 1.0 or 0.0 would both be a lie.
    """
    n = len(a)
    if n == 0:
        return None
    cats = sorted(set(a) | set(b))
    po = percent_agreement(a, b)
    pe = sum((list(a).count(c) / n) * (list(b).count(c) / n) for c in cats)
    if pe >= 1.0:
        return None
    return (po - pe) / (1.0 - pe)


def weighted_kappa(
    a: Sequence[int], b: Sequence[int], categories: Sequence[int] = CATEGORIES
) -> float | None:
    """Quadratic-weighted kappa for ordinal grades.

    Disagreement weight w_ij = (i - j)^2 / (k - 1)^2, so grades one step apart
    cost 1/9 of grades three steps apart. kappa_w = 1 - sum(w * O) / sum(w * E)
    with O the observed and E the chance-expected joint proportions.
    """
    n = len(a)
    if n == 0:
        return None
    k = len(categories)
    index = {c: i for i, c in enumerate(categories)}
    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b, strict=True):
        observed[index[x]][index[y]] += 1.0 / n
    rows = [sum(observed[i]) for i in range(k)]
    cols = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / (k - 1) ** 2
            num += w * observed[i][j]
            den += w * rows[i] * cols[j]
    if den == 0.0:
        return None
    return 1.0 - num / den


def confusion_matrix(
    a: Sequence[int], b: Sequence[int], categories: Sequence[int] = CATEGORIES
) -> list[list[int]]:
    index = {c: i for i, c in enumerate(categories)}
    matrix = [[0] * len(categories) for _ in categories]
    for x, y in zip(a, b, strict=True):
        matrix[index[x]][index[y]] += 1
    return matrix


def interpret_kappa(kappa: float | None) -> str:
    """Landis & Koch (1977) bands — a convention, not a law. Quote the number too."""
    if kappa is None:
        return "undefined (no variation in labels)"
    if kappa < 0:
        return "worse than chance"
    for bound, label in ((0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")):
        if kappa <= bound:
            return label
    return "almost perfect"


def gold_grade(item: GoldenItem, doc_id: str) -> tuple[int, bool]:
    """(grade, assumed). `assumed` is True when the document was never judged."""
    if doc_id in item.relevant:
        return item.relevant[doc_id], False
    if doc_id in item.judged_nonrelevant:
        return 0, False
    return 0, True


def build_task(
    bundle: Bundle,
    annotations: dict,
    annotator: str,
    sample: int,
    seed: int,
    pool: dict | None = None,
) -> dict:
    """A blind annotation batch: items this annotator has not judged yet.

    Candidate documents per item = every judged document + the top pooled
    documents (if a pool exists) + random corpus documents up to four, shuffled
    deterministically. Gold labels are NOT included in the response.
    """
    done = set(annotations.get("annotators", {}).get(annotator, {}))
    remaining = [i for i in bundle.items if i.id not in done]
    rng = random.Random(f"{seed}:{annotator}")
    chosen = rng.sample(remaining, k=min(sample, len(remaining)))
    pool_items = (pool or {}).get("items", {})
    doc_ids = sorted(bundle.corpus)

    batch = []
    for item in chosen:
        candidates = list(item.judged_docs)
        pooled = sorted(
            pool_items.get(item.id, {}).get("docs", {}).items(), key=lambda kv: kv[1]["best_rank"]
        )
        candidates += [d for d, _ in pooled[:5] if d not in candidates]
        item_rng = random.Random(f"{seed}:{annotator}:{item.id}")
        others = [d for d in doc_ids if d not in candidates]
        item_rng.shuffle(others)
        while len(candidates) < 4 and others:
            candidates.append(others.pop())
        candidates = [d for d in candidates if d in bundle.corpus][:8]
        item_rng.shuffle(candidates)
        batch.append(
            {
                "item_id": item.id,
                "query": item.query,
                "lang": item.lang,
                "docs": [
                    {"doc_id": d, "excerpt": bundle.corpus[d].body[:EXCERPT_CHARS]}
                    for d in candidates
                ],
            }
        )
    return {"annotator": annotator, "items": batch, "remaining": len(remaining) - len(chosen)}


def _rater_grades(bundle: Bundle, annotations: dict, rater: str) -> dict[str, dict[str, int]]:
    if rater == GOLD:
        return {}
    return {
        item_id: {d: int(g) for d, g in entry.get("grades", {}).items()}
        for item_id, entry in annotations.get("annotators", {}).get(rater, {}).items()
    }


def agreement_report(bundle: Bundle, annotations: dict, rater_a: str, rater_b: str) -> dict:
    if rater_a == rater_b:
        raise ValueError("choose two different raters")
    if rater_a != GOLD and rater_b == GOLD:
        rater_a, rater_b = rater_b, rater_a
    items = {i.id: i for i in bundle.items}
    grades_a = _rater_grades(bundle, annotations, rater_a)
    grades_b = _rater_grades(bundle, annotations, rater_b)

    a_vals: list[int] = []
    b_vals: list[int] = []
    assumed = 0
    disagreements = []
    per_item: dict[str, dict] = {}
    for item_id, b_item in sorted(grades_b.items()):
        item = items.get(item_id)
        if item is None:
            continue
        if rater_a == GOLD:
            a_item = {}
            for doc_id in b_item:
                grade, was_assumed = gold_grade(item, doc_id)
                a_item[doc_id] = grade
                assumed += int(was_assumed)
        else:
            a_item = grades_a.get(item_id)
            if a_item is None:
                continue
        shared = [d for d in b_item if d in a_item]
        agree = 0
        for doc_id in shared:
            ga, gb = a_item[doc_id], b_item[doc_id]
            a_vals.append(ga)
            b_vals.append(gb)
            if ga == gb:
                agree += 1
            else:
                disagreements.append(
                    {
                        "item_id": item_id,
                        "query": item.query,
                        "doc_id": doc_id,
                        "a": ga,
                        "b": gb,
                        "distance": abs(ga - gb),
                        "binary_flip": (ga > 0) != (gb > 0),
                        "gold_assumed": rater_a == GOLD and gold_grade(item, doc_id)[1],
                    }
                )
        if shared:
            per_item[item_id] = {"pairs": len(shared), "agree": agree}

    binary_a = [int(x > 0) for x in a_vals]
    binary_b = [int(x > 0) for x in b_vals]
    kappa_b = cohens_kappa(binary_a, binary_b)
    kappa_w = weighted_kappa(a_vals, b_vals)
    po_binary = percent_agreement(binary_a, binary_b)
    n = len(a_vals)
    prevalence = sum(binary_a) / n if n else 0.0

    notes = []
    simulated = (annotations or {}).get("simulated", {})
    for rater in (rater_a, rater_b):
        if rater in simulated:
            params = simulated[rater]
            notes.append(
                f"'{rater}' is SIMULATED: grades were generated from the gold labels at "
                f"{params.get('agreement', 0):.0%} agreement (seed {params.get('seed')}), not judged "
                "by a person. Kappa here measures the simulation's noise setting, not label quality."
            )
    if n and po_binary >= 0.85 and kappa_b is not None and kappa_b < 0.6:
        notes.append(
            "Kappa paradox: raw agreement is high because most pairs are 'not relevant' — "
            "chance alone produces that. Kappa discounts it."
        )
    if assumed:
        notes.append(
            f"{assumed} gold grade(s) were ASSUMED 0 because the document was never judged. "
            "A disagreement there may be a gold labelling gap, not an annotator error."
        )
    if n and n < 30:
        notes.append(f"Only {n} pairs: kappa is itself a noisy estimate at this size.")

    return {
        "rater_a": rater_a,
        "rater_b": rater_b,
        "items": len(per_item),
        "pairs": n,
        "assumed_pairs": assumed,
        "percent_agreement": percent_agreement(a_vals, b_vals) if n else None,
        "percent_agreement_binary": po_binary if n else None,
        "kappa_binary": kappa_b,
        "kappa_weighted": kappa_w,
        "interpretation_binary": interpret_kappa(kappa_b),
        "interpretation_weighted": interpret_kappa(kappa_w),
        "prevalence_relevant": prevalence,
        "confusion": confusion_matrix(a_vals, b_vals),
        "confusion_binary": confusion_matrix(binary_a, binary_b, (0, 1)),
        "disagreements": sorted(disagreements, key=lambda d: (-d["distance"], d["item_id"])),
        "notes": notes,
    }


def simulate_annotations(
    bundle: Bundle,
    annotations: dict,
    annotator: str,
    sample: int,
    agreement: float,
    seed: int,
    pool: dict | None = None,
) -> tuple[dict, int]:
    """Fabricate an annotator from the gold labels - for walking the flow, not judging.

    Each candidate document keeps its gold grade with probability `agreement`
    and is otherwise moved to a different grade, adjacent grades likelier than
    distant ones. That produces a plausible-looking kappa without anyone
    reading a document, which is exactly why every entry is stamped
    `simulated` and the rater is registered under `annotations["simulated"]`:
    the agreement report, the readiness check and the Learning Path all read
    that flag and refuse to treat these grades as real judging.
    """
    task = build_task(bundle, annotations, annotator, sample, seed, pool)
    items = {i.id: i for i in bundle.items}
    rng = random.Random(f"sim:{seed}:{annotator}")

    bucket = annotations.setdefault("annotators", {}).setdefault(annotator, {})
    for entry in task["items"]:
        item = items.get(entry["item_id"])
        if item is None:
            continue
        grades: dict[str, int] = {}
        for doc in entry["docs"]:
            gold, _ = gold_grade(item, doc["doc_id"])
            if rng.random() < agreement:
                grades[doc["doc_id"]] = gold
            else:
                others = [g for g in CATEGORIES if g != gold]
                weights = [1.0 / abs(g - gold) for g in others]
                grades[doc["doc_id"]] = rng.choices(others, weights=weights)[0]
        bucket[entry["item_id"]] = {
            "grades": grades,
            "seconds": round(rng.uniform(4.0, 40.0), 1),
            "created_at": now_iso(),
            "simulated": True,
        }

    annotations.setdefault("simulated", {})[annotator] = {
        "agreement": agreement,
        "seed": seed,
        "at": now_iso(),
        "items": len(task["items"]),
    }
    return annotations, len(task["items"])
