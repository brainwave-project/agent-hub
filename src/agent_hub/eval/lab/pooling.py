"""Pooled judging — find the relevant documents your labels missed.

A golden set labels the documents its author thought of. Every other document is
scored as non-relevant by assumption, so a retriever that finds a genuinely
relevant document the author never considered is PENALISED for it. The bias
favours whatever system resembles the labeller's mental search.

TREC's answer is pooling: run several diverse systems, take the union of their
top-`depth` results per question, and judge every document in that pool. What
the pool shows you:

    unjudged pooled docs    the assumption you are currently making, per question
    unique contributions    relevant documents that only ONE system retrieved —
                            evidence that a single-system pool would have missed
                            them, i.e. why pool diversity matters
    relevant never pooled   gold relevant documents no system surfaced within
                            `depth` — hard questions, or bad labels
"""

from __future__ import annotations

from agent_hub.eval.arabic.normalize import normalize
from agent_hub.eval.lab.jobs import Job
from agent_hub.eval.lab.models import GoldenItem, VariantSpec, now_iso
from agent_hub.eval.lab.runner import NORMALIZERS, KnowHubLab, index_is_current
from agent_hub.eval.lab.store import Bundle

__all__ = [
    "default_pool_variants",
    "build_pool",
    "pool_view",
    "simulate_pool_judgements",
    "EXCERPT_CHARS",
]

EXCERPT_CHARS = 500


def default_pool_variants(bundle: Bundle) -> list[VariantSpec]:
    """Diverse by construction: lexical+vector, vector-only, and chunk-level vector."""
    variants = [
        VariantSpec(name="hybrid", level="document", hybrid=True),
        VariantSpec(name="vector", level="document", hybrid=False),
        VariantSpec(name="chunk", level="chunk", hybrid=False),
    ]
    if bundle.meta.lang == "ar" or any(i.lang == "ar" for i in bundle.items):
        variants.append(
            VariantSpec(name="hybrid+qnorm", level="document", query_normalization="conservative")
        )
    return variants


def _ranked_docs(
    kh: KnowHubLab, bundle: Bundle, variant: VariantSpec, query: str, depth: int
) -> list[str]:
    owner = bundle.indexes[variant.corpus_index].owner_id
    sent = normalize(query, NORMALIZERS[variant.query_normalization])
    if variant.level == "chunk":
        hits = kh.search_chunks(sent, owner, 50, variant.min_score)
    else:
        hits = kh.search_documents(sent, owner, max(depth, 10), variant.hybrid, variant.min_score)
    ordered: list[str] = []
    for hit in hits:
        if hit.doc_id not in ordered:
            ordered.append(hit.doc_id)
        if len(ordered) >= depth:
            break
    return ordered


def build_pool(
    kh: KnowHubLab,
    bundle: Bundle,
    item_ids: list[str] | None,
    variants: list[VariantSpec] | None,
    depth: int,
    job: Job | None = None,
) -> dict:
    variants = variants or default_pool_variants(bundle)
    usable = [v for v in variants if index_is_current(bundle, v.corpus_index)[0]]
    skipped = [v.name for v in variants if v not in usable]
    if not usable:
        raise ValueError("no pooling variant has a current index — index the corpus first")
    selected = [i for i in bundle.items if not item_ids or i.id in item_ids]

    pool: dict = {
        "created_at": now_iso(),
        "depth": depth,
        "variants": [v.model_dump() for v in usable],
        "skipped_variants": skipped,
        "items": {},
    }
    if job:
        job.progress(0, len(selected), "pooling")
    for n, item in enumerate(selected, start=1):
        docs: dict[str, dict] = {}
        for variant in usable:
            for rank, doc_id in enumerate(_ranked_docs(kh, bundle, variant, item.query, depth), 1):
                entry = docs.setdefault(doc_id, {"retrieved_by": {}, "best_rank": rank})
                entry["retrieved_by"][variant.name] = rank
                entry["best_rank"] = min(entry["best_rank"], rank)
        pool["items"][item.id] = {"docs": docs}
        if job:
            job.progress(n, message=f"pooled {item.id}")
    return pool


def pool_view(bundle: Bundle, pool: dict | None) -> dict:
    """Join a stored pool with the CURRENT labels, so judging progress is live."""
    if not pool:
        return {"exists": False}
    items = {i.id: i for i in bundle.items}
    variant_names = [v["name"] for v in pool.get("variants", [])]
    unique = {name: 0 for name in variant_names}
    rows = []
    total = judged = relevant_pooled = never_pooled = 0

    for item_id, entry in pool.get("items", {}).items():
        item = items.get(item_id)
        if item is None:
            continue
        docs = []
        for doc_id, info in sorted(entry["docs"].items(), key=lambda kv: kv[1]["best_rank"]):
            if doc_id in item.relevant:
                state = f"relevant:{item.relevant[doc_id]}"
            elif doc_id in item.judged_nonrelevant:
                state = "nonrelevant"
            else:
                state = "unjudged"
            total += 1
            judged += state != "unjudged"
            if state.startswith("relevant"):
                relevant_pooled += 1
                if len(info["retrieved_by"]) == 1:
                    unique[next(iter(info["retrieved_by"]))] += 1
            doc = bundle.corpus.get(doc_id)
            docs.append(
                {
                    "doc_id": doc_id,
                    "judgement": state,
                    "best_rank": info["best_rank"],
                    "retrieved_by": info["retrieved_by"],
                    "excerpt": doc.body[:EXCERPT_CHARS] if doc else "",
                }
            )
        missing = sorted(set(item.relevant) - set(entry["docs"]))
        never_pooled += len(missing)
        rows.append(
            {
                "item_id": item_id,
                "query": item.query,
                "split": item.split,
                "docs": docs,
                "unjudged": sum(1 for d in docs if d["judgement"] == "unjudged"),
                "relevant_not_pooled": missing,
            }
        )

    return {
        "exists": True,
        "created_at": pool.get("created_at"),
        "depth": pool.get("depth"),
        "variants": variant_names,
        "skipped_variants": pool.get("skipped_variants", []),
        "summary": {
            "items": len(rows),
            "pooled_pairs": total,
            "judged_pairs": judged,
            "judged_share": judged / total if total else 0.0,
            "items_fully_judged": sum(1 for r in rows if r["unjudged"] == 0),
            "relevant_pooled": relevant_pooled,
            "relevant_never_pooled": never_pooled,
            "unique_relevant_by_variant": unique,
        },
        "items": sorted(rows, key=lambda r: (-r["unjudged"], r["item_id"])),
    }


def simulate_pool_judgements(
    bundle: Bundle,
    pool: dict | None,
    item_id: str | None = None,
) -> tuple[list[GoldenItem], int]:
    """Mark every still-unjudged pooled document as explicit 0 (not relevant).

    Gold relevant grades are left alone. This is a lab shortcut so a pool can
    be finished without clicking every pair — not a substitute for reading.
    """
    if not pool:
        raise ValueError("no pool yet — build one first")
    rows = list((pool_view(bundle, pool).get("items") or []))
    if item_id:
        rows = [r for r in rows if r["item_id"] == item_id]
        if not rows:
            raise ValueError(f"{item_id} is not in the pool")
    by_id = {i.id: i for i in bundle.items}
    filled = 0
    for row in rows:
        item = by_id.get(row["item_id"])
        if item is None:
            continue
        nonrelevant = list(item.judged_nonrelevant)
        added = 0
        for doc in row["docs"]:
            if doc["judgement"] != "unjudged":
                continue
            doc_id = doc["doc_id"]
            if doc_id in item.relevant or doc_id in nonrelevant:
                continue
            nonrelevant.append(doc_id)
            added += 1
        if added:
            by_id[item.id] = item.model_copy(update={"judged_nonrelevant": sorted(set(nonrelevant))})
            filled += added
    if filled == 0:
        raise ValueError("every pooled document is already judged")
    return [by_id[i.id] for i in bundle.items], filled
