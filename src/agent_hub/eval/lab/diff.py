"""Version diff — what changed between two dataset versions, and what it breaks.

Forking is cheap; the discipline is knowing what a fork did to comparability.
The diff classifies every change by its consequence:

    corpus changed      every index is stale; retrieval scores across versions
                        are not comparable at all
    questions changed   aggregate scores are not comparable (different
                        population); per-question comparisons on unchanged
                        items still are
    labels changed      same questions, different truth — scores move without
                        the system moving
    split changed       dev -> holdout moves are CONTAMINATION: the moved items
                        were tuned on, so the new holdout is optimistic
"""

from __future__ import annotations

import hashlib

from agent_hub.eval.lab.store import Bundle

__all__ = ["diff_versions", "ITEM_FIELDS"]

ITEM_FIELDS = (
    "query",
    "relevant",
    "judged_nonrelevant",
    "evidence",
    "ground_truth",
    "difficulty",
    "trap",
    "tags",
    "split",
    "origin",
)
LABEL_FIELDS = {"relevant", "judged_nonrelevant", "evidence"}


def _body_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def diff_versions(before: Bundle, after: Bundle) -> dict:
    docs_a, docs_b = before.corpus, after.corpus
    docs_added = sorted(set(docs_b) - set(docs_a))
    docs_removed = sorted(set(docs_a) - set(docs_b))
    docs_changed = [
        {
            "doc_id": d,
            "words_before": len(docs_a[d].body.split()),
            "words_after": len(docs_b[d].body.split()),
        }
        for d in sorted(set(docs_a) & set(docs_b))
        if _body_hash(docs_a[d].body) != _body_hash(docs_b[d].body)
    ]

    items_a = {i.id: i for i in before.items}
    items_b = {i.id: i for i in after.items}
    items_added = sorted(set(items_b) - set(items_a))
    items_removed = sorted(set(items_a) - set(items_b))
    items_changed = []
    split_moves = []
    for item_id in sorted(set(items_a) & set(items_b)):
        a, b = items_a[item_id].model_dump(mode="json"), items_b[item_id].model_dump(mode="json")
        fields = {
            f: {"before": a.get(f), "after": b.get(f)} for f in ITEM_FIELDS if a.get(f) != b.get(f)
        }
        if fields:
            items_changed.append({"id": item_id, "fields": fields})
        if a.get("split") != b.get("split"):
            split_moves.append({"id": item_id, "from": a.get("split"), "to": b.get("split")})

    relabeled = [c["id"] for c in items_changed if LABEL_FIELDS & set(c["fields"])]
    requeried = [c["id"] for c in items_changed if "query" in c["fields"]]
    contaminating = [m["id"] for m in split_moves if m["from"] == "dev" and m["to"] == "holdout"]

    consequences = []
    if docs_added or docs_removed or docs_changed:
        consequences.append(
            {
                "severity": "blocker",
                "title": "Corpus changed — indexes are stale",
                "detail": "Retrieval scores on the two versions are not comparable: the thing being "
                "searched is different. Reindex before running anything on the new version.",
            }
        )
    if items_added or items_removed or requeried:
        consequences.append(
            {
                "severity": "warning",
                "title": "Question set changed",
                "detail": "Aggregate scores describe different populations. Compare per-question on "
                "the unchanged items, or re-run both systems on the new version.",
            }
        )
    if relabeled:
        consequences.append(
            {
                "severity": "warning",
                "title": f"{len(relabeled)} item(s) relabelled",
                "detail": "Same questions, different truth. A score can move with no change to the "
                "system — record why each label changed.",
            }
        )
    if contaminating:
        consequences.append(
            {
                "severity": "blocker",
                "title": f"{len(contaminating)} item(s) moved dev → holdout",
                "detail": "These questions were visible while tuning. Putting them in the holdout makes "
                "it optimistic by exactly how much you overfit to them.",
            }
        )
    if not consequences:
        consequences.append(
            {
                "severity": "pass",
                "title": "No content changes",
                "detail": "Only metadata differs; scores remain comparable.",
            }
        )

    return {
        "from_version": before.manifest.version,
        "to_version": after.manifest.version,
        "from_hash": before.current_hash,
        "to_hash": after.current_hash,
        "docs": {"added": docs_added, "removed": docs_removed, "changed": docs_changed},
        "items": {"added": items_added, "removed": items_removed, "changed": items_changed},
        "split_moves": split_moves,
        "relabeled": relabeled,
        "consequences": consequences,
        "comparable": not (
            docs_added
            or docs_removed
            or docs_changed
            or items_added
            or items_removed
            or requeried
            or relabeled
            or contaminating
        ),
    }
