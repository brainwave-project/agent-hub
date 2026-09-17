"""Data Lab — is this dataset fit to evaluate with?

A score is only as good as the set it was measured on, and most eval sets fail
silently: nothing crashes, the number just means less than it appears to. Every
check here targets one specific way that happens and says so in `learn`.

Severity is about what the defect does to a conclusion:

    blocker   the number would be wrong (unresolvable labels, leakage across
              splits, a stale index, a frozen file edited behind your back)
    warning   the number would be right but easy to over-read (tiny strata,
              verbatim queries, underpowered splits)
    info      worth knowing, not worth fixing now
    pass      checked, fine

Verdict: any blocker -> not_ready; any warning -> ready_with_caveats; else ready.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field

from agent_hub.eval.lab.agreement import agreement_report
from agent_hub.eval.lab.chunking import chunk_text, locate_evidence, norm_text
from agent_hub.eval.lab.models import GoldenItem
from agent_hub.eval.lab.pooling import pool_view
from agent_hub.eval.lab.splits import summarize
from agent_hub.eval.lab.store import Bundle
from agent_hub.eval.stats import detectable_effect, min_sample_size

__all__ = ["Check", "assess", "DEFAULT_SCORE_SD"]

#: Placeholder spread for per-query scores until a real run supplies one.
DEFAULT_SCORE_SD = 0.35
MIN_PER_STRATUM = 5
NEAR_DUPLICATE_JACCARD = 0.8
MAX_PAIRWISE_ITEMS = 1500

_TOKEN = re.compile(r"\w+", re.UNICODE)


@dataclass
class Check:
    id: str
    category: str
    severity: str
    title: str
    detail: str
    learn: str
    affected: list[str] = field(default_factory=list)


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(norm_text(text)) if len(t) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def assess(
    bundle: Bundle,
    k: int = 5,
    score_sd: float | None = None,
    pool: dict | None = None,
    annotations: dict | None = None,
) -> dict:
    checks: list[Check] = []
    corpus, items, manifest = bundle.corpus, bundle.items, bundle.manifest
    sd = score_sd if score_sd and score_sd > 0 else DEFAULT_SCORE_SD
    add = checks.append

    def passed(id_: str, category: str, title: str, detail: str, learn: str) -> None:
        add(Check(id_, category, "pass", title, detail, learn))

    # --- integrity --------------------------------------------------------
    current = bundle.current_hash
    if manifest.status == "frozen":
        if manifest.content_hash != current:
            add(
                Check(
                    "integrity.hash",
                    "Integrity",
                    "blocker",
                    "Frozen content was modified",
                    f"Recorded hash {str(manifest.content_hash)[:12]} but files now hash to {current[:12]}.",
                    "A frozen version is a promise that 'v3' always means the same questions and "
                    "documents. Something edited the files outside the lab, so every run scored "
                    "against the old hash is no longer comparable with anything new. Restore the "
                    "files from version control or fork a new version.",
                )
            )
        else:
            passed(
                "integrity.hash",
                "Integrity",
                "Frozen content intact",
                f"Content hash {current[:12]} matches the freeze record.",
                "Recomputed on every check, so an out-of-band edit cannot hide.",
            )
    else:
        add(
            Check(
                "integrity.draft",
                "Integrity",
                "warning",
                "Dataset is a mutable draft",
                "Drafts can change under a run. Freeze before quoting any number.",
                "Data versioning & immutability: a result is a pair (system, dataset hash). If the "
                "dataset can still change, the result has no stable meaning — you cannot rerun it, "
                "and you cannot tell whether a later score moved because the system changed or "
                "because the questions did.",
            )
        )

    # --- labels -----------------------------------------------------------
    if not items:
        add(
            Check(
                "labels.empty",
                "Labels",
                "blocker",
                "No golden items",
                "Add questions with relevance judgements.",
                "Nothing to measure.",
            )
        )
    if not corpus:
        add(
            Check(
                "labels.no_corpus",
                "Labels",
                "blocker",
                "Empty corpus",
                "Add documents to retrieve from.",
                "Nothing to retrieve.",
            )
        )

    unresolved = sorted({f"{i.id}->{d}" for i in items for d in i.relevant if d not in corpus})
    if unresolved:
        add(
            Check(
                "labels.unresolved",
                "Labels",
                "blocker",
                "Relevance labels point at missing documents",
                f"{len(unresolved)} label(s) reference documents not in the corpus.",
                "A label the retriever can never satisfy caps recall below 1.0 for reasons that "
                "have nothing to do with retrieval. The metric silently measures your data bug.",
                unresolved,
            )
        )
    elif items:
        passed(
            "labels.unresolved",
            "Labels",
            "Every label resolves",
            "All relevant documents exist in the corpus.",
            "The retriever can, in principle, achieve a perfect score.",
        )

    unanswerable = [i.id for i in items if not i.answerable]
    if unanswerable:
        add(
            Check(
                "labels.unanswerable",
                "Labels",
                "info",
                "Unanswerable questions",
                f"{len(unanswerable)} item(s) have no relevant documents and are excluded from "
                "retrieval metrics.",
                "Unanswerable questions are valuable for testing abstention in generation, but "
                "every retrieval metric is undefined without a relevant set. Scoring them as 0 "
                "would punish the retriever for the corpus not containing an answer.",
                unanswerable,
            )
        )

    bad_evidence, straddling, stray_evidence = [], [], []
    live_chunks = {d: chunk_text(doc.body) for d, doc in corpus.items()}
    for item in items:
        for ev in item.evidence:
            if ev.doc_id not in item.relevant:
                stray_evidence.append(f"{item.id}->{ev.doc_id}")
                continue
            doc = corpus.get(ev.doc_id)
            if doc is None:
                continue
            located = locate_evidence(doc.body, ev.text, live_chunks[ev.doc_id])
            if not located["found_in_doc"]:
                bad_evidence.append(f"{item.id}->{ev.doc_id}")
            elif located["straddles_boundary"]:
                straddling.append(f"{item.id}->{ev.doc_id}")
    if bad_evidence:
        add(
            Check(
                "labels.evidence_missing",
                "Labels",
                "blocker",
                "Evidence spans not found in their documents",
                f"{len(bad_evidence)} span(s) do not appear verbatim in the referenced document.",
                "Evidence drives chunk-level scoring: a chunk is relevant if it contains the span. "
                "A span that is not in the document can never match, so that question's chunk "
                "recall is zero by construction. Usually a paraphrase instead of a copy.",
                bad_evidence,
            )
        )
    if stray_evidence:
        add(
            Check(
                "labels.evidence_stray",
                "Labels",
                "warning",
                "Evidence on a document not marked relevant",
                f"{len(stray_evidence)} span(s) are ignored because their document has no grade.",
                "Evidence and relevance disagree — one of the two labels is wrong.",
                stray_evidence,
            )
        )
    if straddling:
        add(
            Check(
                "labels.evidence_straddles",
                "Labels",
                "warning",
                "Evidence split across a chunk boundary",
                f"{len(straddling)} span(s) exist in the document but in no single 512-token chunk.",
                "Chunk-level recall for these questions is capped below 1.0 by the chunker, not the "
                "retriever. That is a real finding about chunking — but it must not be read as a "
                "retrieval failure. Inspect them in Dataset Construction → Chunking.",
                straddling,
            )
        )
    missing_evidence = [i.id for i in items if i.answerable and not i.evidence]
    if missing_evidence:
        add(
            Check(
                "labels.no_evidence",
                "Labels",
                "info",
                "Items without evidence spans",
                f"{len(missing_evidence)} item(s) fall back to whole-document relevance at chunk level.",
                "Without a span, any chunk from a relevant document counts as relevant — including "
                "one that never mentions the answer. Chunk-level precision will read optimistic.",
                missing_evidence,
            )
        )

    graded = [g for i in items for g in i.relevant.values()]
    if graded and len(set(graded)) == 1:
        add(
            Check(
                "labels.binary_only",
                "Labels",
                "info",
                "All relevance grades are identical",
                f"Every judgement is grade {graded[0]}, so graded NDCG equals binary NDCG.",
                "Graded relevance distinguishes 'answers the question' (3) from 'related' (1). "
                "With one grade the extra resolution is unused — fine, but then do not read "
                "anything into graded vs binary NDCG.",
            )
        )

    # --- leakage ------------------------------------------------------------
    by_text: dict[str, list[GoldenItem]] = {}
    for item in items:
        by_text.setdefault(" ".join(sorted(_tokens(item.query))), []).append(item)
    exact = [grp for grp in by_text.values() if len(grp) > 1]
    cross_split_exact = [g for g in exact if len({i.split for i in g if i.split}) > 1]
    if cross_split_exact:
        add(
            Check(
                "leakage.duplicate_cross_split",
                "Leakage",
                "blocker",
                "Duplicate questions on both sides of the split",
                f"{len(cross_split_exact)} question(s) appear in dev AND holdout.",
                "Data leakage: tuning on the dev copy is tuning on the holdout. The holdout score "
                "will be optimistic by exactly the amount you overfit.",
                [", ".join(i.id for i in g) for g in cross_split_exact],
            )
        )
    elif exact:
        add(
            Check(
                "leakage.duplicate",
                "Leakage",
                "warning",
                "Duplicate questions",
                f"{len(exact)} group(s) of questions are the same up to word order.",
                "Duplicates double-count one observation, shrinking the confidence interval "
                "without adding evidence.",
                [", ".join(i.id for i in g) for g in exact],
            )
        )

    near, cross_near = [], []
    if len(items) <= MAX_PAIRWISE_ITEMS:
        toks = [(i, _tokens(i.query)) for i in items]
        for a in range(len(toks)):
            for b in range(a + 1, len(toks)):
                (ia, ta), (ib, tb) = toks[a], toks[b]
                if ta == tb:
                    continue
                if _jaccard(ta, tb) >= NEAR_DUPLICATE_JACCARD:
                    pair = f"{ia.id} ~ {ib.id}"
                    (cross_near if ia.split and ib.split and ia.split != ib.split else near).append(
                        pair
                    )
    if cross_near:
        add(
            Check(
                "leakage.near_duplicate_cross_split",
                "Leakage",
                "blocker",
                "Near-duplicate questions straddle the split",
                f"{len(cross_near)} pair(s) share ≥{int(NEAR_DUPLICATE_JACCARD * 100)}% of their words "
                "across dev/holdout.",
                "Paraphrase leakage is the quiet kind: no string matches exactly, but a fix tuned on "
                "one phrasing is tested on its twin. Pin both to the same split or rewrite one.",
                cross_near,
            )
        )
    if near:
        add(
            Check(
                "leakage.near_duplicate",
                "Leakage",
                "info",
                "Near-duplicate questions",
                f"{len(near)} pair(s) are near-paraphrases within one split.",
                "Fine for robustness testing, but they are not independent observations.",
                near,
            )
        )

    copied = []
    for item in items:
        q = norm_text(item.query).rstrip("?.!؟")
        if len(q.split()) >= 5:
            for d in item.relevant:
                if d in corpus and q in norm_text(corpus[d].body):
                    copied.append(f"{item.id}->{d}")
    if copied:
        add(
            Check(
                "leakage.query_copied",
                "Leakage",
                "warning",
                "Questions copied verbatim from their answer document",
                f"{len(copied)} question(s) appear word-for-word in the relevant document.",
                "Real users do not quote the document back at it. A verbatim query is solved by "
                "lexical overlap alone, so these items measure string matching and inflate hybrid "
                "search scores. Write questions the way a user would ask.",
                copied,
            )
        )

    split_cfg = manifest.split_config
    summary = summarize(items, split_cfg)
    if summary.straddling_groups and (split_cfg is None or split_cfg.group_by == "none"):
        add(
            Check(
                "leakage.group_straddle",
                "Leakage",
                "warning",
                "Questions about one document straddle the split",
                f"{len(summary.straddling_groups)} document(s) have questions in both dev and holdout.",
                "Unit of analysis: if you tune chunking or normalisation for a document using its dev "
                "questions, its holdout questions benefit. Regenerate the split with group_by="
                "primary_doc.",
                summary.straddling_groups,
            )
        )

    # --- coverage -----------------------------------------------------------
    answered_docs = {d for i in items for d in i.relevant}
    distractors = [d for d in corpus if d not in answered_docs]
    if corpus and not distractors:
        add(
            Check(
                "coverage.no_distractors",
                "Coverage",
                "warning",
                "No distractor documents",
                "Every document in the corpus is the answer to some question.",
                "With no distractors, a random ranking already does well and the eval cannot "
                "separate a good retriever from a lucky one. Add documents that are topically close "
                "but do not answer anything.",
            )
        )
    elif corpus:
        passed(
            "coverage.distractors",
            "Coverage",
            "Distractors present",
            f"{len(distractors)} of {len(corpus)} documents answer no question.",
            "Distractors are what make ranking a real task.",
        )

    for attr in ("difficulty", "trap", "lang"):
        counts = Counter(getattr(i, attr) for i in items if i.answerable)
        thin = sorted(f"{k}={v}" for k, v in counts.items() if v < MIN_PER_STRATUM)
        if len(counts) > 1 and thin:
            add(
                Check(
                    f"coverage.strata_{attr}",
                    "Coverage",
                    "warning",
                    f"Thin strata by {attr}",
                    f"{len(thin)} {attr} bucket(s) have fewer than {MIN_PER_STRATUM} questions.",
                    "Stratified reporting is how you find where a system fails — but a mean over "
                    "3 questions swings by 33 points on one item. Treat those buckets as anecdotes.",
                    thin,
                )
            )

    small_rel = [i.id for i in items if 0 < len(i.relevant) < k]
    if small_rel:
        add(
            Check(
                "coverage.precision_cap",
                "Coverage",
                "info",
                f"Precision@{k} is capped for most questions",
                f"{len(small_rel)} item(s) have fewer than {k} relevant documents.",
                f"With 1 relevant document, a perfect system scores precision@{k} = {1 / k:.2f}. "
                "Read precision next to recall, never alone.",
            )
        )

    if items:
        primary = Counter(i.primary_doc for i in items if i.answerable)
        heavy = [(d, c) for d, c in primary.items() if c / max(1, len(items)) > 0.3 and c > 2]
        if heavy:
            add(
                Check(
                    "coverage.concentration",
                    "Coverage",
                    "warning",
                    "Questions concentrate on few documents",
                    "; ".join(f"{d} answers {c} questions" for d, c in heavy),
                    "Questions sharing an answer document fail together, so they are not independent. "
                    "Use the cluster bootstrap interval in reports — it resamples documents, not "
                    "questions, and widens accordingly.",
                )
            )

    # --- split & power --------------------------------------------------------
    answerable = [i for i in items if i.answerable]
    dev_n = sum(1 for i in answerable if i.split == "dev")
    hold_n = sum(1 for i in answerable if i.split == "holdout")
    if items and summary.unassigned:
        add(
            Check(
                "split.unassigned",
                "Split",
                "blocker",
                "Split not assigned",
                f"{summary.unassigned} item(s) have no dev/holdout split.",
                "Experiments evaluate on a named split. Without one, every tuning run touches the "
                "data you would later quote — the holdout exists only if it is set aside first.",
            )
        )
    elif items:
        passed(
            "split.assigned",
            "Split",
            "Split assigned",
            f"{dev_n} dev · {hold_n} holdout answerable items.",
            "Tune on dev; quote holdout.",
        )

        imbalance = []
        for stratum, sides in summary.strata.items():
            total = sides["dev"] + sides["holdout"]
            if total >= 4:
                frac = sides["holdout"] / total
                target = split_cfg.holdout_fraction if split_cfg else 0.3
                if abs(frac - target) > 0.2:
                    imbalance.append(f"{stratum}: {sides['holdout']}/{total} holdout")
        if imbalance:
            add(
                Check(
                    "split.imbalance",
                    "Split",
                    "warning",
                    "Strata unevenly split",
                    f"{len(imbalance)} stratum(s) deviate >20 points from the target holdout fraction.",
                    "If holdout is mostly hard questions, a dev-vs-holdout gap is composition, not "
                    "overfitting. Stratify on the attribute that is imbalanced.",
                    imbalance,
                )
            )

    for name, n in (("dev", dev_n), ("holdout", hold_n)):
        if not items or summary.unassigned:
            break
        mde = detectable_effect(n, sd)
        severity = "warning" if (n < 10 or mde > 0.15) else "info"
        add(
            Check(
                f"power.{name}",
                "Power",
                severity,
                f"{name.capitalize()} split can detect effects of ±{mde:.2f}",
                f"n={n}, assumed per-query score sd={sd:.2f}, 95% confidence, 80% power.",
                "Statistical power: with this many paired questions, a true improvement smaller "
                f"than {mde:.2f} will usually produce a CI that crosses zero. 'No significant "
                f"difference' then means 'too small to see', not 'no effect'. Detecting 0.05 needs "
                f"~{min_sample_size(0.05, sd)} questions; 0.10 needs ~{min_sample_size(0.10, sd)}.",
            )
        )

    # --- index ------------------------------------------------------------------
    if not bundle.indexes:
        add(
            Check(
                "index.none",
                "Index",
                "warning",
                "Corpus not indexed",
                "Index the corpus in Dataset Construction before running retrieval.",
                "Indexing is asynchronous; scoring before it finishes yields zeros that look like "
                "retrieval failures but are races.",
            )
        )
    for norm, state in bundle.indexes.items():
        if state.corpus_hash != bundle.corpus_hash(norm):
            add(
                Check(
                    f"index.stale_{norm}",
                    "Index",
                    "blocker",
                    f"Index ({norm}) is stale",
                    "The corpus changed after this index was built.",
                    "Scores against a stale index measure the old corpus. Reindex — the new corpus "
                    "hash gets its own isolated owner namespace, so old and new never mix.",
                )
            )
            continue
        bad = {d: s.status for d, s in state.documents.items() if s.status != "INDEXED"}
        missing = [d for d in corpus if d not in state.documents]
        if bad or missing:
            add(
                Check(
                    f"index.incomplete_{norm}",
                    "Index",
                    "blocker",
                    f"Index ({norm}) incomplete",
                    f"{len(bad)} document(s) not INDEXED, {len(missing)} never uploaded.",
                    "A document that is not INDEXED is invisible to search, so every question it "
                    "answers scores zero for a reason unrelated to retrieval quality.",
                    [f"{d}: {s}" for d, s in sorted(bad.items())] + missing,
                )
            )
        else:
            passed(
                f"index.ok_{norm}",
                "Index",
                f"Index ({norm}) current",
                f"All {len(corpus)} documents INDEXED against corpus {state.corpus_hash[:12]}.",
                "Runs against this index are reproducible.",
            )

    # --- provenance & judgement depth --------------------------------------------
    synthetic = [i for i in items if i.origin == "synthetic"]
    if synthetic:
        share = len(synthetic) / len(items)
        in_holdout = [i.id for i in synthetic if i.split == "holdout"]
        add(
            Check(
                "labels.synthetic_share",
                "Labels",
                "warning" if (in_holdout or share > 0.5) else "info",
                f"{len(synthetic)} synthetic question(s) ({share:.0%} of the set)",
                f"{len(in_holdout)} of them are in the holdout.",
                "Generated questions echo their source text and are always answerable from one "
                "chunk, so they flatter lexical search and miss multi-hop and unanswerable cases. "
                "Fine for growing dev; keep the holdout human-written so the number you quote "
                "reflects real questions.",
                in_holdout,
            )
        )

    if items and not any(i.judged_nonrelevant for i in items):
        add(
            Check(
                "labels.no_negative_judgements",
                "Labels",
                "info",
                "No explicit 'not relevant' judgements",
                "Every document outside a question's relevant set is scored non-relevant by assumption.",
                "Unjudged is not the same as irrelevant. If a retriever surfaces a relevant document "
                "you never labelled, it is penalised for being right. Pool several retrievers and "
                "judge what they return (Dataset Construction → Pool & judge); run diagnostics then "
                "report judged@k.",
            )
        )

    if pool:
        view = pool_view(bundle, pool)
        if view.get("exists"):
            s = view["summary"]
            unjudged = s["pooled_pairs"] - s["judged_pairs"]
            add(
                Check(
                    "labels.pool_coverage",
                    "Labels",
                    "warning" if s["judged_share"] < 0.8 else "pass",
                    f"Pool {s['judged_share']:.0%} judged",
                    f"{unjudged} pooled document(s) unjudged across {s['items']} item(s); "
                    f"{s['relevant_never_pooled']} relevant document(s) were retrieved by no system "
                    f"within depth {pool.get('depth')}.",
                    "Pooled-but-unjudged documents are exactly where a better retriever is penalised "
                    "for finding something you missed. Judge them before comparing systems.",
                    [r["item_id"] for r in view["items"] if r["unjudged"]][:40],
                )
            )

    annotators = (annotations or {}).get("annotators", {})
    simulated_raters = set((annotations or {}).get("simulated", {}))
    if annotators:
        # A real annotator always outranks a simulated one for this check.
        real = {k: v for k, v in annotators.items() if k not in simulated_raters}
        name, entries = max((real or annotators).items(), key=lambda kv: len(kv[1]))
        is_simulated = name in simulated_raters
        if len(entries) >= 5:
            audit = agreement_report(bundle, annotations or {}, "gold", name)
            kw = audit["kappa_weighted"]
            add(
                Check(
                    "labels.agreement",
                    "Labels",
                    "info" if (kw is None or is_simulated) else "warning" if kw < 0.6 else "pass",
                    ("(simulated) " if is_simulated else "")
                    + f"Label agreement with '{name}': weighted κ "
                    + (
                        "undefined"
                        if kw is None
                        else f"{kw:.2f} ({audit['interpretation_weighted']})"
                    ),
                    f"{audit['pairs']} (question, document) pairs across {audit['items']} items; "
                    f"{len(audit['disagreements'])} disagreement(s).",
                    (
                        "These grades were SIMULATED from the gold labels, so this kappa measures the "
                        "simulation's noise setting and nothing about your labels. Run a real blind "
                        "batch to learn anything from it. "
                        if is_simulated
                        else ""
                    )
                    + "Agreement bounds what the labels can resolve. Below ~0.6 a small metric difference "
                    "between systems may be labelling noise. Adjudicate disagreements and tighten the "
                    "grading rubric before trusting close comparisons.",
                )
            )

    blockers = sum(1 for c in checks if c.severity == "blocker")
    warnings = sum(1 for c in checks if c.severity == "warning")
    verdict = "not_ready" if blockers else "ready_with_caveats" if warnings else "ready"

    rel_sizes = [len(i.relevant) for i in answerable]
    return {
        "verdict": verdict,
        "counts": dict(Counter(c.severity for c in checks)),
        "checks": [asdict(c) for c in checks],
        "stats": {
            "items": len(items),
            "answerable": len(answerable),
            "documents": len(corpus),
            "distractors": len(distractors),
            "dev": dev_n,
            "holdout": hold_n,
            "score_sd": sd,
            "score_sd_source": "run" if score_sd else "assumed",
            "relevant_set_size": {
                "mean": statistics.fmean(rel_sizes) if rel_sizes else 0.0,
                "max": max(rel_sizes) if rel_sizes else 0,
                "distribution": dict(sorted(Counter(rel_sizes).items())),
            },
            "by_difficulty": dict(Counter(i.difficulty for i in items)),
            "by_trap": dict(Counter(i.trap for i in items)),
            "by_lang": dict(Counter(i.lang for i in items)),
            "split": summary.as_dict(),
            "content_hash": current,
        },
    }
