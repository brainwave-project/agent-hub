"""Learning Path — missions that turn the lab's features into deliberate practice.

Each mission names the concept it exercises, why it matters, and concrete steps
in the lab. Two kinds of completion:

* **auto** — verified from lab state (a dataset with distractors exists, an
  experiment was registered with one variable changed, ...). You cannot tick
  these by hand; you have to do the thing.
* **reflection** — the result is an understanding, not an artefact (why a
  t-interval under-covers, what Holm fixes). Completing it requires writing
  what you observed and why, in your own words. Explaining is the test.
"""

from __future__ import annotations

from agent_hub.eval.lab.models import VariantSpec, now_iso
from agent_hub.eval.lab.pooling import pool_view
from agent_hub.eval.lab.readiness import assess
from agent_hub.eval.lab.splits import summarize
from agent_hub.eval.lab.store import LabStore

__all__ = ["MODULES", "evaluate", "update_progress", "MIN_REFLECTION_CHARS"]

MIN_REFLECTION_CHARS = 80

MODULES: list[dict] = [
    {
        "id": "construction",
        "title": "1 · Dataset construction",
        "missions": [
            {
                "id": "dc-scale",
                "check": "auto",
                "concept": "Corpus vs dataset · distractors",
                "title": "Build a dataset with real distractors",
                "why": "Without documents that answer nothing, every retriever looks good.",
                "steps": [
                    "Create a dataset (or fork the harness set)",
                    "Reach ≥ 10 documents, ≥ 3 of them distractors",
                    "Write ≥ 15 answerable questions",
                ],
                "tab": "/evaluation/dataset",
            },
            {
                "id": "dc-graded",
                "check": "auto",
                "concept": "Graded relevance · evidence spans",
                "title": "Label with grades and verbatim evidence",
                "why": "Graded NDCG and chunk-level scoring are only as good as these labels.",
                "steps": [
                    "Give ≥ 10 items an evidence span found in the document",
                    "Use at least two different grades across the set",
                ],
                "tab": "/evaluation/dataset",
            },
            {
                "id": "dc-pool",
                "check": "auto",
                "concept": "Pooling · unjudged ≠ irrelevant",
                "title": "Pool three retrievers and judge the pool",
                "why": "Your labels only cover documents you thought of; pooling finds the rest.",
                "steps": [
                    "Index the corpus",
                    "Run pooling (Golden Sets → Judge → Pool)",
                    "Judge every pooled document for ≥ 5 items",
                ],
                "tab": "/evaluation/dataset",
            },
            {
                "id": "dc-kappa",
                "check": "auto",
                "concept": "Inter-annotator agreement · κ",
                "title": "Audit your labels blind",
                "why": "A metric difference smaller than your labelling noise is not a finding.",
                "steps": [
                    "Open Judge → Audit",
                    "Annotate ≥ 10 items as a second annotator",
                    "Read κ, weighted κ, and every disagreement",
                ],
                "tab": "/evaluation/dataset",
            },
            {
                "id": "dc-synthetic",
                "check": "auto",
                "concept": "Synthetic data · leakage",
                "title": "Generate questions — and reject the leaky ones",
                "why": "Synthetic questions echo their source text and flatter lexical search.",
                "steps": [
                    "Generate candidates with the local LLM",
                    "Accept at least one good candidate",
                    "Reject at least one for high lexical overlap or bad evidence",
                ],
                "tab": "/evaluation/dataset",
            },
            {
                "id": "dc-versions",
                "check": "auto",
                "concept": "Data versioning & immutability",
                "title": "Freeze, fork, and read the diff",
                "why": "Every change to data changes what a number means.",
                "steps": [
                    "Freeze a version",
                    "Fork it and change something",
                    "Open Freeze → Diff and read the consequences",
                ],
                "tab": "/evaluation/dataset",
            },
        ],
    },
    {
        "id": "design",
        "title": "2 · Experiment design",
        "missions": [
            {
                "id": "ed-split",
                "check": "auto",
                "concept": "Dev / held-out set · leakage",
                "title": "Design a leakage-free split",
                "why": "Questions about one document on both sides of the split leak tuning into the holdout.",
                "steps": [
                    "Assign a split grouped by primary document",
                    "Holdout ≥ 25% and ≥ 5 answerable items",
                    "Zero documents straddling the split",
                ],
                "tab": "/evaluation/data-lab?tab=split",
            },
            {
                "id": "ed-ready",
                "check": "auto",
                "concept": "Data readiness",
                "title": "Reach zero blockers on a frozen version",
                "why": "Blockers are defects that make the number wrong, not just noisy.",
                "steps": ["Fix every blocker in Gate → Checks", "Freeze the version"],
                "tab": "/evaluation/data-lab?tab=checks",
            },
            {
                "id": "ed-register",
                "check": "auto",
                "concept": "Pre-registration · control group",
                "title": "Pre-register a one-variable experiment",
                "why": "Change two things and no result can be attributed to either.",
                "steps": [
                    "Write a hypothesis with a primary metric and minimum effect",
                    "Arms differ in exactly one field",
                    "Register",
                ],
                "tab": "/evaluation/experiments",
            },
            {
                "id": "ed-conclude",
                "check": "auto",
                "concept": "Confirmatory analysis",
                "title": "Run the holdout once and conclude",
                "why": "A decision is only honest when it comes from data you did not tune on.",
                "steps": [
                    "Iterate on dev",
                    "Evaluate on holdout",
                    "Conclude with a note on what you will do",
                ],
                "tab": "/evaluation/experiments",
            },
        ],
    },
    {
        "id": "metrics",
        "title": "3 · Retrieval metrics",
        "missions": [
            {
                "id": "rm-graded",
                "check": "auto",
                "concept": "Binary vs graded NDCG · IDCG",
                "title": "Find a query where graded and binary NDCG disagree",
                "why": "Binary relevance cannot tell 'answers it' from 'mentions it'.",
                "steps": [
                    "Grade documents 1–3 on a multi-document item",
                    "Run a split",
                    "Find a result where the two NDCGs differ by ≥ 0.05",
                ],
                "tab": "/evaluation/retrieval",
            },
            {
                "id": "rm-ksweep",
                "check": "reflection",
                "concept": "Hit rate saturation · precision/recall trade-off",
                "title": "Read a k-sweep",
                "why": "k is a decision about what the generator consumes.",
                "steps": [
                    "Open a run → Sweeps",
                    "Find the k where hit rate stops moving",
                    "Explain why precision falls while recall rises, and which k you would pick",
                ],
                "tab": "/evaluation/experiments",
            },
            {
                "id": "rm-threshold",
                "check": "reflection",
                "concept": "Similarity thresholds",
                "title": "Recreate the CRAG_MIN_SCORE trade-off",
                "why": "Too low keeps noise; too high starves retrieval.",
                "steps": [
                    "Open a run → Sweeps → threshold",
                    "Find where empty results start",
                    "Explain the two failure modes",
                ],
                "tab": "/evaluation/experiments",
            },
            {
                "id": "rm-failures",
                "check": "auto",
                "concept": "Error analysis",
                "title": "Tag five failures",
                "why": "A mean says THAT it fails; error analysis says why.",
                "steps": ["Open a run → Failures", "Tag ≥ 5 failed queries with a cause"],
                "tab": "/evaluation/experiments",
            },
        ],
    },
    {
        "id": "statistics",
        "title": "4 · Statistical reasoning",
        "missions": [
            {
                "id": "st-bootstrap",
                "check": "reflection",
                "concept": "Resampling · bootstrap distribution",
                "title": "Look inside a bootstrap interval",
                "why": "An interval is two percentiles of a distribution you usually never see.",
                "steps": [
                    "Stats Lab → Bootstrap on a real run",
                    "Compare bootstrap, t and normal intervals",
                    "Explain why the bootstrap histogram is lumpy on binary scores",
                ],
                "tab": "/evaluation/stats",
            },
            {
                "id": "st-coverage",
                "check": "reflection",
                "concept": "Confidence level vs interval · t for small n",
                "title": "Break a 95% interval",
                "why": "'95%' is a property of the procedure, and it can fail.",
                "steps": [
                    "Stats Lab → Coverage, bimodal population, n = 10",
                    "Find the method whose coverage is furthest below 95%",
                    "Explain why, and what changes at n = 100",
                ],
                "tab": "/evaluation/stats",
            },
            {
                "id": "st-clt",
                "check": "reflection",
                "concept": "Central Limit Theorem",
                "title": "Watch the CLT arrive",
                "why": "Normal-theory intervals are only as good as the CLT at your n.",
                "steps": [
                    "Stats Lab → CLT on a skewed population",
                    "Compare skew at n = 5 vs 100",
                    "Explain when the normal approximation becomes safe",
                ],
                "tab": "/evaluation/stats",
            },
            {
                "id": "st-power",
                "check": "reflection",
                "concept": "Statistical power · unit of analysis",
                "title": "Size an experiment before running it",
                "why": "An underpowered 'no difference' is not evidence of no difference.",
                "steps": [
                    "Stats Lab → Power with your minimum effect",
                    "Find n for 80% power",
                    "Compare empirical and analytic power and explain any gap",
                ],
                "tab": "/evaluation/stats",
            },
            {
                "id": "st-phacking",
                "check": "reflection",
                "concept": "P-hacking · selective reporting · multiple comparisons",
                "title": "Manufacture significance from nothing",
                "why": "Seeing false positives appear from pure noise is the best inoculation.",
                "steps": [
                    "Stats Lab → P-hacking with 7 metrics",
                    "Note raw vs Holm family-wise error",
                    "Turn on peeking and explain why stopping early inflates false positives",
                ],
                "tab": "/evaluation/stats",
            },
            {
                "id": "st-cluster",
                "check": "reflection",
                "concept": "Normal vs cluster bootstrap",
                "title": "Explain a cluster interval",
                "why": "Questions that share a document are not independent observations.",
                "steps": [
                    "Open an experiment report",
                    "Compare the bootstrap and cluster CI of the primary metric",
                    "Explain which one you would quote and why",
                ],
                "tab": "/evaluation/experiments",
            },
        ],
    },
]


def _variant_diff(a: VariantSpec, b: VariantSpec) -> list[str]:
    return [f for f in VariantSpec.model_fields if f != "name" and getattr(a, f) != getattr(b, f)]


def _auto_checks(store: LabStore) -> dict[str, tuple[bool, str]]:
    results: dict[str, tuple[bool, str]] = {}
    bundles = []
    for ds in store.list_datasets():
        for v in ds["versions"]:
            bundles.append(store.bundle(ds["id"], v["version"]))

    # dc-scale
    rows = []
    for b in bundles:
        answered = {d for i in b.items for d in i.relevant}
        rows.append(
            (
                b,
                len(b.corpus),
                len([d for d in b.corpus if d not in answered]),
                sum(1 for i in b.items if i.answerable),
            )
        )
    ok = [r for r in rows if r[1] >= 10 and r[2] >= 3 and r[3] >= 15]
    if ok:
        results["dc-scale"] = (True, f"{ok[0][0].meta.id} v{ok[0][0].manifest.version}")
    elif rows:
        top = max(rows, key=lambda r: (r[2], r[3], r[1]))
        results["dc-scale"] = (
            False,
            f"best: {top[0].meta.id} v{top[0].manifest.version} — {top[1]} docs, {top[2]} distractors, {top[3]} answerable",
        )
    else:
        results["dc-scale"] = (False, "no dataset yet")

    # dc-graded
    graded = []
    for b in bundles:
        with_ev = sum(
            1
            for i in b.items
            if i.evidence
            and all(
                e.text.split()
                and " ".join(e.text.split()).casefold()
                in " ".join(b.corpus[e.doc_id].body.split()).casefold()
                for e in i.evidence
                if e.doc_id in b.corpus
            )
        )
        grades = {g for i in b.items for g in i.relevant.values()}
        graded.append((b, with_ev, len(grades)))
    ok = [g for g in graded if g[1] >= 10 and g[2] >= 2]
    results["dc-graded"] = (
        (True, f"{ok[0][0].meta.id} v{ok[0][0].manifest.version}")
        if ok
        else (
            False,
            f"best: {max(graded, key=lambda g: (g[1], g[2]))[1]} items with found evidence, "
            f"{max(graded, key=lambda g: (g[1], g[2]))[2]} distinct grades"
            if graded
            else "no dataset yet",
        )
    )

    # dc-pool
    pools = []
    for b in bundles:
        view = pool_view(b, store.artifact(b.meta.id, b.manifest.version, "pool"))
        if view.get("exists"):
            pools.append((b, view["summary"]["items_fully_judged"]))
    ok = [p for p in pools if p[1] >= 5]
    results["dc-pool"] = (
        (True, f"{ok[0][0].meta.id} v{ok[0][0].manifest.version}: {ok[0][1]} items fully judged")
        if ok
        else (
            False,
            f"best pool: {max(p[1] for p in pools)} items fully judged"
            if pools
            else "no pool built yet",
        )
    )

    # dc-kappa
    counts = []
    simulated_seen = 0
    for b in bundles:
        ann = store.artifact(b.meta.id, b.manifest.version, "annotations", {}) or {}
        simulated = set(ann.get("simulated", {}))
        for name, entries in ann.get("annotators", {}).items():
            # Simulated raters never satisfy this mission - the judging is the point.
            if name in simulated:
                simulated_seen += 1
                continue
            counts.append((b, name, len(entries)))
    ok = [c for c in counts if c[2] >= 10]
    sim_note = (
        f" · {simulated_seen} simulated annotator(s) ignored - this mission needs real judging"
        if simulated_seen
        else ""
    )
    results["dc-kappa"] = (
        (True, f"{ok[0][1]} annotated {ok[0][2]} items")
        if ok
        else (
            False,
            (f"most: {max(c[2] for c in counts)} items annotated" if counts else "no annotations yet")
            + sim_note,
        )
    )

    # dc-synthetic
    acc = rej = 0
    for b in bundles:
        for c in store.artifact(b.meta.id, b.manifest.version, "candidates", []) or []:
            acc += c.get("status") == "accepted"
            rej += c.get("status") == "rejected"
    results["dc-synthetic"] = (acc >= 1 and rej >= 1, f"{acc} accepted · {rej} rejected")

    # dc-versions
    versioned = [
        ds
        for ds in store.list_datasets()
        if len(ds["versions"]) >= 2 and any(v["status"] == "frozen" for v in ds["versions"])
    ]
    results["dc-versions"] = (
        bool(versioned),
        versioned[0]["id"] if versioned else "need ≥ 2 versions, one frozen",
    )

    # ed-split
    split_rows = []
    for b in bundles:
        cfg = b.manifest.split_config
        summary = summarize(b.items, cfg)
        answerable_hold = sum(1 for i in b.items if i.split == "holdout" and i.answerable)
        total = summary.dev + summary.holdout
        frac = summary.holdout / total if total else 0.0
        good = (
            cfg is not None
            and cfg.group_by == "primary_doc"
            and summary.unassigned == 0
            and frac >= 0.25
            and answerable_hold >= 5
            and not summary.straddling_groups
        )
        split_rows.append((b, good, frac, answerable_hold, len(summary.straddling_groups)))
    ok = [s for s in split_rows if s[1]]
    results["ed-split"] = (
        (True, f"{ok[0][0].meta.id} v{ok[0][0].manifest.version}")
        if ok
        else (
            False,
            "best: holdout {:.0%}, {} answerable, {} straddling".format(
                *max(split_rows, key=lambda s: (s[3], s[2]))[2:]
            )
            if split_rows
            else "no dataset yet",
        )
    )

    # ed-ready
    ready = []
    for b in bundles:
        if b.manifest.status != "frozen":
            continue
        report = assess(b)
        if not report["counts"].get("blocker"):
            ready.append(b)
    results["ed-ready"] = (
        bool(ready),
        f"{ready[0].meta.id} v{ready[0].manifest.version}"
        if ready
        else "no frozen version without blockers",
    )

    experiments = store.list_experiments()
    one_var = [
        e
        for e in experiments
        if e.status in ("registered", "concluded")
        and len(_variant_diff(e.baseline, e.treatment)) == 1
    ]
    results["ed-register"] = (
        bool(one_var),
        one_var[0].id if one_var else "no registered one-variable experiment",
    )
    concluded = [e for e in experiments if e.status == "concluded"]
    results["ed-conclude"] = (
        bool(concluded),
        concluded[0].id if concluded else "no concluded experiment",
    )

    # rm-graded
    found = None
    for summary in store.list_runs():
        run = store.get_run(summary["id"])
        for r in run.results:
            if (
                "graded_ndcg@k" in r.metrics
                and abs(r.metrics["graded_ndcg@k"] - r.metrics["ndcg@k"]) >= 0.05
            ):
                found = f"{run.id} · {r.item_id}"
                break
        if found:
            break
    results["rm-graded"] = (bool(found), found or "no result with |graded − binary| ≥ 0.05 yet")

    tagged = max(
        (
            sum(1 for v in ann.values() if v.get("tags"))
            for ann in store.all_run_annotations().values()
        ),
        default=0,
    )
    results["rm-failures"] = (tagged >= 5, f"{tagged} failure(s) tagged in one run")
    return results


def evaluate(store: LabStore) -> dict:
    progress = store.learning_progress()
    auto = _auto_checks(store)
    modules = []
    done = total = 0
    for module in MODULES:
        missions = []
        for m in module["missions"]:
            state = progress.get(m["id"], {})
            if m["check"] == "auto":
                passed, evidence = auto.get(m["id"], (False, ""))
                status = "done" if passed else "todo"
            else:
                status = "done" if state.get("done") else "todo"
                evidence = ""
            total += 1
            done += status == "done"
            missions.append(
                {
                    **m,
                    "status": status,
                    "evidence": evidence,
                    "reflection": state.get("reflection", ""),
                    "completed_at": state.get("completed_at"),
                }
            )
        modules.append(
            {
                "id": module["id"],
                "title": module["title"],
                "missions": missions,
                "done": sum(1 for x in missions if x["status"] == "done"),
            }
        )
    return {
        "modules": modules,
        "done": done,
        "total": total,
        "share": done / total if total else 0.0,
        "min_reflection_chars": MIN_REFLECTION_CHARS,
    }


def update_progress(store: LabStore, mission_id: str, done: bool, reflection: str) -> dict:
    mission = next((m for mod in MODULES for m in mod["missions"] if m["id"] == mission_id), None)
    if mission is None:
        raise KeyError(mission_id)
    progress = store.learning_progress()
    entry = progress.get(mission_id, {})
    entry["reflection"] = reflection
    if mission["check"] == "auto":
        entry["done"] = False  # auto missions are verified, never self-reported
    else:
        if done and len(reflection.strip()) < MIN_REFLECTION_CHARS:
            raise ValueError(
                f"write at least {MIN_REFLECTION_CHARS} characters explaining what you observed and why"
            )
        entry["done"] = done
        entry["completed_at"] = now_iso() if done else None
    progress[mission_id] = entry
    store.save_learning_progress(progress)
    return entry
