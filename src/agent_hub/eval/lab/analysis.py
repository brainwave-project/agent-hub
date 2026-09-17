"""Experiment reports — intervals, paired deltas, and pre-registered decisions.

What this module refuses to do is as important as what it does:

* It never reports a mean without an interval.
* It never compares unpaired: deltas are per question, over the questions both
  arms scored.
* It never picks the best run. The report is built from the LATEST run of each
  arm on the requested split, and it lists every run the experiment ever made,
  so a favourable run cannot be quietly selected.
* It never calls a dev-set result a decision. Only a registered experiment on
  the holdout split produces a confirmatory verdict; everything else is
  labelled exploratory.
* The decision rule is the one registered BEFORE the data was seen: the
  primary metric, its direction, and the minimum effect worth acting on.
"""

from __future__ import annotations

import math
import statistics

from agent_hub.eval.lab.models import LAB_METRICS, Experiment, GoldenItem, Run, SuccessCriteria
from agent_hub.eval.stats import (
    bootstrap_means,
    cluster_mean_ci,
    cohens_dz,
    detectable_effect,
    holm_adjust,
    mean_ci,
    min_sample_size,
    paired_cluster_delta_ci,
    paired_delta_ci,
    paired_permutation_p,
    paired_t_p,
    t_interval,
)

__all__ = ["arm_summary", "paired_comparison", "decide", "experiment_report", "compare_runs"]

ARM_RESAMPLES = 2000
DELTA_RESAMPLES = 5000


def _finite(x: float) -> float | None:
    return x if math.isfinite(x) else None


def _scores(run: Run, metric: str) -> dict[str, float]:
    return {
        r.item_id: r.metrics[metric] for r in run.results if r.error is None and metric in r.metrics
    }


def arm_summary(
    run: Run, metric: str, clusters: dict[str, str], confidence: float, ids: list[str]
) -> dict:
    by_id = _scores(run, metric)
    values = [by_id[i] for i in ids]
    if not values:
        return {"n": 0}
    boot = mean_ci(values, confidence, ARM_RESAMPLES)
    t = t_interval(values, confidence)
    cluster = cluster_mean_ci(values, [clusters.get(i, i) for i in ids], confidence, ARM_RESAMPLES)
    return {
        "n": len(values),
        "mean": boot.mean,
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "bootstrap": {"low": _finite(boot.low), "high": _finite(boot.high)},
        "t": {"low": _finite(t.low), "high": _finite(t.high)},
        "cluster": {
            "low": _finite(cluster.low),
            "high": _finite(cluster.high),
            "clusters": cluster.n,
        },
        "share_zero": sum(1 for v in values if v == 0.0) / len(values),
        "share_one": sum(1 for v in values if v == 1.0) / len(values),
    }


def paired_comparison(
    run_a: Run, run_b: Run, metric: str, clusters: dict[str, str], confidence: float
) -> dict:
    a, b = _scores(run_a, metric), _scores(run_b, metric)
    ids = sorted(set(a) & set(b))
    base = [a[i] for i in ids]
    treat = [b[i] for i in ids]
    delta = paired_delta_ci(base, treat, confidence, DELTA_RESAMPLES)
    cluster = paired_cluster_delta_ci(
        base, treat, [clusters.get(i, i) for i in ids], confidence, DELTA_RESAMPLES
    )
    diffs = [t - s for s, t in zip(base, treat, strict=True)]
    return {
        "n": len(ids),
        "dropped": sorted((set(a) | set(b)) - set(ids)),
        "mean": delta.mean,
        "low": _finite(delta.low),
        "high": _finite(delta.high),
        "significant": delta.significant,
        "cluster": {
            "low": _finite(cluster.low),
            "high": _finite(cluster.high),
            "clusters": cluster.n,
            "significant": cluster.significant,
        },
        "p_permutation": paired_permutation_p(base, treat, DELTA_RESAMPLES) if ids else 1.0,
        "p_ttest": paired_t_p(base, treat) if len(ids) > 1 else 1.0,
        "effect_size_dz": cohens_dz(base, treat) if len(ids) > 1 else 0.0,
        "sd_delta": statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
        "wins": sum(1 for d in diffs if d > 0),
        "losses": sum(1 for d in diffs if d < 0),
        "ties": sum(1 for d in diffs if d == 0),
        "per_query": [
            {"item_id": i, "baseline": s, "treatment": t, "delta": t - s}
            for i, s, t in zip(ids, base, treat, strict=True)
        ],
    }


def decide(delta: dict, criteria: SuccessCriteria) -> dict:
    """Apply the pre-registered rule to one paired delta.

    Two thresholds, deliberately separate: ZERO answers "is there an effect?"
    and `min_effect` answers "is it big enough to matter?". A CI that excludes
    zero but sits entirely below min_effect is a real effect nobody should ship
    for; a CI that spans both is not evidence of absence, it is an underpowered
    experiment.
    """
    m = criteria.min_effect
    mean, low, high = delta["mean"], delta["low"], delta["high"]
    if low is None or high is None:
        return {
            "verdict": "inconclusive",
            "headline": "Not enough paired questions to form an interval.",
            "reasons": [],
        }

    if criteria.direction == "decrease":
        mean, low, high = -mean, -high, -low

    if criteria.direction == "non_inferior":
        if low > -m:
            return {
                "verdict": "success",
                "headline": f"Non-inferior: the CI lower bound {low:+.3f} stays above the margin −{m:.3f}.",
                "reasons": ["Treatment is not meaningfully worse than baseline."],
            }
        if high < -m:
            return {
                "verdict": "harm",
                "headline": f"Inferior: the entire CI sits below the margin −{m:.3f}.",
                "reasons": [],
            }
        return {
            "verdict": "inconclusive",
            "headline": f"The CI [{low:+.3f}, {high:+.3f}] crosses the non-inferiority margin −{m:.3f}.",
            "reasons": ["Cannot rule out a loss larger than the margin."],
        }

    if low > 0 and mean >= m:
        return {
            "verdict": "success",
            "headline": f"Effect detected ({mean:+.3f}, CI [{low:+.3f}, {high:+.3f}]) and it clears the "
            f"minimum effect {m:.3f}.",
            "reasons": [
                "The CI excludes zero.",
                "The point estimate meets the pre-registered minimum.",
            ],
        }
    if low > 0:
        return {
            "verdict": "detectable_but_small",
            "headline": f"Real but small: CI excludes zero, yet {mean:+.3f} is below the minimum effect {m:.3f}.",
            "reasons": ["Statistically detectable is not the same as worth shipping."],
        }
    if high < 0:
        return {
            "verdict": "harm",
            "headline": f"Regression: the entire CI [{low:+.3f}, {high:+.3f}] is below zero.",
            "reasons": [],
        }
    if high < m:
        return {
            "verdict": "no_meaningful_effect",
            "headline": f"No meaningful effect: the CI [{low:+.3f}, {high:+.3f}] rules out an improvement "
            f"as large as {m:.3f}.",
            "reasons": [
                "Whatever the true effect is, it is smaller than the one you said matters."
            ],
        }
    return {
        "verdict": "inconclusive",
        "headline": f"Inconclusive: the CI [{low:+.3f}, {high:+.3f}] contains both zero and the minimum "
        f"effect {m:.3f}.",
        "reasons": ["The experiment is underpowered for the effect it was designed to detect."],
    }


def _latest(runs: list[Run], arm: str, split: str) -> Run | None:
    candidates = [r for r in runs if r.arm == arm and r.split == split and r.status == "done"]
    return max(candidates, key=lambda r: r.created_at) if candidates else None


def _strata(items: dict[str, GoldenItem], per_query: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for attr in ("difficulty", "trap", "lang"):
        buckets: dict[str, list[dict]] = {}
        for row in per_query:
            item = items.get(row["item_id"])
            if item is not None:
                buckets.setdefault(str(getattr(item, attr)), []).append(row)
        if len(buckets) < 2:
            continue
        out[attr] = {
            key: {
                "n": len(rows),
                "baseline": statistics.fmean(r["baseline"] for r in rows),
                "treatment": statistics.fmean(r["treatment"] for r in rows),
                "delta": statistics.fmean(r["delta"] for r in rows),
            }
            for key, rows in sorted(buckets.items())
        }
    return out


def _metric_role(metric: str, experiment: Experiment) -> str:
    if metric == experiment.criteria.primary_metric:
        return "primary"
    if any(g.metric == metric for g in experiment.criteria.guardrails):
        return "guardrail"
    if metric in experiment.secondary_metrics:
        return "secondary"
    return "exploratory"


def _comparison_block(
    baseline: Run,
    treatment: Run,
    items: dict[str, GoldenItem],
    confidence: float,
    metrics: tuple[str, ...] = LAB_METRICS,
) -> dict:
    clusters = {i.id: i.primary_doc for i in items.values()}
    block = {}
    for metric in metrics:
        delta = paired_comparison(baseline, treatment, metric, clusters, confidence)
        ids = [row["item_id"] for row in delta["per_query"]]
        block[metric] = {
            "baseline": arm_summary(baseline, metric, clusters, confidence, ids),
            "treatment": arm_summary(treatment, metric, clusters, confidence, ids),
            "delta": delta,
        }
    return block


def experiment_report(
    experiment: Experiment,
    runs: list[Run],
    items: list[GoldenItem],
    current_hash: str,
    split: str,
) -> dict:
    criteria = experiment.criteria
    item_map = {i.id: i for i in items}
    baseline = _latest(runs, "baseline", split)
    treatment = _latest(runs, "treatment", split)

    all_runs = [
        {
            "id": r.id,
            "arm": r.arm,
            "split": r.split,
            "status": r.status,
            "created_at": r.created_at,
            "dataset_hash": r.dataset_hash,
            "primary_mean": r.aggregates.get(criteria.primary_metric, {}).get("mean"),
            "used_in_report": r is baseline or r is treatment,
        }
        for r in sorted(runs, key=lambda r: r.created_at)
    ]
    report: dict = {
        "experiment_id": experiment.id,
        "split": split,
        "criteria": criteria.model_dump(),
        "all_runs": all_runs,
        "holdout_access": [h.model_dump() for h in experiment.holdout_access],
        "warnings": [],
    }
    if baseline is None or treatment is None:
        report["status"] = "missing_runs"
        report["missing"] = [
            a for a, r in (("baseline", baseline), ("treatment", treatment)) if r is None
        ]
        return report

    warnings: list[dict] = report["warnings"]
    confirmatory = (
        split == "holdout"
        and experiment.status in ("registered", "concluded")
        and baseline.dataset_status == "frozen"
        and baseline.dataset_hash == treatment.dataset_hash == experiment.dataset_hash
    )

    if split == "dev":
        warnings.append(
            {
                "severity": "info",
                "title": "Exploratory result",
                "detail": "Dev-set numbers guide iteration. They are not a decision — you have "
                "(implicitly) tuned against these questions.",
            }
        )
    if baseline.dataset_hash != treatment.dataset_hash:
        warnings.append(
            {
                "severity": "blocker",
                "title": "Arms scored on different dataset content",
                "detail": "The two runs have different dataset hashes. The comparison is not paired "
                "over the same questions and labels.",
            }
        )
    if experiment.dataset_hash and baseline.dataset_hash != experiment.dataset_hash:
        warnings.append(
            {
                "severity": "blocker",
                "title": "Dataset changed since registration",
                "detail": "The runs do not match the dataset hash recorded at registration.",
            }
        )
    if baseline.dataset_hash != current_hash:
        warnings.append(
            {
                "severity": "warning",
                "title": "Runs predate the current dataset content",
                "detail": "The dataset has changed since these runs. Re-run to report on it.",
            }
        )
    if experiment.status == "draft":
        warnings.append(
            {
                "severity": "warning",
                "title": "Experiment not registered",
                "detail": "Without pre-registration, the hypothesis and success criteria could have "
                "been chosen after seeing the result.",
            }
        )
    holdout_runs = sum(1 for r in runs if r.split == "holdout")
    if split == "holdout" and len(experiment.holdout_access) > 1:
        warnings.append(
            {
                "severity": "warning",
                "title": f"Holdout evaluated {len(experiment.holdout_access)} times",
                "detail": "Every look at the holdout that informs a change turns it into a dev set. "
                "The latest number is optimistic by an unknown amount.",
            }
        )
    if criteria.primary_metric not in LAB_METRICS:
        warnings.append(
            {
                "severity": "blocker",
                "title": "Unknown primary metric",
                "detail": criteria.primary_metric,
            }
        )
        report["status"] = "invalid"
        return report
    for run in (baseline, treatment):
        errors = [r.item_id for r in run.results if r.error]
        if errors:
            warnings.append(
                {
                    "severity": "warning",
                    "title": f"{len(errors)} errored queries in {run.arm}",
                    "detail": "Excluded from both arms (pairing), not scored as zero: "
                    + ", ".join(errors[:10]),
                }
            )

    block = _comparison_block(baseline, treatment, item_map, criteria.confidence)
    primary = block[criteria.primary_metric]
    delta = primary["delta"]
    decision = decide(delta, criteria)

    guardrails = []
    for guard in criteria.guardrails:
        if guard.metric not in block:
            continue
        g_low = block[guard.metric]["delta"]["low"]
        ok = g_low is not None and g_low >= -guard.max_drop
        guardrails.append(
            {
                "metric": guard.metric,
                "max_drop": guard.max_drop,
                "delta": block[guard.metric]["delta"]["mean"],
                "low": g_low,
                "passed": ok,
            }
        )
    if decision["verdict"] == "success" and not all(g["passed"] for g in guardrails):
        decision = {
            **decision,
            "verdict": "guardrail_breach",
            "headline": "Primary metric improved, but a guardrail metric may have regressed beyond "
            "its tolerance.",
            "reasons": [
                f"{g['metric']}: CI lower bound {g['low']} < −{g['max_drop']}"
                for g in guardrails
                if not g["passed"]
            ],
        }

    n = delta["n"]
    sd_delta = delta["sd_delta"] or 0.0
    mde = detectable_effect(n, sd_delta, criteria.confidence) if sd_delta > 0 else None
    if n < 30:
        warnings.append(
            {
                "severity": "info",
                "title": f"Small sample (n={n})",
                "detail": "Intervals are wide and the bootstrap itself is coarse at this size.",
            }
        )
    base_mean = primary["baseline"].get("mean", 0.0)
    if criteria.direction == "increase" and base_mean >= 0.95:
        warnings.append(
            {
                "severity": "warning",
                "title": "Ceiling effect",
                "detail": f"Baseline {criteria.primary_metric} is already {base_mean:.2f}; there is "
                "almost no room to show an improvement. Pick a harder set or a less "
                "saturated metric.",
            }
        )
    tested = 1 + len(experiment.secondary_metrics)
    if tested > 1:
        family = 1 - criteria.confidence**tested
        warnings.append(
            {
                "severity": "info",
                "title": "Multiple comparisons",
                "detail": f"{tested} metrics tested at {criteria.confidence:.0%} each: even with no real "
                f"effect, P(at least one 'significant') ≈ {family:.0%}. Only the primary "
                "metric decides.",
            }
        )

    # Holm over exactly the metrics that were registered as tested: the primary
    # plus declared secondaries. Exploratory metrics are not a family you chose.
    tested = [criteria.primary_metric, *[m for m in experiment.secondary_metrics if m in block]]
    for metric, p_adjusted in holm_adjust(
        {m: block[m]["delta"]["p_permutation"] for m in tested}
    ).items():
        block[metric]["delta"]["p_holm"] = p_adjusted
    from agent_hub.eval.lab.simulate import histogram  # leaf helper; keeps import graph flat

    primary_boot = bootstrap_means([row["delta"] for row in delta["per_query"]], DELTA_RESAMPLES)

    report.update(
        {
            "status": "ok",
            "primary_bootstrap": histogram(primary_boot, 40) if primary_boot else [],
            "confirmatory": confirmatory,
            "holdout_runs": holdout_runs,
            "baseline_run": baseline.id,
            "treatment_run": treatment.id,
            "baseline_variant": baseline.variant.model_dump(),
            "treatment_variant": treatment.variant.model_dump(),
            "decision": decision,
            "guardrails": guardrails,
            "metrics": {m: {**v, "role": _metric_role(m, experiment)} for m, v in block.items()},
            "strata": _strata(item_map, delta["per_query"]),
            "per_query": [
                {
                    **row,
                    "query": item_map[row["item_id"]].query if row["item_id"] in item_map else "",
                    "difficulty": getattr(item_map.get(row["item_id"]), "difficulty", ""),
                    "trap": getattr(item_map.get(row["item_id"]), "trap", ""),
                }
                for row in delta["per_query"]
            ],
            "power": {
                "n": n,
                "sd_delta": sd_delta,
                "mde": mde,
                "n_needed_for_min_effect": min_sample_size(criteria.min_effect, sd_delta)
                if sd_delta > 0
                else None,
            },
        }
    )
    return report


def compare_runs(run_a: Run, run_b: Run, items: list[GoldenItem], confidence: float = 0.95) -> dict:
    """Ad-hoc paired comparison of any two runs — always exploratory."""
    warnings = [
        {
            "severity": "info",
            "title": "Exploratory comparison",
            "detail": "Not pre-registered. Useful for iteration; not a basis for a claim.",
        }
    ]
    if run_a.dataset_hash != run_b.dataset_hash:
        warnings.append(
            {
                "severity": "blocker",
                "title": "Different dataset content",
                "detail": "These runs were scored against different dataset hashes; per-question "
                "pairing is not meaningful.",
            }
        )
    if run_a.split != run_b.split:
        warnings.append(
            {
                "severity": "warning",
                "title": "Different splits",
                "detail": f"{run_a.split} vs {run_b.split}: only overlapping questions are paired.",
            }
        )
    if run_a.k != run_b.k:
        warnings.append(
            {
                "severity": "warning",
                "title": "Different k",
                "detail": f"k={run_a.k} vs k={run_b.k}: set metrics are not comparable.",
            }
        )
    item_map = {i.id: i for i in items}
    return {
        "run_a": run_a.id,
        "run_b": run_b.id,
        "variant_a": run_a.variant.model_dump(),
        "variant_b": run_b.variant.model_dump(),
        "warnings": warnings,
        "metrics": _comparison_block(run_a, run_b, item_map, confidence),
    }
