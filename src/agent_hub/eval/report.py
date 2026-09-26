"""Rendering eval results — console table plus a machine-readable JSON record.

Design rule: never print a bare mean. Every aggregate carries a confidence
interval, and every comparison is a paired delta whose interval either excludes
zero or does not. A number without a CI invites exactly the mistake this harness
exists to prevent — acting on a difference that is indistinguishable from noise.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from agent_hub.eval.datasets import EVAL_DIR, GoldenRecord
from agent_hub.eval.retrieval import STANDARD_COMPARISONS, RunResult
from agent_hub.eval.stats import mean_ci, min_sample_size, paired_delta_ci

REPORTS_DIR = EVAL_DIR / "reports"

#: Metrics shown in the console summary. The full set still goes to JSON.
HEADLINE = ("recall@k", "mrr", "ndcg@k")

_RULE = "-" * 76


def _fmt_pct(value: float) -> str:
    return f"{value:.3f}"


def render_variants(runs: dict[str, RunResult]) -> str:
    """One row per retrieval configuration, with bootstrap CIs."""
    lines = [_RULE, "RETRIEVAL — per configuration (95% bootstrap CI)", _RULE]
    header = f"{'variant':<28}" + "".join(f"{m:>16}" for m in HEADLINE)
    lines.append(header)
    lines.append("-" * len(header))

    for name, run in runs.items():
        cells = []
        for metric in HEADLINE:
            interval = mean_ci(run.metric_scores(metric))
            cells.append(f"{interval.mean:.3f}".rjust(16))
        lines.append(f"{name:<28}" + "".join(cells))

    first = next(iter(runs.values()), None)
    if first is not None:
        lines.append("")
        lines.append(f"  k={first.k}   n={len(first.record_ids)} questions per variant")
    return "\n".join(lines)


def render_comparisons(runs: dict[str, RunResult], metric: str = "recall@k") -> str:
    """Paired deltas — the part that actually supports a decision."""
    lines = [
        "",
        _RULE,
        f"COMPARISONS — paired delta on {metric} (95% CI)",
        _RULE,
    ]
    for baseline_name, variant_name, question in STANDARD_COMPARISONS:
        base = runs.get(baseline_name)
        variant = runs.get(variant_name)
        if base is None or variant is None:
            continue
        if base.record_ids != variant.record_ids:
            lines.append(f"  {question}\n    SKIPPED — variants ran different question sets")
            continue

        delta = paired_delta_ci(base.metric_scores(metric), variant.metric_scores(metric))
        verdict = {
            "improvement": "IMPROVED",
            "REGRESSION": "REGRESSED",
            "no change": "no significant change",
        }[delta.direction]
        lines.append(f"  {question}")
        lines.append(f"    {baseline_name} -> {variant_name}")
        lines.append(f"    {delta}")
        lines.append(f"    -> {verdict}")
        lines.append("")
    return "\n".join(lines)


def render_strata(
    run: RunResult,
    records: Sequence[GoldenRecord],
    metric: str = "recall@k",
    attr: str = "trap",
) -> str:
    """Per-stratum means. A flat average hides which failure mode is biting."""
    buckets = run.by_stratum(records, attr, metric)
    if not buckets:
        return ""
    lines = [
        "",
        _RULE,
        f"STRATIFIED — {run.variant.name} | {metric} by {attr}",
        _RULE,
    ]
    for key, value in sorted(buckets.items(), key=lambda kv: kv[1]):
        bar = "#" * int(round(value * 30))
        lines.append(f"  {key:<16}{_fmt_pct(value):>7}  {bar}")
    return "\n".join(lines)


def render_diagnosis(
    runs: dict[str, RunResult],
    metric: str = "recall@k",
    generation: dict[str, dict[str, float]] | None = None,
) -> str:
    """Point at the component responsible, not just the score.

    Component-level attribution is the whole reason to run several variants
    rather than one end-to-end number. Each rule below reads a specific
    contrast and names what it implicates.
    """
    lines = ["", _RULE, "DIAGNOSIS", _RULE]
    notes: list[str] = []

    # A judge pinned at 1.000 has stopped measuring. Say so before anyone reads
    # it as "perfect" — a saturated metric cannot detect the regression it is
    # there to catch, and near-perfect scores from an 8B judge on a small,
    # clean corpus are far more likely to be leniency than excellence.
    saturated = sorted(
        {
            metric
            for scores in (generation or {}).values()
            for metric, value in scores.items()
            if not metric.startswith("_") and value >= 0.98
        }
    )
    if saturated:
        notes.append(
            f"JUDGE SATURATED: {', '.join(saturated)} at >=0.98. At the ceiling these "
            "cannot detect a regression, and a local 8B judge is more likely lenient "
            "than the system is flawless. Before trusting them, hand-label ~30 answers "
            "and check agreement, and cross-check with `make evals-ragas-real` — two "
            "judges disagreeing by more than your effect size means neither is usable "
            "as a gate yet."
        )

    # Abstention next: it invalidates every generation number above it, so a
    # reader must see it before reading faithfulness.
    for lang, scores in sorted((generation or {}).items()):
        abstained = scores.get("_abstained", 0.0)
        scored = scores.get("_scored", 0.0)
        total = abstained + scored
        if total and abstained / total > 0.3:
            notes.append(
                f"{lang}: CRAG abstained on {abstained:.0f}/{total:.0f} questions. "
                "The usual cause is CRAG_MIN_SCORE being too low: the retriever then "
                "returns top_k chunks regardless of quality, and the grade prompt is "
                "strict enough to reject a context that is mostly off-topic — so the "
                "corrective loop exhausts CRAG_MAX_LOOPS and abstains on answerable "
                "questions. Raise min_score before trusting any score below. "
                "COMPONENT: retrieval threshold / grader interaction."
            )

    en, ar = runs.get("en:baseline"), runs.get("ar:baseline")
    if en and ar:
        gap = en.mean(metric) - ar.mean(metric)
        if gap > 0.10:
            notes.append(
                f"Arabic baseline trails English by {gap:.3f} on {metric}. The pipeline "
                "is not language-neutral — orthographic variation is suppressing retrieval."
            )

    base, query_norm, both_norm = (
        runs.get("ar:baseline"),
        runs.get("ar:query-norm"),
        runs.get("ar:both-norm"),
    )
    if base and query_norm and both_norm:
        cheap = paired_delta_ci(base.metric_scores(metric), query_norm.metric_scores(metric))
        full = paired_delta_ci(query_norm.metric_scores(metric), both_norm.metric_scores(metric))
        if cheap.significant and cheap.mean > 0:
            notes.append(
                f"Normalising the query alone recovers {cheap.mean:+.3f} with no reindex — "
                "a config change, deployable today. COMPONENT: query preprocessing."
            )
        if full.significant and full.mean > 0:
            notes.append(
                f"Normalising the corpus recovers a further {full.mean:+.3f}. This requires "
                "re-embedding every document — the marks live in the stored text, so no "
                "query-side change can reach them. COMPONENT: ingestion."
            )
        elif not full.significant:
            notes.append(
                "Normalising the corpus adds nothing beyond the query-side fix on this set — "
                "the cheap fix is sufficient here. COMPONENT: query preprocessing."
            )

    ar_vec, ar_hybrid = runs.get("ar:vector-only"), runs.get("ar:baseline")
    en_vec, en_hybrid = runs.get("en:vector-only"), runs.get("en:baseline")
    if ar_vec and ar_hybrid and en_vec and en_hybrid:
        ar_lift = paired_delta_ci(ar_vec.metric_scores(metric), ar_hybrid.metric_scores(metric))
        en_lift = paired_delta_ci(en_vec.metric_scores(metric), en_hybrid.metric_scores(metric))
        if en_lift.mean > 0.01 and ar_lift.mean <= en_lift.mean / 2:
            notes.append(
                f"Hybrid search lifts English by {en_lift.mean:+.3f} but Arabic by only "
                f"{ar_lift.mean:+.3f}. The tsvector is built with to_tsvector('english', …) "
                "(infra/postgres/init/05-semantic.sql), so the lexical half of RRF is close "
                "to inert on Arabic — you are paying for hybrid and getting vector. "
                "COMPONENT: lexical index configuration."
            )

    n = len(next(iter(runs.values())).record_ids) if runs else 0
    needed = min_sample_size(effect=0.05)
    if n < needed:
        notes.append(
            f"SAMPLE SIZE: {n} questions. Detecting a 5-point change at 80% power needs "
            f"roughly {needed}. Treat any comparison whose CI crosses zero as unresolved "
            "rather than negative — this set is too small to prove a small effect absent."
        )

    if not notes:
        notes.append("No contrast crossed a reporting threshold.")

    for note in notes:
        lines.append(f"  * {note}")
    return "\n".join(lines)


JUDGE_LABELS = {
    "judge": "built-in rubric judge (eval/judge.py)",
    "ragas": "RAGAS (isolated venv)",
}


def render_generation(
    generation: dict[str, dict[str, float]] | None,
    judge: str = "judge",
) -> str:
    """Render generation scores, naming the judge that produced them.

    The judge is part of the result, not a footnote. Two judges disagree by
    more than most changes you would ship, so a number without its judge
    attached is not comparable to anything.
    """
    if not generation:
        return ""
    label = JUDGE_LABELS.get(judge, judge)
    lines = ["", _RULE, f"GENERATION — {label}, local Ollama", _RULE]
    # _scored / _abstained are bookkeeping, not quality — rendered separately
    # below so they cannot be mistaken for metrics.
    metrics: list[str] = sorted(
        {m for scores in generation.values() for m in scores if not m.startswith("_")}
    )
    if metrics:
        header = f"{'language':<28}" + "".join(f"{m:>16}" for m in metrics)
        lines.append(header)
        lines.append("-" * len(header))
        for lang, scores in sorted(generation.items()):
            lines.append(
                f"{lang:<28}"
                + "".join(f"{scores.get(m, float('nan')):.3f}".rjust(16) for m in metrics)
            )
        lines.append("")

    for lang, scores in sorted(generation.items()):
        scored = scores.get("_scored", 0.0)
        abstained = scores.get("_abstained", 0.0)
        total = scored + abstained
        if total:
            lines.append(
                f"  {lang}: scored {scored:.0f}/{total:.0f}, abstained {abstained:.0f}"
            )
    lines.append("")
    lines.append("  Abstentions are excluded from the scores above (an abstention has no")
    lines.append("  claims to verify), so a high abstention rate makes them unrepresentative.")
    lines.append("  A local 8B judge is also noisy. Validate it against human labels before")
    lines.append("  treating these as ground truth — see eval/README.md.")
    return "\n".join(lines)


def build_payload(
    runs: dict[str, RunResult],
    records_by_lang: dict[str, Sequence[GoldenRecord]],
    generation: dict[str, dict[str, float]] | None,
    config_snapshot: dict[str, Any],
    dataset_version: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset_version": dataset_version,
        "config": config_snapshot,
        "variants": {},
        "comparisons": [],
        "generation": generation or {},
    }

    for name, run in runs.items():
        intervals = {}
        for metric, value in run.means().items():
            interval = mean_ci(run.metric_scores(metric))
            intervals[metric] = {
                "mean": value,
                "ci_low": interval.low,
                "ci_high": interval.high,
            }
        records = records_by_lang.get(run.variant.lang, [])
        payload["variants"][name] = {
            "description": run.variant.description,
            "lang": run.variant.lang,
            "k": run.k,
            "n": len(run.record_ids),
            "metrics": intervals,
            "by_trap": run.by_stratum(records, "trap", "recall@k"),
            "by_difficulty": run.by_stratum(records, "difficulty", "recall@k"),
            "errors": run.errors,
        }

    for baseline_name, variant_name, question in STANDARD_COMPARISONS:
        base, variant = runs.get(baseline_name), runs.get(variant_name)
        if not base or not variant or base.record_ids != variant.record_ids:
            continue
        delta = paired_delta_ci(base.metric_scores("recall@k"), variant.metric_scores("recall@k"))
        payload["comparisons"].append(
            {
                "question": question,
                "baseline": baseline_name,
                "variant": variant_name,
                "metric": "recall@k",
                "delta": delta.mean,
                "ci_low": delta.low,
                "ci_high": delta.high,
                "significant": delta.significant,
                "direction": delta.direction,
            }
        )
    return payload


def write_report(payload: dict[str, Any], directory: pathlib.Path = REPORTS_DIR) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = payload["timestamp"].replace(":", "-").replace("+00-00", "Z")
    path = directory / f"{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
