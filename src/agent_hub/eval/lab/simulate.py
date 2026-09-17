"""Stats Lab — simulations that turn statistical claims into things you can watch.

Each function answers one question that is easy to recite and hard to feel:

    bootstrap_demo  What IS a bootstrap interval? (the resampling distribution itself)
    coverage_sim    What does "95% confidence" promise? (how often intervals cover
                    the truth, per method, at your n, on your score shape)
    clt_sim         When does the CLT rescue a normal-theory interval?
    power_sim       How often does an experiment of size n detect an effect of size d?
    phacking_sim    How do many metrics and peeking manufacture significance?

All simulations draw from an explicit, finite **population** of per-query scores
— either a real run's scores (treated as the population) or a synthetic shape.
Because the population is known, the true mean is known, and "did the interval
cover it?" has an answer. That is the only setting in which coverage can be
measured at all: in real evaluation the true mean is exactly what you lack.

Every run is seeded, so the same inputs give the same pictures.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Literal

from pydantic import BaseModel, Field

from agent_hub.eval.lab.store import LabStore
from agent_hub.eval.stats import (
    bootstrap_means,
    holm_adjust,
    mean_ci,
    min_sample_size,
    normal_interval,
    paired_t_p,
    t_interval,
)

__all__ = [
    "PopulationSpec",
    "population_values",
    "histogram",
    "bootstrap_demo",
    "coverage_sim",
    "clt_sim",
    "power_sim",
    "phacking_sim",
    "MAX_WORK",
]

POPULATION_SIZE = 1000
#: Upper bound on sims x resamples x n, so one request stays a few seconds.
MAX_WORK = 40_000_000


class PopulationSpec(BaseModel):
    kind: Literal["run", "bernoulli", "bimodal", "skewed"] = "bimodal"
    run_id: str | None = None
    metric: str = "recall@k"
    #: bernoulli: success rate. bimodal: share of exact 1.0 scores.
    p: float = Field(default=0.7, ge=0.0, le=1.0)
    #: bimodal: share of exact 0.0 scores (the rest are spread across (0, 1)).
    p_zero: float = Field(default=0.2, ge=0.0, le=1.0)


def population_values(spec: PopulationSpec, store: LabStore) -> tuple[list[float], str]:
    if spec.kind == "run":
        if not spec.run_id:
            raise ValueError("run population needs run_id")
        run = store.get_run(spec.run_id)
        values = [
            r.metrics[spec.metric]
            for r in run.results
            if r.error is None and spec.metric in r.metrics
        ]
        if len(values) < 2:
            raise ValueError(f"run {spec.run_id} has fewer than 2 scores for {spec.metric}")
        return values, f"{spec.metric} of {len(values)} queries in {spec.run_id}"
    if spec.kind == "bernoulli":
        ones = round(spec.p * POPULATION_SIZE)
        return [1.0] * ones + [0.0] * (POPULATION_SIZE - ones), f"Bernoulli(p={spec.p:.2f})"
    if spec.kind == "bimodal":
        if spec.p + spec.p_zero > 1.0:
            raise ValueError("p + p_zero must not exceed 1")
        ones = round(spec.p * POPULATION_SIZE)
        zeros = round(spec.p_zero * POPULATION_SIZE)
        middle = POPULATION_SIZE - ones - zeros
        spread = [(i + 0.5) / middle for i in range(middle)]
        return (
            [1.0] * ones + [0.0] * zeros + spread,
            f"bimodal: {spec.p:.0%} at 1, {spec.p_zero:.0%} at 0, rest spread",
        )
    # skewed: most mass near 0, a long tail toward 1 (mean ~0.25)
    return (
        [((i + 0.5) / POPULATION_SIZE) ** 3 for i in range(POPULATION_SIZE)],
        "right-skewed: most scores near 0, long tail",
    )


def histogram(
    values: list[float], bins: int = 30, lo: float | None = None, hi: float | None = None
) -> list[dict]:
    if not values:
        return []
    lo = min(values) if lo is None else lo
    hi = max(values) if hi is None else hi
    if hi <= lo:
        hi = lo + 1e-9
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = int((v - lo) / width)
        counts[min(bins - 1, max(0, idx))] += 1
    return [
        {"lo": lo + i * width, "hi": lo + (i + 1) * width, "count": c} for i, c in enumerate(counts)
    ]


def _finite(x: float) -> float | None:
    return x if math.isfinite(x) else None


def _skew(values: list[float]) -> float:
    n = len(values)
    if n < 3:
        return 0.0
    mean = statistics.fmean(values)
    sd = statistics.pstdev(values)
    if sd == 0:
        return 0.0
    return sum(((v - mean) / sd) ** 3 for v in values) / n


# --- bootstrap ---------------------------------------------------------------------


def bootstrap_demo(values: list[float], resamples: int, confidence: float, seed: int) -> dict:
    means = bootstrap_means(values, resamples, seed)
    boot = mean_ci(values, confidence, resamples, seed)
    t = t_interval(values, confidence)
    z = normal_interval(values, confidence)
    n = len(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    return {
        "n": n,
        "mean": statistics.fmean(values),
        "sd": sd,
        "se": sd / math.sqrt(n) if n else 0.0,
        "distinct_values": len(set(values)),
        "distinct_resample_means": len(set(round(m, 9) for m in means)),
        "bootstrap_sd": statistics.pstdev(means) if means else 0.0,
        "bootstrap_skew": _skew(means),
        "intervals": {
            "bootstrap": {"low": _finite(boot.low), "high": _finite(boot.high)},
            "t": {"low": _finite(t.low), "high": _finite(t.high)},
            "normal": {"low": _finite(z.low), "high": _finite(z.high)},
        },
        "values_hist": histogram(values, 20, 0.0, 1.0),
        "bootstrap_hist": histogram(means, 40),
    }


# --- coverage ----------------------------------------------------------------------


def coverage_sim(
    values: list[float], n: int, sims: int, resamples: int, confidence: float, seed: int
) -> dict:
    """Draw `sims` samples of size n, build three intervals on each, count coverage.

    Nominal coverage is `confidence`. Actual coverage below it means the method
    is overconfident at this n on this score shape — its "95%" is a label, not a
    property.
    """
    true_mean = statistics.fmean(values)
    rng = random.Random(seed)
    records: dict[str, list[tuple[float, float]]] = {"bootstrap": [], "t": [], "normal": []}
    sample_means = []
    for s in range(sims):
        sample = rng.choices(values, k=n)
        sample_means.append(statistics.fmean(sample))
        for name, interval in (
            ("bootstrap", mean_ci(sample, confidence, resamples, seed + s)),
            ("t", t_interval(sample, confidence)),
            ("normal", normal_interval(sample, confidence)),
        ):
            records[name].append((interval.low, interval.high))

    methods = {}
    for name, intervals in records.items():
        covers = [lo <= true_mean <= hi for lo, hi in intervals]
        finite = [(lo, hi) for lo, hi in intervals if math.isfinite(lo) and math.isfinite(hi)]
        methods[name] = {
            "coverage": sum(covers) / sims,
            "mean_width": statistics.fmean(hi - lo for lo, hi in finite) if finite else None,
            "outside_range_share": sum(1 for lo, hi in intervals if lo < 0 or hi > 1) / sims,
            "zero_width_share": sum(1 for lo, hi in intervals if hi - lo == 0) / sims,
            "miss_above": sum(1 for lo, _ in intervals if lo > true_mean) / sims,
            "miss_below": sum(1 for _, hi in intervals if hi < true_mean) / sims,
            "examples": [
                {"low": _finite(lo), "high": _finite(hi), "covers": c}
                for (lo, hi), c in list(zip(intervals, covers, strict=True))[:60]
            ],
        }
    return {
        "true_mean": true_mean,
        "n": n,
        "sims": sims,
        "confidence": confidence,
        "methods": methods,
        "sample_means_hist": histogram(sample_means, 30, 0.0, 1.0),
    }


# --- CLT ---------------------------------------------------------------------------


def clt_sim(values: list[float], ns: list[int], sims: int, seed: int) -> dict:
    """Sampling distribution of the mean at several n.

    The CLT says it approaches Normal(mu, sigma/sqrt(n)). The skew and the share
    of sample means sitting exactly at 0 or 1 show how far from normal it still
    is at the n you actually have.
    """
    rng = random.Random(seed)
    sigma = statistics.pstdev(values)
    out = []
    for n in ns:
        means = [statistics.fmean(rng.choices(values, k=n)) for _ in range(sims)]
        out.append(
            {
                "n": n,
                "mean": statistics.fmean(means),
                "sd": statistics.pstdev(means),
                "theoretical_se": sigma / math.sqrt(n),
                "skew": _skew(means),
                "share_at_bounds": sum(1 for m in means if m in (0.0, 1.0)) / sims,
                "hist": histogram(means, 30, 0.0, 1.0),
            }
        )
    return {
        "population_mean": statistics.fmean(values),
        "population_sd": sigma,
        "population_skew": _skew(values),
        "population_hist": histogram(values, 20, 0.0, 1.0),
        "distributions": out,
    }


# --- power -------------------------------------------------------------------------


def _discordant_deltas(rng: random.Random, n: int, p_win: float, p_loss: float) -> list[float]:
    deltas = []
    for _ in range(n):
        u = rng.random()
        deltas.append(1.0 if u < p_win else -1.0 if u < p_win + p_loss else 0.0)
    return deltas


def power_sim(
    n: int,
    effect: float,
    discordance: float,
    sims: int,
    resamples: int,
    confidence: float,
    seed: int,
) -> dict:
    """Paired retrieval experiment under a 'discordant pairs' model.

    Each question independently: with probability p_win the treatment fixes it
    (delta +1), with p_loss it breaks it (delta -1), otherwise nothing changes
    (delta 0) — which is what binary retrieval deltas actually look like.
    effect = p_win - p_loss; discordance = p_win + p_loss.

    Empirical power = share of simulated experiments whose paired bootstrap CI
    excludes zero. Analytic power = the normal approximation behind
    `min_sample_size`. Where they disagree, the approximation is the thing that
    is wrong.
    """
    if not 0.0 <= abs(effect) <= discordance <= 1.0:
        raise ValueError("need 0 <= |effect| <= discordance <= 1")
    p_win = (discordance + effect) / 2
    p_loss = (discordance - effect) / 2
    sd = math.sqrt(max(0.0, discordance - effect**2))
    z = statistics.NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    alpha = 1 - confidence
    norm = statistics.NormalDist()

    def analytic(size: int) -> float:
        if sd == 0:
            return 1.0 if effect != 0 else 0.0
        ncp = abs(effect) * math.sqrt(size) / sd
        return norm.cdf(ncp - z) + norm.cdf(-ncp - z)

    rng = random.Random(seed)
    significant = wrong_sign = 0
    observed = []
    for s in range(sims):
        deltas = _discordant_deltas(rng, n, p_win, p_loss)
        interval = mean_ci(deltas, confidence, resamples, seed + s)
        observed.append(statistics.fmean(deltas))
        if interval.low > 0 or interval.high < 0:
            significant += 1
            if effect != 0 and (interval.low > 0) != (effect > 0):
                wrong_sign += 1
    empirical = significant / sims
    grid = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000]
    return {
        "n": n,
        "effect": effect,
        "discordance": discordance,
        "p_win": p_win,
        "p_loss": p_loss,
        "sd_delta": sd,
        "alpha": alpha,
        "analytic_power": analytic(n),
        "empirical_power": empirical,
        "empirical_se": math.sqrt(empirical * (1 - empirical) / sims),
        "wrong_sign_rate": wrong_sign / sims,
        "n_for_80": min_sample_size(abs(effect), sd, confidence, 0.8)
        if effect != 0 and sd > 0
        else None,
        "curve": [{"n": g, "power": analytic(g)} for g in grid],
        "observed_effect_hist": histogram(observed, 30),
    }


# --- p-hacking ---------------------------------------------------------------------


def phacking_sim(
    n: int,
    metrics: int,
    sims: int,
    confidence: float,
    discordance: float,
    correlation: float,
    batch: int,
    seed: int,
) -> dict:
    """Two ways to find significance in a world where nothing changed (effect = 0).

    Many comparisons: test `metrics` null comparisons per experiment and call it
    a win if ANY is significant. With correlation rho, each question's delta on
    metric j copies metric 0's with probability rho — real retrieval metrics
    are correlated, which makes the inflation smaller than the independent
    formula but never zero.

    Optional stopping: add questions `batch` at a time, test after every batch,
    stop at the first p < alpha. Each look is another chance to cross the line.
    """
    alpha = 1 - confidence
    half = discordance / 2
    zeros = [0.0] * n
    rng = random.Random(seed)

    raw_hits = holm_hits = bonf_hits = 0
    min_ps = []
    for _ in range(sims):
        base = _discordant_deltas(rng, n, half, half)
        pvals = {}
        for j in range(metrics):
            if j == 0:
                deltas = base
            else:
                fresh = _discordant_deltas(rng, n, half, half)
                deltas = [
                    b if rng.random() < correlation else f for b, f in zip(base, fresh, strict=True)
                ]
            pvals[f"m{j}"] = paired_t_p(zeros, deltas)
        smallest = min(pvals.values())
        min_ps.append(smallest)
        raw_hits += smallest < alpha
        bonf_hits += smallest * metrics < alpha
        holm_hits += min(holm_adjust(pvals).values()) < alpha

    looks = list(range(batch, n + 1, batch))
    if not looks or looks[-1] != n:
        looks.append(n)
    fixed_fp = peek_fp = 0
    stop_ns = []
    for _ in range(sims):
        deltas = _discordant_deltas(rng, n, half, half)
        fixed_fp += paired_t_p([0.0] * n, deltas) < alpha
        for size in looks:
            if paired_t_p([0.0] * size, deltas[:size]) < alpha:
                peek_fp += 1
                stop_ns.append(size)
                break

    return {
        "alpha": alpha,
        "n": n,
        "sims": sims,
        "many_metrics": {
            "metrics": metrics,
            "correlation": correlation,
            "raw_fwer": raw_hits / sims,
            "bonferroni_fwer": bonf_hits / sims,
            "holm_fwer": holm_hits / sims,
            "independent_formula": 1 - (1 - alpha) ** metrics,
            "min_p_hist": histogram(min_ps, 20, 0.0, 1.0),
        },
        "optional_stopping": {
            "batch": batch,
            "looks": len(looks),
            "fixed_n_fpr": fixed_fp / sims,
            "peeking_fpr": peek_fp / sims,
            "mean_stop_n": statistics.fmean(stop_ns) if stop_ns else None,
            "stop_hist": histogram([float(s) for s in stop_ns], min(20, len(looks)), 0.0, float(n))
            if stop_ns
            else [],
        },
    }
