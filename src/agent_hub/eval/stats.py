"""Statistics for eval scores — confidence intervals and regression detection.

The gap this closes: almost every RAG eval reports a bare mean. "Recall went
from 0.71 to 0.74" is not a result, it is a number. On a 20-question set that
difference is well inside the noise, and acting on it is indistinguishable from
acting on a coin flip.

Everything here is stdlib-only (no numpy/scipy) on purpose — the arithmetic is
simple enough to read, and the harness should not need a scientific stack to
tell you whether a number moved.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "Interval",
    "Delta",
    "mean_ci",
    "paired_delta_ci",
    "cluster_mean_ci",
    "paired_cluster_delta_ci",
    "t_interval",
    "paired_permutation_p",
    "min_sample_size",
    "detectable_effect",
    "DEFAULT_RESAMPLES",
]

DEFAULT_RESAMPLES = 5000


@dataclass(frozen=True)
class Interval:
    """A point estimate with a bootstrap confidence interval."""

    mean: float
    low: float
    high: float
    n: int
    confidence: float = 0.95

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.low:.3f}, {self.high:.3f}] n={self.n}"


@dataclass(frozen=True)
class Delta:
    """A paired difference (variant B minus variant A) with its CI."""

    mean: float
    low: float
    high: float
    n: int
    confidence: float = 0.95

    @property
    def significant(self) -> bool:
        """True when the interval excludes zero.

        This is the only question the harness should ask of an A/B result. If
        the interval spans zero, the observed difference is consistent with no
        difference at all, however large the point estimate looks.
        """
        return (self.low > 0.0) or (self.high < 0.0)

    @property
    def direction(self) -> str:
        if not self.significant:
            return "no change"
        return "improvement" if self.mean > 0 else "REGRESSION"

    def __str__(self) -> str:
        marker = "" if self.significant else "  (CI crosses zero — not significant)"
        return f"{self.mean:+.3f} [{self.low:+.3f}, {self.high:+.3f}] n={self.n}{marker}"


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile. `q` in [0, 1]."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return sorted_values[int(pos)]
    weight = pos - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def mean_ci(
    scores: Sequence[float],
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 20260829,
) -> Interval:
    """Bootstrap CI for the mean of per-question scores.

    Bootstrap rather than a t-interval because eval scores are typically bounded
    in [0, 1], heavily skewed, and often bimodal (a question either works or it
    doesn't). None of that satisfies a t-test's normality assumption, but the
    bootstrap does not care about the shape of the distribution.

    `seed` is fixed so a report is reproducible — rerunning the same scores
    yields the same interval.
    """
    n = len(scores)
    if n == 0:
        return Interval(0.0, 0.0, 0.0, 0, confidence)
    if n == 1:
        # One observation carries no information about spread. Report the point
        # and an infinitely wide interval rather than a fake zero-width one.
        return Interval(scores[0], float("-inf"), float("inf"), 1, confidence)

    rng = random.Random(seed)
    values = list(scores)
    means = sorted(
        statistics.fmean(rng.choices(values, k=n)) for _ in range(resamples)
    )

    tail = (1.0 - confidence) / 2
    return Interval(
        mean=statistics.fmean(values),
        low=_percentile(means, tail),
        high=_percentile(means, 1.0 - tail),
        n=n,
        confidence=confidence,
    )


def paired_delta_ci(
    baseline: Sequence[float],
    variant: Sequence[float],
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 20260829,
) -> Delta:
    """Bootstrap CI for the mean per-question difference (variant - baseline).

    PAIRED, not two-sample: both variants are run over the same questions, so
    the comparison resamples the *differences* rather than the two score sets
    independently. This cancels per-question difficulty, which is by far the
    largest source of variance in a small eval set — an unpaired test on 20
    questions will almost never reach significance even for a real improvement.

    Raises on a length mismatch: unequal lengths mean the runs were not actually
    paired, and silently truncating would produce a meaningless number.
    """
    if len(baseline) != len(variant):
        raise ValueError(
            f"paired comparison needs equal lengths, got {len(baseline)} and {len(variant)}"
        )

    diffs = [v - b for b, v in zip(baseline, variant, strict=True)]
    n = len(diffs)
    if n == 0:
        return Delta(0.0, 0.0, 0.0, 0, confidence)
    if n == 1:
        return Delta(diffs[0], float("-inf"), float("inf"), 1, confidence)

    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(diffs, k=n)) for _ in range(resamples))

    tail = (1.0 - confidence) / 2
    return Delta(
        mean=statistics.fmean(diffs),
        low=_percentile(means, tail),
        high=_percentile(means, 1.0 - tail),
        n=n,
        confidence=confidence,
    )


def min_sample_size(
    effect: float = 0.05,
    score_sd: float = 0.35,
    confidence: float = 0.95,
    power: float = 0.80,
) -> int:
    """Roughly how many questions are needed to detect `effect` reliably.

    Normal-approximation sample size for a paired design:

        n = ((z_alpha/2 + z_beta) * sd / effect)^2

    Treat the output as an order of magnitude, not a precise requirement — its
    job is to make the cost of small eval sets concrete. With the defaults
    (detect a 5-point change, sd 0.35) it returns a few hundred questions, which
    is the honest reason a 20-question set cannot adjudicate a 2-point
    difference no matter how carefully the mean is computed.

    `score_sd` should come from your own data: `statistics.stdev(scores)` on a
    previous run of the same metric.
    """
    if effect <= 0:
        raise ValueError("effect must be > 0")

    # Two-sided z for alpha, one-sided z for beta.
    z_alpha = statistics.NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    z_beta = statistics.NormalDist().inv_cdf(power)
    n = ((z_alpha + z_beta) * score_sd / effect) ** 2
    return max(1, math.ceil(n))


def detectable_effect(
    n: int,
    score_sd: float = 0.35,
    confidence: float = 0.95,
    power: float = 0.80,
) -> float:
    """The smallest effect `n` paired questions can detect — `min_sample_size` inverted.

        effect = (z_alpha/2 + z_beta) * sd / sqrt(n)

    This is the number to put next to every "no significant difference" result.
    "No change detected" on a set whose minimum detectable effect is 0.15 says
    nothing about a 0.05 improvement: the experiment could not have seen it.
    """
    if n <= 0:
        return float("inf")
    z_alpha = statistics.NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    z_beta = statistics.NormalDist().inv_cdf(power)
    return (z_alpha + z_beta) * score_sd / math.sqrt(n)


# --- cluster bootstrap ---------------------------------------------------------


def _group(values: Sequence[float], clusters: Sequence[str]) -> list[list[float]]:
    if len(values) != len(clusters):
        raise ValueError(
            f"every score needs a cluster label, got {len(values)} scores and {len(clusters)} labels"
        )
    grouped: dict[str, list[float]] = {}
    for value, cluster in zip(values, clusters, strict=True):
        grouped.setdefault(cluster, []).append(value)
    # Sorted by label so the resampling order (and therefore the interval) is
    # independent of the order records happened to arrive in.
    return [grouped[k] for k in sorted(grouped)]


def _cluster_bootstrap(
    values: Sequence[float],
    clusters: Sequence[str],
    confidence: float,
    resamples: int,
    seed: int,
) -> tuple[float, float, float, int]:
    groups = _group(values, clusters)
    point = statistics.fmean(values) if values else 0.0
    if len(groups) < 2:
        # One cluster is one observation of the thing that varies. Same honesty
        # rule as mean_ci with n=1: no spread information, infinite interval.
        return point, float("-inf"), float("inf"), len(groups)

    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        drawn = rng.choices(groups, k=len(groups))
        total = sum(sum(g) for g in drawn)
        count = sum(len(g) for g in drawn)
        means.append(total / count)
    means.sort()
    tail = (1.0 - confidence) / 2
    return point, _percentile(means, tail), _percentile(means, 1.0 - tail), len(groups)


def cluster_mean_ci(
    scores: Sequence[float],
    clusters: Sequence[str],
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 20260829,
) -> Interval:
    """Bootstrap CI that resamples CLUSTERS of questions, not questions.

    The ordinary bootstrap assumes every question is an independent draw. That
    is false whenever several questions share an answer document: if retrieval
    handles `en-leave-policy` badly, all four leave questions fail together, and
    the four failures are one piece of evidence counted four times. Resampling
    question-level scores then produces an interval that is too NARROW:
    confident about a number that rests on far fewer independent observations
    than `n` suggests.

    Resampling whole clusters (here: questions grouped by their primary
    relevant document) keeps correlated questions together. The interval widens
    exactly as much as the correlation deserves; when every cluster is a
    singleton it reduces to the ordinary bootstrap. `n` on the result is the
    number of CLUSTERS, which is the effective unit of analysis.
    """
    point, low, high, groups = _cluster_bootstrap(scores, clusters, confidence, resamples, seed)
    return Interval(point, low, high, groups, confidence)


def paired_cluster_delta_ci(
    baseline: Sequence[float],
    variant: Sequence[float],
    clusters: Sequence[str],
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 20260829,
) -> Delta:
    """Paired per-question deltas, resampled by cluster. See `cluster_mean_ci`."""
    if len(baseline) != len(variant):
        raise ValueError(
            f"paired comparison needs equal lengths, got {len(baseline)} and {len(variant)}"
        )
    diffs = [v - b for b, v in zip(baseline, variant, strict=True)]
    if not diffs:
        return Delta(0.0, 0.0, 0.0, 0, confidence)
    point, low, high, groups = _cluster_bootstrap(diffs, clusters, confidence, resamples, seed)
    return Delta(point, low, high, groups, confidence)


# --- normal-theory interval, for contrast ---------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / (c if abs(c) > tiny else tiny)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / (c if abs(c) > tiny else tiny)
        step = d * c
        h *= step
        if abs(step - 1.0) < 1e-12:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _t_cdf(t: float, df: float) -> float:
    tail = 0.5 * _betai(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t >= 0 else tail


_T_QUANTILE_CACHE: dict[tuple[float, float], float] = {}


def _t_quantile(p: float, df: float) -> float:
    """Inverse Student-t CDF by bisection: slow, exact enough, dependency-free.

    Memoised, because the coverage simulator asks for the same (p, df) for
    every one of thousands of simulated samples.
    """
    key = (p, df)
    if key not in _T_QUANTILE_CACHE:
        low, high = -1e3, 1e3
        for _ in range(200):
            mid = (low + high) / 2
            if _t_cdf(mid, df) < p:
                low = mid
            else:
                high = mid
        _T_QUANTILE_CACHE[key] = (low + high) / 2
    return _T_QUANTILE_CACHE[key]


def t_interval(scores: Sequence[float], confidence: float = 0.95) -> Interval:
    """Classic Student-t interval: mean +/- t* x sd / sqrt(n).

    Here for CONTRAST with the bootstrap, not as a recommendation. It assumes
    the sampling distribution of the mean is normal. By the CLT that becomes
    true as n grows, but eval scores are bounded in [0, 1] and frequently
    bimodal (a query works or it does not), and on 20-40 questions the CLT has
    not yet rescued you. The symptom is visible: the t-interval is symmetric by
    construction and happily extends below 0 or above 1, bounds a retrieval
    metric can never take. The bootstrap interval, built from the data's own
    shape, cannot.
    """
    n = len(scores)
    if n == 0:
        return Interval(0.0, 0.0, 0.0, 0, confidence)
    mean = statistics.fmean(scores)
    if n == 1:
        return Interval(mean, float("-inf"), float("inf"), 1, confidence)
    sd = statistics.stdev(scores)
    t_star = _t_quantile(1 - (1 - confidence) / 2, n - 1)
    half = t_star * sd / math.sqrt(n)
    return Interval(mean, mean - half, mean + half, n, confidence)


# --- permutation test ----------------------------------------------------------


def paired_permutation_p(
    baseline: Sequence[float],
    variant: Sequence[float],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 20260829,
) -> float:
    """Two-sided sign-flip permutation p-value for the mean paired difference.

    Shuffle, not bootstrap. The bootstrap asks "how much would the delta vary if
    I re-drew the questions?" and yields an interval. The permutation test asks
    "if the two variants were interchangeable, how often would randomly swapping
    their labels per question produce a delta at least this large?" and yields a
    p-value. Under the null a question's delta is equally likely to be +d or -d,
    so flipping signs at random enumerates the null distribution.

    Reported NEXT TO the interval, never instead of it. A p-value compresses the
    evidence into one threshold-crossing bit, which is exactly what makes it
    easy to p-hack: run enough variants and one will cross 0.05 by chance.
    """
    if len(baseline) != len(variant):
        raise ValueError(
            f"paired comparison needs equal lengths, got {len(baseline)} and {len(variant)}"
        )
    diffs = [v - b for b, v in zip(baseline, variant, strict=True)]
    n = len(diffs)
    if n == 0:
        return 1.0
    observed = abs(statistics.fmean(diffs))
    if observed == 0.0:
        return 1.0
    rng = random.Random(seed)
    extreme = 0
    for _ in range(resamples):
        flipped = sum(d if rng.random() < 0.5 else -d for d in diffs) / n
        if abs(flipped) >= observed - 1e-12:
            extreme += 1
    # +1 on both sides: the observed labelling is itself one permutation, and it
    # keeps p strictly above zero. "Never seen in 5000 shuffles" is not p=0.
    return (extreme + 1) / (resamples + 1)


# --- building blocks for the Stats Lab -------------------------------------------


def bootstrap_means(
    values: Sequence[float], resamples: int = DEFAULT_RESAMPLES, seed: int = 20260829
) -> list[float]:
    """The bootstrap distribution itself: the sorted means of `resamples` resamples.

    `mean_ci` reports only two percentiles of this list. Looking at the whole
    distribution is how you see what a bootstrap interval is — and, on a small
    binary sample with few distinct resample means, why the percentiles read
    off it are coarse and lumpy.
    """
    n = len(values)
    if n == 0:
        return []
    rng = random.Random(seed)
    pool = list(values)
    return sorted(statistics.fmean(rng.choices(pool, k=n)) for _ in range(resamples))


def normal_interval(scores: Sequence[float], confidence: float = 0.95) -> Interval:
    """Large-sample (z) interval: mean +/- z* x sd / sqrt(n).

    What the CLT promises in the limit. At small n it is narrower than the
    t-interval because it ignores that `sd` is itself an estimate — so it
    under-covers exactly when you have the least data.
    """
    n = len(scores)
    if n == 0:
        return Interval(0.0, 0.0, 0.0, 0, confidence)
    mean = statistics.fmean(scores)
    if n == 1:
        return Interval(mean, float("-inf"), float("inf"), 1, confidence)
    z = statistics.NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    half = z * statistics.stdev(scores) / math.sqrt(n)
    return Interval(mean, mean - half, mean + half, n, confidence)


def paired_t_p(baseline: Sequence[float], variant: Sequence[float]) -> float:
    """Two-sided paired t-test p-value.

    Assumes the per-question deltas are roughly normal, which retrieval deltas
    (mostly exactly 0, a few +/-1) are not. Used by the simulators because it is
    cheap enough to call tens of thousands of times; reports use the
    permutation test instead.
    """
    if len(baseline) != len(variant):
        raise ValueError("paired test needs equal lengths")
    diffs = [v - b for b, v in zip(baseline, variant, strict=True)]
    n = len(diffs)
    if n < 2:
        return 1.0
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    if sd == 0.0:
        return 1.0 if mean == 0.0 else 0.0
    t = mean / (sd / math.sqrt(n))
    return max(0.0, min(1.0, 2.0 * (1.0 - _t_cdf(abs(t), n - 1))))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjusted p-values.

    Test m hypotheses at level alpha each and the chance that at least one null
    "wins" by luck grows to about 1 - (1 - alpha)^m. Holm controls that
    family-wise error: sort the p-values, multiply the i-th smallest by
    (m - i), and never let an adjusted value fall below the one before it. It
    is uniformly more powerful than plain Bonferroni (every p x m) while giving
    the same guarantee.
    """
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    running = 0.0
    adjusted: dict[str, float] = {}
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted


def cohens_dz(baseline: Sequence[float], variant: Sequence[float]) -> float:
    """Standardised paired effect size: mean(delta) / sd(delta).

    Answers "how large is the change relative to its own noise?", independent
    of n — unlike a p-value, which a big enough n drives to zero for any
    non-zero effect.

    MISLEADS WHEN: most deltas are exactly zero. sd(delta) is then tiny, so a
    handful of flipped questions produces a "huge" d. On retrieval data, read d
    next to the raw delta and the win/loss counts, never alone.
    """
    diffs = [v - b for b, v in zip(baseline, variant, strict=True)]
    if len(diffs) < 2:
        return 0.0
    sd = statistics.stdev(diffs)
    if sd == 0.0:
        return 0.0
    return statistics.fmean(diffs) / sd
