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
    "min_sample_size",
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
