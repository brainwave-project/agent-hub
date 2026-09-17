"""Cluster bootstrap, t-interval, permutation test, detectable effect.

Pure functions, no stack. Where a reference value exists it is stated in the
comment above the assertion, so a regression here is checkable by hand.
"""

import math

from agent_hub.eval.stats import (
    cluster_mean_ci,
    detectable_effect,
    mean_ci,
    min_sample_size,
    paired_cluster_delta_ci,
    paired_delta_ci,
    paired_permutation_p,
    t_interval,
)


def test_t_interval_matches_textbook_critical_value():
    # n=10, df=9: t*(0.975) = 2.2622 (standard t-table).
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    iv = t_interval(scores)
    sd = 0.30276503540974914  # statistics.stdev(scores)
    expected_half = 2.2622 * sd / math.sqrt(10)
    assert math.isclose(iv.half_width, expected_half, rel_tol=1e-3)


def test_t_interval_escapes_the_metric_range_on_bimodal_scores():
    # The documented failure: bounded, bimodal scores. Mostly 1.0 with two 0.0s.
    scores = [1.0] * 8 + [0.0] * 2
    t = t_interval(scores)
    boot = mean_ci(scores)
    assert t.high > 1.0  # impossible value for a recall
    assert boot.high <= 1.0  # the bootstrap cannot leave the data's range


def test_cluster_ci_equals_plain_bootstrap_when_every_cluster_is_a_singleton():
    scores = [0.0, 0.5, 1.0, 1.0, 0.25, 0.75, 0.0, 1.0]
    plain = mean_ci(scores)
    clustered = cluster_mean_ci(scores, [f"c{i}" for i in range(len(scores))])
    # Same seed, same number of draws per resample -> identical intervals.
    assert math.isclose(plain.low, clustered.low)
    assert math.isclose(plain.high, clustered.high)


def test_cluster_ci_widens_when_questions_fail_together():
    # Four documents, three questions each; each document either works or fails wholesale.
    scores = [1.0] * 3 + [0.0] * 3 + [1.0] * 3 + [0.0] * 3
    clusters = ["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3
    plain = mean_ci(scores)
    clustered = cluster_mean_ci(scores, clusters)
    assert clustered.n == 4  # effective unit of analysis
    assert clustered.half_width > plain.half_width


def test_single_cluster_reports_infinite_interval():
    iv = cluster_mean_ci([1.0, 0.0, 1.0], ["same", "same", "same"])
    assert math.isinf(iv.low) and math.isinf(iv.high)


def test_paired_cluster_delta_matches_paired_delta_for_singletons():
    base = [0.2, 0.4, 0.6, 0.8, 0.5]
    treat = [0.3, 0.5, 0.6, 0.9, 0.7]
    plain = paired_delta_ci(base, treat)
    clustered = paired_cluster_delta_ci(base, treat, ["a", "b", "c", "d", "e"])
    assert math.isclose(plain.low, clustered.low)
    assert math.isclose(plain.high, clustered.high)


def test_permutation_p_is_one_for_identical_arms():
    scores = [0.1, 0.5, 0.9]
    assert paired_permutation_p(scores, scores) == 1.0


def test_permutation_p_is_small_for_a_consistent_effect():
    # 20 questions, every one improves by 0.2. Under the null, all 20 signs
    # agreeing happens with probability 2 * 0.5^20 — essentially never.
    base = [0.4] * 20
    treat = [0.6] * 20
    p = paired_permutation_p(base, treat)
    assert p < 0.001


def test_permutation_p_never_reports_zero():
    p = paired_permutation_p([0.0] * 30, [1.0] * 30, resamples=500)
    assert p > 0.0


def test_detectable_effect_inverts_min_sample_size():
    n = min_sample_size(effect=0.05, score_sd=0.35)
    assert detectable_effect(n, score_sd=0.35) <= 0.05
    assert detectable_effect(n - 1, score_sd=0.35) > 0.05


def test_detectable_effect_shrinks_with_root_n():
    # Quadrupling n halves the detectable effect.
    assert math.isclose(detectable_effect(100), detectable_effect(25) / 2)
