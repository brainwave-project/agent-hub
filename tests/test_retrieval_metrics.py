"""Retrieval metric unit tests — pure functions, no stack, no network.

Every expected value below is computed by hand in the comment above it. That is
deliberate: these metrics are the instrument the rest of the eval work depends
on, so a silent arithmetic error here would quietly corrupt every conclusion
drawn later. Hand-checked fixtures are the only way to catch that.
"""

import math

from agent_hub.eval.retrieval_metrics import (
    average_precision,
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    score_all,
)

# Canonical fixture used across most tests:
#   ranked   = [a, b, c, d, e]   (system output, best first)
#   relevant = {a, c}            (ground truth)
RANKED = ["a", "b", "c", "d", "e"]
RELEVANT = {"a", "c"}


def test_hit_rate_finds_relevant_in_top_k():
    assert hit_rate(RANKED, RELEVANT, k=1) == 1.0   # 'a' is at rank 1
    assert hit_rate(RANKED, RELEVANT, k=5) == 1.0


def test_hit_rate_is_zero_when_nothing_relevant_in_window():
    # 'a' at rank 3, window is only 2 deep.
    assert hit_rate(["x", "y", "a"], {"a"}, k=2) == 0.0


def test_hit_rate_saturates_and_hides_ranking_quality():
    # The documented failure mode: rank-1 and rank-5 score identically.
    assert hit_rate(["a", "x", "y", "z", "w"], {"a"}, k=5) == 1.0
    assert hit_rate(["x", "y", "z", "w", "a"], {"a"}, k=5) == 1.0


def test_recall_at_k():
    # Both relevant docs (a, c) are inside top 5 → 2/2
    assert recall_at_k(RANKED, RELEVANT, k=5) == 1.0
    # Only 'a' inside top 2 → 1/2
    assert recall_at_k(RANKED, RELEVANT, k=2) == 0.5
    # Neither inside top 0
    assert recall_at_k(RANKED, RELEVANT, k=0) == 0.0


def test_precision_at_k_caps_below_one_when_relevant_set_is_small():
    # The documented failure mode: 1 relevant doc, k=5 → best possible is 0.2
    assert precision_at_k(["a", "x", "y", "z", "w"], {"a"}, k=5) == 0.2
    # 2 of 5 retrieved are relevant → 0.4
    assert precision_at_k(RANKED, RELEVANT, k=5) == 0.4


def test_mrr_uses_first_relevant_rank():
    assert mrr(RANKED, RELEVANT) == 1.0                       # 'a' at rank 1 → 1/1
    assert mrr(["x", "y", "a"], {"a"}) == 1.0 / 3             # rank 3 → 1/3
    assert mrr(["x", "y", "z"], {"a"}) == 0.0                 # never found


def test_mrr_is_blind_to_recall():
    # Documented failure mode: retrieving 1 of 3 relevant docs at rank 1 scores
    # a perfect 1.0, identical to retrieving all 3.
    one_of_three = mrr(["a", "x", "y"], {"a", "b", "c"})
    all_three = mrr(["a", "b", "c"], {"a", "b", "c"})
    assert one_of_three == all_three == 1.0


def test_ndcg_matches_hand_computation():
    # DCG  = 1/log2(1+1) + 1/log2(3+1)   ('a' rank 1, 'c' rank 3)
    #      = 1/1 + 1/2 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3)  (2 relevant docs packed at ranks 1,2)
    #      = 1.0 + 0.6309297... = 1.6309297...
    # NDCG = 1.5 / 1.6309297 = 0.9197207...
    expected = 1.5 / (1.0 + 1.0 / math.log2(3))
    assert math.isclose(ndcg_at_k(RANKED, RELEVANT, k=10), expected, rel_tol=1e-9)
    assert math.isclose(expected, 0.9197207891481876, rel_tol=1e-9)


def test_ndcg_is_one_for_a_perfect_ranking():
    assert math.isclose(ndcg_at_k(["a", "c", "x"], {"a", "c"}, k=10), 1.0, rel_tol=1e-9)


def test_ndcg_log_discount_barely_moves_for_deep_changes():
    # Documented failure mode: a 9-place improvement deep in the ranking
    # produces a change most people would call noise.
    deep = ndcg_at_k(["x"] * 19 + ["a"], {"a"}, k=20)          # rank 20
    shallower = ndcg_at_k(["x"] * 10 + ["a"] + ["y"] * 9, {"a"}, k=20)  # rank 11
    assert shallower - deep < 0.06


def test_average_precision_matches_hand_computation():
    # Hits at rank 1 (precision 1/1 = 1.0) and rank 3 (precision 2/3 = 0.6667).
    # AP = (1.0 + 0.666666...) / min(2, 10) = 1.666666... / 2 = 0.833333...
    expected = (1.0 + 2 / 3) / 2
    assert math.isclose(average_precision(RANKED, RELEVANT, k=10), expected, rel_tol=1e-9)


def test_empty_and_degenerate_inputs_score_zero_not_raise():
    assert hit_rate([], {"a"}) == 0.0
    assert recall_at_k([], {"a"}) == 0.0
    assert precision_at_k([], {"a"}) == 0.0
    assert mrr([], {"a"}) == 0.0
    assert ndcg_at_k([], {"a"}) == 0.0
    assert average_precision([], {"a"}) == 0.0
    # Empty ground truth is a malformed golden record, not a perfect score.
    assert recall_at_k(RANKED, set()) == 0.0
    assert ndcg_at_k(RANKED, set()) == 0.0


def test_score_all_returns_every_metric():
    scores = score_all(RANKED, RELEVANT, k=5)
    assert set(scores) == {"hit_rate", "recall@k", "precision@k", "mrr", "ndcg@k", "map"}
    assert all(0.0 <= v <= 1.0 for v in scores.values())
    assert scores["recall@k"] == 1.0
    assert scores["mrr"] == 1.0


def test_duplicate_ids_do_not_inflate_recall():
    # A system returning the same doc five times has found one doc, not five.
    assert recall_at_k(["a", "a", "a"], {"a", "b"}, k=5) == 0.5
