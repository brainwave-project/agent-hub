"""Graded NDCG — hand-computed fixtures, same discipline as test_retrieval_metrics.py."""

import math

from agent_hub.eval.retrieval_metrics import graded_ndcg_at_k, ndcg_at_k


def test_graded_ndcg_hand_computed():
    # ranked = [a, b, c], grades a=1, c=3
    # DCG  = (2^1-1)/log2(2) + (2^3-1)/log2(4) = 1 + 3.5 = 4.5
    # IDCG = (2^3-1)/log2(2) + (2^1-1)/log2(3) = 7 + 0.63093 = 7.63093
    expected = 4.5 / (7 + 1 / math.log2(3))
    assert math.isclose(graded_ndcg_at_k(["a", "b", "c"], {"a": 1, "c": 3}, k=10), expected)


def test_graded_ndcg_rewards_best_answer_first_where_binary_cannot():
    grades = {"best": 3, "related": 1}
    good = ["best", "related"]
    bad = ["related", "best"]
    # Binary NDCG sees two relevant docs in the top two either way.
    assert ndcg_at_k(good, grades, k=10) == ndcg_at_k(bad, grades, k=10) == 1.0
    assert graded_ndcg_at_k(good, grades, k=10) == 1.0
    assert graded_ndcg_at_k(bad, grades, k=10) < 1.0


def test_graded_ndcg_equals_binary_when_all_grades_equal():
    ranked = ["x", "a", "y", "b"]
    grades = {"a": 2, "b": 2}
    assert math.isclose(graded_ndcg_at_k(ranked, grades, k=10), ndcg_at_k(ranked, grades, k=10))


def test_graded_ndcg_ignores_zero_grades_and_empty_sets():
    assert graded_ndcg_at_k(["a"], {}, k=5) == 0.0
    assert graded_ndcg_at_k(["a"], {"a": 0}, k=5) == 0.0
