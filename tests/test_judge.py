"""Built-in judge — the parts testable without an LLM.

The rubric quality itself can only be assessed against human labels (see
eval/README.md). What IS mechanically testable is the bookkeeping around it,
and that is where a silent error would do the most damage: if abstentions
leaked into the scored set, faithfulness would be dragged down by the system
behaving correctly, and the number would be wrong in a direction nobody checks.
"""

import asyncio
from dataclasses import dataclass

from agent_hub.eval.judge import (
    DECOMPOSE_RUBRIC,
    RELEVANCY_RUBRIC,
    VERIFY_RUBRIC,
    JudgedAnswer,
    LocalJudge,
    score_answers,
)


@dataclass
class FakeOutcome:
    """Shaped like CragOutcome, without needing the graph."""

    record_id: str
    question: str = "q?"
    answer: str = ""
    contexts: tuple = ()
    abstained: bool = False
    error: str = ""


def test_rubrics_are_present_and_substantive():
    """The rubrics ARE the judge — an empty one silently degrades every score."""
    for name, rubric in (
        ("decompose", DECOMPOSE_RUBRIC),
        ("verify", VERIFY_RUBRIC),
        ("relevancy", RELEVANCY_RUBRIC),
    ):
        assert rubric.strip(), f"{name} rubric is empty"
        assert len(rubric) > 80, f"{name} rubric is suspiciously short"


def test_abstentions_and_errors_are_excluded_from_scoring():
    """Nothing scorable means no LLM call at all, and honest counts."""
    outcomes = [
        FakeOutcome("a", answer="", abstained=True),
        FakeOutcome("b", answer="I don't know.", abstained=True),
        FakeOutcome("c", answer="", error="boom"),
        FakeOutcome("d", answer="   "),          # whitespace only
    ]
    scores, judged = asyncio.run(score_answers(outcomes, verbose=False))

    assert judged == []                           # never reached the model
    assert scores["_scored"] == 0.0
    assert scores["_abstained"] == 2.0
    assert "faithfulness" not in scores           # no fabricated score
    assert "answer_relevancy" not in scores


def test_judged_answer_reports_unsupported_claims():
    judged = JudgedAnswer(
        record_id="x",
        faithfulness=1 / 3,
        answer_relevancy=0.9,
        claims=["supported one", "invented one", "another invented"],
        supported=[True, False, False],
    )
    assert judged.unsupported_claims == ["invented one", "another invented"]


def test_judge_construction_does_not_require_a_reachable_model():
    """Constructing the client must not connect — tests stay offline."""
    judge = LocalJudge()
    assert judge.model_name


def test_empty_answer_scores_zero_relevancy_without_a_call():
    judge = LocalJudge()
    assert asyncio.run(judge.answer_relevancy("q?", "")) == 0.0


def test_faithfulness_with_no_context_scores_zero_without_a_call():
    judge = LocalJudge()
    score, claims, supported = asyncio.run(judge.faithfulness("some answer", []))
    assert score == 0.0
    assert claims == []
    assert supported == []
