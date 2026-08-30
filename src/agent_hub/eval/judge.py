"""LLM-as-judge for generation quality — rubrics written out, not imported.

Exists for two reasons.

The practical one: RAGAS cannot share a virtualenv with agent-hub. It imports
`langchain_community.chat_models.vertexai` at module load, which was removed in
langchain-community 0.4, and agent-hub runs the langchain-core 1.x stack. RAGAS
is still available through `make evals-ragas`, which scores in an isolated
environment — but the default path should not depend on a resolvable-only-by-
luck dependency tree.

The substantive one: a judge you cannot read is a number you cannot defend. The
rubrics below are editable text, the bias controls are explicit and commented,
and the scoring arithmetic is visible. When someone asks "why is faithfulness
0.83?", the answer is in this file rather than in a library's prompt templates.

**Neither this judge nor RAGAS is ground truth.** Both are an 8B model with an
opinion. Run both (`make evals-ragas` reports them side by side), and when they
disagree, hand-label 30 answers to see which one tracks you. That comparison is
the point — an unvalidated judge is an opinion with a decimal point.

### Known biases, and what is done about them

Verbosity — long answers read as more thorough and score higher.
    Faithfulness is scored PER CLAIM, so padding an answer adds unsupported
    claims and lowers the score instead of raising it.

Positional — items early in a list are judged more favourably.
    Each claim is verified in its own call, so there is no list position
    for a claim to occupy.

Self-consistency — the same input scores differently run to run.
    temperature=0.0, and the claim set is sorted before verification, so the
    sequence of calls is identical across runs.

Leniency — models tend to agree with whatever they are shown.
    The rubrics state the strict reading explicitly and demand a verdict per
    claim rather than a global impression of the answer.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from agent_hub.config import Settings, get_settings
from agent_hub.observability.logging import get_logger

log = get_logger("judge")

__all__ = ["LocalJudge", "JudgedAnswer", "score_answers"]


# --- structured verdicts -----------------------------------------------------


class ClaimList(BaseModel):
    """Atomic factual claims extracted from an answer."""

    claims: list[str] = Field(
        default_factory=list,
        description="Each standalone factual assertion, one per item. No opinions.",
    )


class ClaimVerdict(BaseModel):
    supported: bool = Field(description="Is this claim directly supported by the context?")
    reason: str = Field(default="", description="One short sentence.")


class RelevancyVerdict(BaseModel):
    addresses_question: bool = Field(description="Does the answer actually answer the question?")
    score: float = Field(description="0.0 (ignores the question) to 1.0 (fully answers it).")
    reason: str = Field(default="", description="One short sentence.")


# --- rubrics (edit these; they are the judge) --------------------------------

DECOMPOSE_RUBRIC = (
    "You split an answer into atomic factual claims. A claim is a single "
    "assertion that could be independently checked as true or false. Split "
    "compound sentences. Drop hedges, pleasantries, and citation markers such "
    "as [doc:abc]. If the answer states no checkable facts, return an empty "
    "list. Return ONLY the claims."
)

VERIFY_RUBRIC = (
    "You verify whether a single claim is supported by the provided context. "
    "Supported means the context states it or directly entails it. Be strict: "
    "if the claim adds any specific detail — a number, a date, a name, a "
    "condition — that the context does not contain, it is NOT supported. "
    "Plausibility is not support. World knowledge is not support."
)

RELEVANCY_RUBRIC = (
    "You judge whether an answer addresses the question that was asked. Judge "
    "only relevance, never truth — a wrong answer to the right question is "
    "still relevant. Penalise answers that dodge, restate the question, answer "
    "a different question, or bury the response in unrelated material. Length "
    "is not quality: a short direct answer scores higher than a long evasive one."
)


@dataclass
class JudgedAnswer:
    """One answer's scores, with the claim-level detail kept for inspection."""

    record_id: str
    faithfulness: float | None
    answer_relevancy: float | None
    claims: list[str] = field(default_factory=list)
    supported: list[bool] = field(default_factory=list)
    error: str = ""

    @property
    def unsupported_claims(self) -> list[str]:
        return [c for c, ok in zip(self.claims, self.supported, strict=False) if not ok]


class LocalJudge:
    """Faithfulness + answer relevancy against a local Ollama model."""

    def __init__(
        self,
        settings: Settings | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        s = settings or get_settings()
        self._model_name = model or s.agent_reasoning_model
        # temperature=0.0 is not optional. A judge that returns a different
        # number for the same input cannot detect a regression — the noise it
        # adds is indistinguishable from the change you are trying to measure.
        self._chat = ChatOllama(
            model=self._model_name, base_url=s.ollama_base_url, temperature=temperature
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    async def _decompose(self, answer: str) -> list[str]:
        result = await self._chat.with_structured_output(ClaimList).ainvoke(
            [
                SystemMessage(content=DECOMPOSE_RUBRIC),
                HumanMessage(content=f"ANSWER:\n{answer}"),
            ]
        )
        claims = result.claims if isinstance(result, ClaimList) else ClaimList(**result).claims
        # Sorted so the verification order is stable across runs — same input,
        # same sequence of calls, same score.
        return sorted({c.strip() for c in claims if c and c.strip()})

    async def _verify(self, claim: str, context: str) -> bool:
        result = await self._chat.with_structured_output(ClaimVerdict).ainvoke(
            [
                SystemMessage(content=VERIFY_RUBRIC),
                HumanMessage(content=f"CONTEXT:\n{context}\n\nCLAIM:\n{claim}"),
            ]
        )
        verdict = result if isinstance(result, ClaimVerdict) else ClaimVerdict(**result)
        return bool(verdict.supported)

    async def faithfulness(self, answer: str, contexts: Sequence[str]) -> tuple[float, list, list]:
        """Fraction of the answer's claims that the context supports.

        Returns (score, claims, per-claim verdicts). An answer with no checkable
        claims returns 1.0 — it asserts nothing, so it invents nothing. That is
        the same convention RAGAS uses, and it is why faithfulness must always
        be read next to answer relevancy: "I don't know" is perfectly faithful
        and perfectly useless.
        """
        context = "\n\n".join(contexts) if contexts else ""
        if not context.strip() or not answer.strip():
            return 0.0, [], []

        claims = await self._decompose(answer)
        if not claims:
            return 1.0, [], []

        # Independent calls: no claim can see another's verdict, which removes
        # the ordering effect a single "grade this list" call would introduce.
        verdicts = await asyncio.gather(
            *(self._verify(c, context) for c in claims), return_exceptions=True
        )
        supported = [v is True for v in verdicts]
        return sum(supported) / len(supported), claims, supported

    async def answer_relevancy(self, question: str, answer: str) -> float:
        """How well the answer addresses the question, ignoring correctness.

        Differs from RAGAS deliberately. RAGAS reverse-generates questions from
        the answer and compares their embeddings to the original; that needs an
        embedding round-trip and is sensitive to the embedding model's handling
        of the language — which, for Arabic, is exactly the variable under test
        elsewhere in this harness. A direct rubric keeps the two measurements
        independent. The trade-off is that it inherits the judge's leniency
        instead of the embedder's blind spots.
        """
        if not answer.strip():
            return 0.0
        result = await self._chat.with_structured_output(RelevancyVerdict).ainvoke(
            [
                SystemMessage(content=RELEVANCY_RUBRIC),
                HumanMessage(content=f"QUESTION:\n{question}\n\nANSWER:\n{answer}"),
            ]
        )
        verdict = result if isinstance(result, RelevancyVerdict) else RelevancyVerdict(**result)
        return max(0.0, min(1.0, float(verdict.score)))

    async def judge(
        self, record_id: str, question: str, answer: str, contexts: Sequence[str]
    ) -> JudgedAnswer:
        try:
            faith, claims, supported = await self.faithfulness(answer, contexts)
            relevancy = await self.answer_relevancy(question, answer)
            return JudgedAnswer(record_id, faith, relevancy, claims, supported)
        except Exception as exc:                       # noqa: BLE001
            log.warning("judge.failed", record_id=record_id, error=str(exc))
            return JudgedAnswer(record_id, None, None, error=f"{type(exc).__name__}: {exc}")


async def score_answers(
    outcomes: Sequence,
    settings: Settings | None = None,
    verbose: bool = True,
) -> tuple[dict[str, float], list[JudgedAnswer]]:
    """Judge a list of CragOutcome-shaped objects.

    Abstentions and errored turns are excluded: an abstention is the correct
    response to an unanswerable question, and scoring it would penalise the
    system for behaving well. The counts are returned so the report can show
    how much of the set was actually scored.
    """
    settings = settings or get_settings()
    judge = LocalJudge(settings)

    scorable = [o for o in outcomes if o.answer.strip() and not o.abstained and not o.error]
    judged: list[JudgedAnswer] = []

    for index, outcome in enumerate(scorable, start=1):
        if verbose:
            print(f"    judging [{index}/{len(scorable)}] {outcome.record_id}", flush=True)
        judged.append(
            await judge.judge(
                outcome.record_id, outcome.question, outcome.answer, outcome.contexts
            )
        )

    faith = [j.faithfulness for j in judged if j.faithfulness is not None]
    rel = [j.answer_relevancy for j in judged if j.answer_relevancy is not None]

    scores: dict[str, float] = {
        "_scored": float(len(faith)),
        "_abstained": float(sum(1 for o in outcomes if o.abstained)),
        "_judge_errors": float(sum(1 for j in judged if j.error)),
    }
    if faith:
        scores["faithfulness"] = sum(faith) / len(faith)
    if rel:
        scores["answer_relevancy"] = sum(rel) / len(rel)
    return scores, judged
