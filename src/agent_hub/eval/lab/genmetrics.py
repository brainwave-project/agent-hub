"""Generation metrics — faithfulness, answer relevance, context precision/recall, correctness.

Every metric here returns its *working*, not only its number: the claims it
extracted, the verdict on each, the per-chunk decisions, the generated
questions and their cosines. "Why is faithfulness 0.67?" must have an answer
you can point at, and so must "why is it lying to me?".

The five RAGAS-style metrics and the question each one answers:

    faithfulness        Is every claim in the ANSWER supported by the CONTEXT?
    answer relevance    Does the ANSWER address the QUESTION?
    context precision   Are the useful CONTEXT chunks ranked at the top?
    context recall      Does the CONTEXT contain everything the REFERENCE says?
    answer correctness  Does the ANSWER agree with the REFERENCE?

Each compares two of four texts (question, context, answer, reference). That is
the whole trick to reading them: a metric is blind to every text it does not
read. Faithfulness never reads the reference, so a faithful answer built on a
stale document scores 1.0. Answer relevance never reads the context or the
reference, so a confident wrong answer scores high. Those blind spots are the
failure modes — see FAILURE_MODES below and the scenarios in `scenarios.py`.

Label-based twins (`*_labels`) compute context precision and recall from the
golden set's evidence spans, with no LLM at all. When the LLM version and the
label version disagree, one of them is wrong — and finding out which is the
cheapest judge validation there is.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from agent_hub.eval.lab.chunking import norm_text
from agent_hub.eval.lab.llm import LLM, LLMError, cosine, default_model

__all__ = [
    "METRIC_PROMPT_VERSION",
    "PROMPTS",
    "FAILURE_MODES",
    "GEN_METRICS",
    "MetricJudge",
    "MetricResult",
    "is_abstention",
    "split_statements",
    "faithfulness",
    "answer_relevance",
    "answer_relevance_embedding",
    "context_precision",
    "context_recall",
    "answer_correctness",
    "answer_verdict",
    "context_precision_labels",
    "context_recall_labels",
    "average_precision",
]

#: Bump when any prompt below changes. It is part of the judge fingerprint, so
#: runs scored with different prompts refuse to be compared by the gate.
METRIC_PROMPT_VERSION = "genmetrics-v1"

PROMPTS: dict[str, str] = {
    "decompose": (
        "You split an answer into atomic factual claims. A claim is a single assertion "
        "that could be checked as true or false on its own. Split compound sentences. "
        "Drop hedges, greetings, advice to 'check the policy', and citation markers such as "
        "[doc:abc]. If the answer states no checkable facts (for example it says it does not "
        'know), return an empty list. Reply as JSON: {"claims": ["...", "..."]}'
    ),
    "verify": (
        "You verify whether ONE claim is supported by the provided context. Supported means "
        "the context states it or directly entails it. Be strict: if the claim adds any "
        "specific detail — a number, a date, a name, a condition — that the context does not "
        "contain, it is NOT supported. Plausibility is not support. World knowledge is not "
        'support. Reply as JSON: {"reason": "<one sentence>", "supported": true|false}'
    ),
    "relevance": (
        "You judge whether an answer addresses the question that was asked. Judge only "
        "relevance, never truth: a wrong answer to the right question is still relevant. "
        "Penalise answers that dodge, restate the question, answer a different question, or "
        "bury the response in unrelated material. Length is not quality.\n"
        "Scale: 5 = directly and completely addresses it; 4 = addresses it with minor "
        "digressions; 3 = partly addresses it; 2 = mostly off-target or evasive; "
        "1 = ignores the question.\n"
        'Reply as JSON: {"reason": "<one sentence>", "score": 1-5}'
    ),
    "question_gen": (
        "Read an answer and write the questions it would be a direct reply to. Write each "
        "question the way a user would ask it. If the answer is evasive, says it does not "
        "know, or gives no information, set noncommittal to true.\n"
        'Reply as JSON: {"questions": ["...", "..."], "noncommittal": true|false}'
    ),
    "context_useful": (
        "You are given a question, a reference answer and ONE passage of retrieved context. "
        "Decide whether the passage itself STATES the facts in the reference answer. Do not "
        "infer, assume or fill gaps: if you would have to guess a fact, the passage is NOT "
        "useful. A passage on the same topic that does not state the facts is NOT useful. A "
        "passage that contradicts the reference (for example an outdated version) is NOT "
        "useful.\n"
        'Reply as JSON: {"reason": "<one sentence>", "useful": true|false}'
    ),
    "attribute": (
        "You are given ONE statement taken from a reference answer, and a retrieved context. "
        "Decide whether the context explicitly states this statement (or states facts that "
        "logically entail it). Implication, guessing and topic overlap are NOT attribution. Be "
        "strict about numbers, dates, names and conditions.\n"
        'Reply as JSON: {"reason": "<one sentence>", "attributed": true|false}'
    ),
    "claim_vs_reference": (
        "You are given a question, ONE claim from an answer to it, and the reference answer. "
        "Decide whether what the claim says is TRUE according to the reference. A claim "
        "usually covers only PART of the reference: that is expected and still supported — "
        "do not penalise a claim for what it leaves out. Wording differences and details that "
        "only restate the question do not matter. The claim is NOT supported only when it "
        "contradicts the reference, or states a number, date, name or condition that the "
        "reference does not give. "
        'Reply as JSON: {"reason": "<one sentence>", "supported": true|false}'
    ),
    "reference_vs_answer": (
        "You are given ONE statement from a reference answer, and an answer. Decide whether "
        "the answer conveys this statement — the same facts, in any wording. Extra wording or "
        "extra facts in the answer do not matter; only ask whether this statement's facts are "
        "in it.\n"
        'Reply as JSON: {"reason": "<one sentence>", "covered": true|false}'
    ),
    "answer_verdict": (
        "You grade an answer against a reference answer. The answer is CORRECT when it gives "
        "the same core answer as the reference: the same number, date, yes/no or condition "
        "that the question asks about. Extra correct detail is fine. Missing minor detail is "
        "fine. A different number, a wrong condition, or a contradiction is INCORRECT. "
        "Saying it does not know is INCORRECT when the reference contains an answer.\n"
        'Reply as JSON: {"reason": "<one sentence>", "correct": true|false}'
    ),
}

#: The failure mode of each metric — shown next to every score in the UI.
FAILURE_MODES: dict[str, str] = {
    "faithfulness": (
        "Blind to the reference. A faithful answer built on a stale or wrong chunk scores 1.0. "
        "An answer with no claims ('I don't know') scores 1.0 by convention — read it next to "
        "answer relevance. Claim splitting is itself an LLM call: split one claim into three "
        "and the score moves."
    ),
    "answer_relevance": (
        "Blind to truth. A confident, specific, wrong answer is perfectly relevant. The rubric "
        "judge also inherits LLM leniency; the embedding method inherits the embedder's view "
        "of the language and rewards topic overlap."
    ),
    "context_precision": (
        "Needs a reference. Rank-weighted: the same chunks in a different order score "
        "differently. Punishes extra harmless chunks. Says nothing about whether the useful "
        "chunk contains ALL of the answer."
    ),
    "context_recall": (
        "Only as good as the reference. A vague or incomplete reference gives high recall for "
        "free. Statement splitting is an LLM call. Blind to noise: 20 chunks with the answer in "
        "one still score 1.0."
    ),
    "answer_correctness": (
        "Needs a reference and inherits its errors. Claim-level F1 gives partial credit to a "
        "wrong answer (the wrong number is one claim among several) and penalises omitting "
        "side-notes in the reference as much as a wrong core fact. The same facts in different "
        "words split into different claims and can score differently. Read it next to the "
        "binary answer verdict."
    ),
}

GEN_METRICS: tuple[str, ...] = (
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
    "answer_correctness",
    "context_precision_labels",
    "context_recall_labels",
)

_ABSTAIN = re.compile(
    r"(don't|do not|doesn't|does not) (have|contain) (enough|sufficient|the) information"
    r"|not (covered|mentioned|specified|addressed|stated) in the (provided |available )?"
    r"(context|documents?|policy|policies)"
    r"|(cannot|can't|could not|unable to) (find|answer|determine)"
    r"|no information (about|on|regarding)",
    re.IGNORECASE,
)


def is_abstention(answer: str) -> bool:
    """True when the answer declines to answer. Deliberately a visible regex, not an LLM."""
    return bool(_ABSTAIN.search(answer or ""))


def split_statements(reference: str) -> list[str]:
    """Split a reference into sentences. Deterministic, so recall's denominator never drifts."""
    parts = re.split(r"(?<=[.!?])\s+", (reference or "").strip())
    return [p.strip() for p in parts if len(p.strip()) > 2]


@dataclass
class MetricResult:
    name: str
    score: float | None
    steps: list[dict] = field(default_factory=list)
    note: str = ""
    flags: list[str] = field(default_factory=list)
    calls: int = 0
    error: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class MetricJudge:
    """The LLM plus the settings every metric call uses. Its fingerprint travels with scores."""

    def __init__(
        self, llm: LLM, model: str | None = None, temperature: float = 0.0, seed: int = 7
    ) -> None:
        self.llm = llm
        self.model = model or default_model()
        self.temperature = temperature
        self.seed = seed
        self.calls = 0

    @property
    def config(self) -> dict:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "seed": self.seed,
            "prompt_version": METRIC_PROMPT_VERSION,
        }

    @property
    def fingerprint(self) -> str:
        """Same fingerprint ⇔ same judge. Change the model, temperature or a prompt and it moves."""
        payload = json.dumps({**self.config, "prompts": PROMPTS}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def ask(self, prompt: str, user: str) -> dict:
        self.calls += 1
        return self.llm.chat_json(
            PROMPTS[prompt], user, model=self.model, temperature=self.temperature, seed=self.seed
        )


def _bool(data: dict, key: str) -> bool:
    value = data.get(key)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


def _context_block(contexts: Sequence[str]) -> str:
    if not contexts:
        return "<no context>"
    return "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))


def _guard(name: str, fn):
    """Run one metric; an LLM failure becomes an errored result, never an exception or a 0."""
    try:
        return fn()
    except LLMError as exc:
        return MetricResult(name, None, error=str(exc), note="judge call failed — not scored")


# --- faithfulness --------------------------------------------------------------------


def decompose(judge: MetricJudge, text: str) -> list[str]:
    data = judge.ask("decompose", f"ANSWER:\n{text}")
    claims = data.get("claims", [])
    if not isinstance(claims, list):
        claims = []
    # Sorted and de-duplicated: the same answer always produces the same call sequence.
    return sorted({str(c).strip() for c in claims if str(c).strip()})


def faithfulness(judge: MetricJudge, answer: str, contexts: Sequence[str]) -> MetricResult:
    """Share of the answer's claims that the context supports.

    FAILS WHEN: the context itself is wrong (faithful to a stale policy = 1.0), or
    the answer asserts nothing ("I don't know" = 1.0, flagged `vacuous`).
    """

    def run() -> MetricResult:
        before = judge.calls
        if not answer.strip():
            return MetricResult("faithfulness", None, note="empty answer — not scored")
        claims = decompose(judge, answer)
        flags = ["abstention"] if is_abstention(answer) else []
        if not claims:
            return MetricResult(
                "faithfulness",
                1.0,
                note="No checkable claims, so nothing is unsupported: 1.0 by convention. "
                "This is why faithfulness must be read next to answer relevance.",
                flags=[*flags, "vacuous"],
                calls=judge.calls - before,
            )
        context = _context_block(contexts)
        steps = []
        for claim in claims:
            # One call per claim: no claim can see another's verdict, and there is
            # no list position to be judged favourably for.
            verdict = judge.ask("verify", f"CONTEXT:\n{context}\n\nCLAIM:\n{claim}")
            steps.append(
                {
                    "claim": claim,
                    "supported": _bool(verdict, "supported"),
                    "reason": str(verdict.get("reason", "")),
                }
            )
        supported = sum(1 for s in steps if s["supported"])
        return MetricResult(
            "faithfulness",
            supported / len(steps),
            steps=steps,
            note=f"{supported} of {len(steps)} claims supported by the context.",
            flags=flags,
            calls=judge.calls - before,
        )

    return _guard("faithfulness", run)


# --- answer relevance ----------------------------------------------------------------


def answer_relevance(judge: MetricJudge, question: str, answer: str) -> MetricResult:
    """Rubric judge, 1-5 mapped to 0-1. FAILS WHEN: the answer is relevant and wrong."""

    def run() -> MetricResult:
        before = judge.calls
        if not answer.strip():
            return MetricResult("answer_relevance", 0.0, note="empty answer")
        data = judge.ask("relevance", f"QUESTION:\n{question}\n\nANSWER:\n{answer}")
        try:
            raw = int(round(float(data.get("score", 1))))
        except (TypeError, ValueError):
            raw = 1
        raw = max(1, min(5, raw))
        return MetricResult(
            "answer_relevance",
            (raw - 1) / 4,
            steps=[{"rubric_score": raw, "reason": str(data.get("reason", ""))}],
            note=f"Rubric score {raw}/5 → ({raw} − 1) / 4 = {(raw - 1) / 4:.2f}.",
            flags=["abstention"] if is_abstention(answer) else [],
            calls=judge.calls - before,
        )

    return _guard("answer_relevance", run)


def answer_relevance_embedding(
    judge: MetricJudge, question: str, answer: str, n: int = 3
) -> MetricResult:
    """RAGAS's method: generate questions from the answer, embed, mean cosine to the original.

    FAILS WHEN: the answer is on-topic but wrong (the generated questions are still
    about the right topic), and on languages the embedder handles poorly. Cosines
    from nomic-embed-text rarely fall below ~0.4 even for unrelated text, so the
    scale is compressed: 0.75 is not "75% relevant".
    """

    def run() -> MetricResult:
        before = judge.calls
        if not answer.strip():
            return MetricResult("answer_relevance_embedding", 0.0, note="empty answer")
        data = judge.ask("question_gen", f"Write {n} questions.\n\nANSWER:\n{answer}")
        questions = [str(q).strip() for q in data.get("questions", []) if str(q).strip()][:n]
        noncommittal = _bool(data, "noncommittal")
        if not questions:
            return MetricResult(
                "answer_relevance_embedding",
                0.0,
                note="The model generated no questions.",
                calls=judge.calls - before,
            )
        vectors = judge.llm.embed([question, *questions])
        base = vectors[0]
        steps = [
            {"generated_question": q, "cosine": cosine(base, v)}
            for q, v in zip(questions, vectors[1:], strict=False)
        ]
        mean = sum(s["cosine"] for s in steps) / len(steps)
        score = 0.0 if noncommittal else mean
        return MetricResult(
            "answer_relevance_embedding",
            score,
            steps=steps,
            note=(
                "Noncommittal answer → 0 (RAGAS convention)."
                if noncommittal
                else f"Mean cosine of {len(steps)} generated questions to the real one."
            ),
            flags=["noncommittal"] if noncommittal else [],
            calls=judge.calls - before,
        )

    return _guard("answer_relevance_embedding", run)


# --- context precision / recall ------------------------------------------------------


def average_precision(verdicts: Sequence[bool]) -> tuple[float, list[float]]:
    """RAGAS context precision: Σ_k precision@k · v_k / (number of useful chunks)."""
    hits = 0
    running: list[float] = []
    total = 0.0
    for k, useful in enumerate(verdicts, start=1):
        if useful:
            hits += 1
        p_at_k = hits / k
        running.append(p_at_k)
        if useful:
            total += p_at_k
    return (total / hits if hits else 0.0), running


def context_precision(
    judge: MetricJudge, question: str, contexts: Sequence[str], reference: str
) -> MetricResult:
    """Are the chunks that helped reach the reference ranked first?

    FAILS WHEN: there is no reference (it cannot be computed honestly), and on
    harmless extra chunks, which lower it even when the answer is perfect.
    """

    def run() -> MetricResult:
        before = judge.calls
        if not contexts:
            return MetricResult("context_precision", None, note="no context retrieved")
        if not reference.strip():
            return MetricResult("context_precision", None, note="needs a reference answer")
        verdicts = []
        steps = []
        for rank, chunk in enumerate(contexts, start=1):
            data = judge.ask(
                "context_useful",
                f"QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference}\n\nPASSAGE:\n{chunk}",
            )
            useful = _bool(data, "useful")
            verdicts.append(useful)
            steps.append({"rank": rank, "useful": useful, "reason": str(data.get("reason", ""))})
        score, running = average_precision(verdicts)
        for step, p in zip(steps, running, strict=True):
            step["precision_at_k"] = p
        useful_n = sum(verdicts)
        return MetricResult(
            "context_precision",
            score,
            steps=steps,
            note=f"{useful_n} of {len(contexts)} chunks useful; mean precision@k at the useful ranks.",
            calls=judge.calls - before,
        )

    return _guard("context_precision", run)


def context_recall(judge: MetricJudge, contexts: Sequence[str], reference: str) -> MetricResult:
    """Share of the reference's statements that the context supports.

    FAILS WHEN: the reference is vague or incomplete — fewer, softer statements
    are easier to attribute, so a lazy reference inflates recall.
    """

    def run() -> MetricResult:
        before = judge.calls
        statements = split_statements(reference)
        if not statements:
            return MetricResult("context_recall", None, note="needs a reference answer")
        if not contexts:
            return MetricResult(
                "context_recall",
                0.0,
                steps=[
                    {"statement": s, "attributed": False, "reason": "no context"}
                    for s in statements
                ],
                note="no context retrieved",
            )
        context = _context_block(contexts)
        steps = []
        for statement in statements:
            data = judge.ask("attribute", f"STATEMENT:\n{statement}\n\nCONTEXT:\n{context}")
            steps.append(
                {
                    "statement": statement,
                    "attributed": _bool(data, "attributed"),
                    "reason": str(data.get("reason", "")),
                }
            )
        hit = sum(1 for s in steps if s["attributed"])
        return MetricResult(
            "context_recall",
            hit / len(steps),
            steps=steps,
            note=f"{hit} of {len(steps)} reference statements found in the context.",
            calls=judge.calls - before,
        )

    return _guard("context_recall", run)


# --- correctness ---------------------------------------------------------------------


def answer_correctness(
    judge: MetricJudge, question: str, answer: str, reference: str
) -> MetricResult:
    """Claim-level F1 of the answer against the reference: TP / (TP + ½(FP + FN)).

    TP = answer claim the reference supports; FP = answer claim it does not;
    FN = reference statement the answer leaves out.
    """

    def run() -> MetricResult:
        before = judge.calls
        if not reference.strip():
            return MetricResult("answer_correctness", None, note="needs a reference answer")
        if not answer.strip():
            return MetricResult("answer_correctness", 0.0, note="empty answer")
        claims = decompose(judge, answer)
        statements = split_statements(reference)
        steps: list[dict] = []
        tp = fp = fn = 0
        for claim in claims:
            data = judge.ask(
                "claim_vs_reference",
                f"QUESTION:\n{question}\n\nCLAIM:\n{claim}\n\nREFERENCE ANSWER:\n{reference}",
            )
            ok = _bool(data, "supported")
            tp += ok
            fp += not ok
            steps.append(
                {
                    "side": "answer claim",
                    "text": claim,
                    "label": "TP" if ok else "FP",
                    "reason": str(data.get("reason", "")),
                }
            )
        for statement in statements:
            data = judge.ask("reference_vs_answer", f"STATEMENT:\n{statement}\n\nANSWER:\n{answer}")
            covered = _bool(data, "covered")
            fn += not covered
            steps.append(
                {
                    "side": "reference statement",
                    "text": statement,
                    "label": "covered" if covered else "FN",
                    "reason": str(data.get("reason", "")),
                }
            )
        denom = tp + 0.5 * (fp + fn)
        score = tp / denom if denom else 0.0
        return MetricResult(
            "answer_correctness",
            score,
            steps=steps,
            note=f"TP={tp}, FP={fp}, FN={fn} → {tp} / ({tp} + ½·({fp} + {fn})) = {score:.2f}.",
            flags=["abstention"] if is_abstention(answer) else [],
            calls=judge.calls - before,
        )

    return _guard("answer_correctness", run)


def answer_verdict(judge: MetricJudge, question: str, answer: str, reference: str) -> MetricResult:
    """Binary: does the answer give the reference's core answer? The pipeline's 'answer_ok'."""

    def run() -> MetricResult:
        before = judge.calls
        if not reference.strip():
            return MetricResult("answer_verdict", None, note="needs a reference answer")
        data = judge.ask(
            "answer_verdict",
            f"QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference}\n\nANSWER:\n{answer}",
        )
        correct = _bool(data, "correct")
        return MetricResult(
            "answer_verdict",
            1.0 if correct else 0.0,
            steps=[{"correct": correct, "reason": str(data.get("reason", ""))}],
            calls=judge.calls - before,
        )

    return _guard("answer_verdict", run)


# --- label-based twins (no LLM) --------------------------------------------------------


def _units(evidence: Sequence[tuple[str, str]], relevant: dict[str, int]) -> list[tuple[str, str]]:
    with_spans = {d for d, _ in evidence}
    return list(evidence) + [(d, "") for d in sorted(relevant) if d not in with_spans]


def _contains(unit: tuple[str, str], doc_id: str, text: str) -> bool:
    doc, span = unit
    if doc != doc_id:
        return False
    return True if not span else norm_text(span) in norm_text(text)


def context_precision_labels(
    contexts: Sequence[tuple[str, str]],
    evidence: Sequence[tuple[str, str]],
    relevant: dict[str, int],
) -> MetricResult:
    """Context precision from labels: a chunk is useful if it contains an evidence unit."""
    if not contexts:
        return MetricResult("context_precision_labels", None, note="no context")
    units = _units(evidence, relevant)
    if not units:
        return MetricResult("context_precision_labels", None, note="unanswerable — no labels")
    verdicts = [any(_contains(u, d, t) for u in units) for d, t in contexts]
    score, running = average_precision(verdicts)
    return MetricResult(
        "context_precision_labels",
        score,
        steps=[
            {"rank": k, "doc_id": d, "useful": v, "precision_at_k": p}
            for k, ((d, _), v, p) in enumerate(zip(contexts, verdicts, running, strict=True), 1)
        ],
        note="Useful = contains a verbatim evidence span (or is a relevant doc without spans).",
    )


def context_recall_labels(
    contexts: Sequence[tuple[str, str]],
    evidence: Sequence[tuple[str, str]],
    relevant: dict[str, int],
) -> MetricResult:
    """Context recall from labels: share of evidence units present in the context."""
    units = _units(evidence, relevant)
    if not units:
        return MetricResult("context_recall_labels", None, note="unanswerable — no labels")
    steps = [
        {
            "doc_id": u[0],
            "span": u[1] or "(whole document)",
            "found": any(_contains(u, d, t) for d, t in contexts),
        }
        for u in units
    ]
    hit = sum(1 for s in steps if s["found"])
    return MetricResult(
        "context_recall_labels",
        hit / len(steps),
        steps=steps,
        note=f"{hit} of {len(steps)} evidence units present in the context.",
    )
