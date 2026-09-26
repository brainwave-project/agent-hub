"""An instrumented CRAG-shaped pipeline — so a failure can be traced to the stage that caused it.

End-to-end scoring answers "how often is the answer wrong?". It cannot answer
"where did it go wrong?", and that is the only question you can act on. This
module runs the same stages CRAG runs, records what each stage received and
produced, and judges each stage *given its input*:

    ingestion   Is every evidence span intact in some chunk of an INDEXED document?
    retrieval   Do the stage-1 candidates contain the evidence?
    rerank      Did reranking + the top-n cut keep the evidence in the context?
    grade       Did the relevance grader accept a context that holds the evidence?
    generation  Given good context, is the answer correct (and faithful)?

`blame` is the first stage that failed. A stage whose input was already broken
is `skip`, not `fail`: retrieval cannot be faulted for missing a chunk that was
never indexed. That conditional judgement is the whole point — without it every
failure shows up as a failure of every stage after the first broken one.

Evidence is the golden set's verbatim spans, so the component checks need no
LLM: a chunk "has the evidence" when it contains the span. Only the grader,
the generator and the answer judge call a model.

Differences from agent-hub's CRAG service, stated so nobody mistakes the lab
for production: one retrieval (no query-rewrite loop — each stage is then
attributable exactly once), document chunks from the lab's own index owner,
and prompts from `adapters.prompts.DEFAULT_PROMPTS` (not Langfuse).

**Fault injection** breaks exactly one stage on purpose. If attribution is
right, a run with `fault="rerank"` blames rerank. `fault_matrix` checks this
for every fault at once — a diagnostic you have not tested on a known fault is
a diagnostic you are trusting on faith.
"""

from __future__ import annotations

import re
import statistics
import time
import uuid
from collections import Counter
from collections.abc import Callable

from agent_hub.adapters.prompts import (
    CRAG_CHECK_GROUNDED,
    CRAG_GENERATE,
    CRAG_GRADE,
    DEFAULT_PROMPTS,
)
from agent_hub.application.crag.nodes import ABSTAIN_MESSAGE
from agent_hub.eval.lab import genmetrics as gm
from agent_hub.eval.lab.chunking import chunk_text, locate_evidence, norm_text
from agent_hub.eval.lab.genmodels import (
    STAGES,
    CtxChunk,
    GenItemResult,
    GenRun,
    PipelineSpec,
    StageResult,
)
from agent_hub.eval.lab.jobs import Job
from agent_hub.eval.lab.llm import LLM, LLMError
from agent_hub.eval.lab.models import GoldenItem, IndexedDoc, now_iso
from agent_hub.eval.lab.store import Bundle
from agent_hub.eval.stats import mean_ci

__all__ = [
    "GENERATOR_PROMPTS",
    "evidence_units",
    "run_item",
    "run_generation",
    "components_summary",
    "aggregate_gen",
    "fault_matrix",
]

_STRICT = (
    DEFAULT_PROMPTS[CRAG_GENERATE]
    + f" If the context does not contain the answer, reply with exactly: {ABSTAIN_MESSAGE}"
)

GENERATOR_PROMPTS: dict[str, str] = {
    "strict": _STRICT,
    "lenient": (
        "You are a helpful HR assistant. Use the context if it helps and your general "
        "knowledge otherwise. Always give a definite, specific answer. Never say you do not "
        "know."
    ),
    "verbose": _STRICT
    + " Write a thorough answer of at least 150 words: explain the background, the reasons "
    "behind the rule and any related advice.",
}


_WORD = re.compile(r"\w+", re.UNICODE)


# --- evidence units ---------------------------------------------------------------------


def evidence_units(item: GoldenItem) -> list[tuple[str, str]]:
    """One unit per evidence span; a relevant document without spans is one whole-doc unit."""
    units = [(e.doc_id, e.text) for e in item.evidence if e.doc_id in item.relevant]
    with_spans = {d for d, _ in units}
    return units + [(d, "") for d in sorted(item.relevant) if d not in with_spans]


def _units_in(doc_id: str, text: str, units: list[tuple[str, str]]) -> list[int]:
    body = norm_text(text)
    return [
        n
        for n, (ud, span) in enumerate(units)
        if ud == doc_id and (not span or norm_text(span) in body)
    ]


def _label(unit: tuple[str, str]) -> str:
    doc, span = unit
    return f"{doc}: “{span}”" if span else f"{doc} (whole document)"


def _covered(chunks: list[CtxChunk]) -> set[int]:
    return {u for c in chunks for u in c.evidence_units}


# --- stages -------------------------------------------------------------------------------


def _ingestion(bundle: Bundle, units: list[tuple[str, str]], fault: str) -> StageResult:
    index = bundle.indexes.get("off")
    retrievable, problems = [], []
    for n, (doc_id, span) in enumerate(units):
        doc = bundle.corpus.get(doc_id)
        status = index.documents.get(doc_id, IndexedDoc()).status if index else "MISSING"
        if doc is None:
            problems.append(f"{doc_id}: not in the corpus")
            continue
        if status != "INDEXED":
            problems.append(f"{doc_id}: index status {status}")
            continue
        if span:
            where = locate_evidence(doc.body, span, chunk_text(doc.body))
            if not where["found_in_doc"]:
                problems.append(f"span not in {doc_id} (label error)")
                continue
            if where["straddles_boundary"]:
                problems.append(f"span straddles a chunk boundary in {doc_id}")
                continue
        retrievable.append(n)
    data = {"retrievable_units": retrievable, "units": len(units)}
    if fault == "ingestion":
        return StageResult(
            stage="ingestion",
            status="fail",
            detail="Injected fault: the chunks holding the evidence were never indexed.",
            data=data,
        )
    if not retrievable:
        return StageResult(stage="ingestion", status="fail", detail="; ".join(problems), data=data)
    if problems:
        return StageResult(stage="ingestion", status="warn", detail="; ".join(problems), data=data)
    return StageResult(
        stage="ingestion",
        status="ok",
        detail=f"All {len(units)} evidence unit(s) sit whole inside an indexed chunk.",
        data=data,
    )


def _retrieval(candidates: list[CtxChunk], units: list[tuple[str, str]]) -> StageResult:
    covered = _covered(candidates)
    data = {"covered_units": sorted(covered), "units": len(units), "candidates": len(candidates)}
    best = min((c.rank for c in candidates if c.evidence_units), default=None)
    if best is not None:
        data["first_evidence_rank"] = best
    if not covered:
        return StageResult(
            stage="retrieval",
            status="fail",
            detail=f"No evidence in the top {len(candidates)} candidates.",
            data=data,
        )
    if len(covered) < len(units):
        return StageResult(
            stage="retrieval",
            status="warn",
            detail=f"{len(covered)} of {len(units)} evidence units retrieved (first at rank {best}).",
            data=data,
        )
    return StageResult(
        stage="retrieval",
        status="ok",
        detail=f"All evidence retrieved; first evidence chunk at rank {best}.",
        data=data,
    )


def _rerank_stage(
    candidates: list[CtxChunk], contexts: list[CtxChunk], units: list[tuple[str, str]], mode: str
) -> StageResult:
    before, after = _covered(candidates), _covered(contexts)
    lost = before - after
    data = {"kept_units": sorted(after), "lost_units": sorted(lost), "mode": mode}
    if not after:
        return StageResult(
            stage="rerank",
            status="fail",
            detail=f"Evidence was in the candidates but none survived the cut to top {len(contexts)}.",
            data=data,
        )
    if lost:
        return StageResult(
            stage="rerank",
            status="warn",
            detail=f"{len(lost)} evidence unit(s) dropped by rerank + top-n cut.",
            data=data,
        )
    return StageResult(
        stage="rerank",
        status="ok",
        detail="Every retrieved evidence unit reached the context.",
        data=data,
    )


# --- LLM steps ----------------------------------------------------------------------------


def _format_context(chunks: list[CtxChunk]) -> str:
    # Same shape as adapters.ollama._format_context — the generator sees what CRAG's sees.
    if not chunks:
        return "<no context retrieved>"
    return "\n\n".join(f"[{c.rank}] (doc:{c.doc_id} score:{c.score:.2f})\n{c.text}" for c in chunks)


def _lexical(query: str, text: str) -> float:
    words = {w for w in _WORD.findall(query.casefold()) if len(w) >= 4}
    if not words:
        return 0.0
    body = set(_WORD.findall(text.casefold()))
    return len(words & body) / len(words)


def _rerank(
    llm: LLM, model: str | None, query: str, candidates: list[CtxChunk], mode: str
) -> tuple[list[CtxChunk], list[dict]]:
    if mode == "lexical":
        scored = [(c, _lexical(query, c.text)) for c in candidates]
        notes = [{"chunk_id": c.chunk_id, "rerank_score": s} for c, s in scored]
        return [c for c, _ in sorted(scored, key=lambda cs: (-cs[1], cs[0].rank))], notes
    if mode == "llm":
        scored = []
        for c in candidates:
            try:
                data = llm.chat_json(
                    "Grade how well a passage answers a question. 3 = answers it fully, "
                    "2 = partly, 1 = on topic but does not answer, 0 = unrelated. "
                    'Reply as JSON: {"grade": 0-3}',
                    f"QUESTION:\n{query}\n\nPASSAGE:\n{c.text}",
                    model=model,
                    temperature=0.0,
                )
                grade = float(data.get("grade", 0))
            except (LLMError, TypeError, ValueError):
                grade = 0.0
            scored.append((c, grade))
        notes = [{"chunk_id": c.chunk_id, "rerank_score": s} for c, s in scored]
        return [c for c, _ in sorted(scored, key=lambda cs: (-cs[1], cs[0].rank))], notes
    return list(candidates), []


def _grade(llm: LLM, model: str | None, question: str, contexts: list[CtxChunk]) -> dict:
    try:
        data = llm.chat_json(
            DEFAULT_PROMPTS[CRAG_GRADE]
            + ' Reply as JSON: {"reason": "<one sentence>", "relevant": true|false}',
            f"QUESTION:\n{question}\n\nCONTEXT:\n{_format_context(contexts)}",
            model=model,
            temperature=0.0,
        )
        return {"relevant": gm._bool(data, "relevant"), "reason": str(data.get("reason", ""))}
    except LLMError as exc:
        # Fails closed, like the service: an unreadable verdict is "not relevant".
        return {"relevant": False, "reason": f"grader error — treated as not relevant ({exc})"}


def _generate(
    llm: LLM, spec: PipelineSpec, question: str, contexts: list[CtxChunk], retry: bool = False
) -> str:
    if spec.fault == "generation":
        system, context = GENERATOR_PROMPTS["lenient"], "<no context retrieved>"
    else:
        system, context = GENERATOR_PROMPTS[spec.generator], _format_context(contexts)
    if retry:
        system += " Your previous answer contained claims the context does not support. Answer again using ONLY the context."
    return llm.chat_text(
        system,
        f"QUESTION:\n{question}\n\nCONTEXT:\n{context}",
        model=spec.model,
        temperature=spec.generate_temperature,
        seed=11 if retry else 7,
    )


def _grounded(llm: LLM, model: str | None, answer: str, contexts: list[CtxChunk]) -> dict:
    try:
        data = llm.chat_json(
            DEFAULT_PROMPTS[CRAG_CHECK_GROUNDED]
            + ' Reply as JSON: {"reason": "<one sentence>", "grounded": true|false}',
            f"ANSWER:\n{answer}\n\nCONTEXT:\n{_format_context(contexts)}",
            model=model,
            temperature=0.0,
        )
        return {"grounded": gm._bool(data, "grounded"), "reason": str(data.get("reason", ""))}
    except LLMError as exc:
        return {"grounded": False, "reason": f"grounding check error ({exc})"}


# --- one item -----------------------------------------------------------------------------


def run_item(
    kh,
    llm: LLM,
    judge: gm.MetricJudge,
    bundle: Bundle,
    item: GoldenItem,
    spec: PipelineSpec,
    owner: str,
    llm_context_metrics: bool = True,
) -> GenItemResult:
    started = time.perf_counter()
    units = evidence_units(item)
    result = GenItemResult(
        item_id=item.id,
        query=item.query,
        answerable=item.answerable,
        ground_truth=item.ground_truth,
        trap=item.trap,
        difficulty=item.difficulty,
        units=[_label(u) for u in units],
    )

    # 1. retrieval (+ simulated ingestion/retrieval faults)
    extra = 10 if spec.fault in ("ingestion", "retrieval") else 0
    hits = kh.search_chunks(item.query, owner, spec.candidate_k + extra, spec.min_score)
    raw = [
        CtxChunk(
            rank=0,
            doc_id=h.doc_id,
            chunk_id=h.chunk_id,
            score=h.score,
            text=h.text,  # full chunk: the grader and generator must see what CRAG sees
            evidence_units=_units_in(h.doc_id, h.text, units),
        )
        for h in hits
    ]
    if spec.fault in ("ingestion", "retrieval"):
        raw = [c for c in raw if not c.evidence_units]
    candidates = [
        c.model_copy(update={"rank": n}) for n, c in enumerate(raw[: spec.candidate_k], 1)
    ]
    result.candidates = candidates

    # 2. rerank + cut
    ordered, rerank_notes = _rerank(llm, spec.model, item.query, candidates, spec.rerank)
    if spec.fault == "rerank":
        ordered = list(reversed(ordered))
    contexts = [c.model_copy(update={"rank": n}) for n, c in enumerate(ordered[: spec.top_n], 1)]
    result.contexts = contexts
    if rerank_notes:
        result.details["rerank"] = {"scores": rerank_notes}

    # 3. grade
    accepted = True
    if spec.fault == "grade":
        result.grade = {
            "relevant": False,
            "reason": "Injected fault: the grader rejects everything.",
        }
        accepted = False
    elif spec.grader:
        result.grade = _grade(llm, spec.model, item.query, contexts)
        accepted = bool(result.grade["relevant"])

    # 4. generate (or abstain, as CRAG does after a rejected grade)
    if accepted:
        answer = _generate(llm, spec, item.query, contexts)
        if spec.grounding and not gm.is_abstention(answer):
            verdict = _grounded(llm, spec.model, answer, contexts)
            regenerated = False
            if not verdict["grounded"]:
                answer = _generate(llm, spec, item.query, contexts, retry=True)
                regenerated = True
            result.grounding = {**verdict, "regenerated": regenerated}
    else:
        answer = ABSTAIN_MESSAGE
    result.answer = answer
    result.abstained = gm.is_abstention(answer)

    # 5. metrics
    ctx_text = [c.text for c in contexts]
    ctx_pairs = [(c.doc_id, c.text) for c in contexts]
    metrics: dict[str, float] = {}
    details: dict[str, dict] = result.details

    def keep(res: gm.MetricResult) -> None:
        details[res.name] = res.as_dict()
        if res.score is not None:
            metrics[res.name] = res.score

    if item.answerable:
        spans = [u for u in units if u[1]]
        keep(gm.context_recall_labels(ctx_pairs, spans, item.relevant))
        keep(gm.context_precision_labels(ctx_pairs, spans, item.relevant))
        metrics["retrieval_recall_labels"] = (
            len(_covered(candidates)) / len(units) if units else 0.0
        )
        if llm_context_metrics and item.ground_truth:
            keep(gm.context_precision(judge, item.query, ctx_text, item.ground_truth))
            keep(gm.context_recall(judge, ctx_text, item.ground_truth))
        if result.abstained:
            metrics["answer_ok"] = 0.0
            metrics["false_abstention"] = 1.0
            metrics["answer_correctness"] = 0.0
            details["answer_correctness"] = {
                "name": "answer_correctness",
                "score": 0.0,
                "note": "Abstained on an answerable question: scored 0 without a judge call.",
            }
        else:
            metrics["false_abstention"] = 0.0
            keep(gm.faithfulness(judge, answer, ctx_text))
            keep(gm.answer_relevance(judge, item.query, answer))
            keep(gm.answer_correctness(judge, item.query, answer, item.ground_truth))
            verdict = gm.answer_verdict(judge, item.query, answer, item.ground_truth)
            details["answer_verdict"] = verdict.as_dict()
            if verdict.score is not None:
                metrics["answer_ok"] = verdict.score
    else:
        metrics["answered_unanswerable"] = 0.0 if result.abstained else 1.0
        metrics["answer_ok"] = 1.0 if result.abstained else 0.0
        if not result.abstained:
            keep(gm.faithfulness(judge, answer, ctx_text))

    result.metrics = metrics

    # 6. stage verdicts, conditional on each stage's input
    result.stages, result.blame, result.outcome, result.flags = _attribute(
        bundle, spec, item, units, result
    )
    result.latency_ms = (time.perf_counter() - started) * 1000
    return result


def _attribute(bundle, spec, item, units, r: GenItemResult):
    stages: list[StageResult] = []
    flags: list[str] = []
    if not item.answerable:
        for s in ("ingestion", "retrieval", "rerank"):
            stages.append(
                StageResult(stage=s, status="skip", detail="Unanswerable: nothing to find.")
            )
        if r.grade is None:
            stages.append(StageResult(stage="grade", status="skip", detail="Grader off."))
        elif r.grade["relevant"]:
            stages.append(
                StageResult(
                    stage="grade",
                    status="fail",
                    detail="False accept: no document answers this, but the grader passed the context.",
                    data=r.grade,
                )
            )
        else:
            stages.append(
                StageResult(
                    stage="grade",
                    status="ok",
                    detail="Correctly rejected an off-topic context.",
                    data=r.grade,
                )
            )
        if r.abstained:
            outcome = "correct_abstention"
            if r.grade is None or r.grade["relevant"]:
                stages.append(StageResult(stage="generation", status="ok", detail="Abstained."))
            else:
                stages.append(
                    StageResult(
                        stage="generation",
                        status="skip",
                        detail="Not reached (abstained after the grade).",
                    )
                )
        else:
            outcome = "answered_unanswerable"
            stages.append(
                StageResult(
                    stage="generation",
                    status="fail",
                    detail="Answered a question no document answers.",
                )
            )
        blame = next((s.stage for s in stages if s.status == "fail"), "none")
        return stages, blame, outcome, flags

    ing = _ingestion(bundle, units, spec.fault)
    stages.append(ing)
    ret = (
        _retrieval(r.candidates, units)
        if ing.status != "fail"
        else StageResult(
            stage="retrieval",
            status="skip",
            detail="Upstream failure: nothing indexed to retrieve.",
        )
    )
    stages.append(ret)
    rr = (
        _rerank_stage(r.candidates, r.contexts, units, spec.rerank)
        if ret.status in ("ok", "warn")
        else StageResult(
            stage="rerank", status="skip", detail="Upstream failure: no evidence to keep."
        )
    )
    stages.append(rr)
    has_evidence = bool(_covered(r.contexts))
    if r.grade is None:
        stages.append(StageResult(stage="grade", status="skip", detail="Grader off."))
    elif not has_evidence:
        stages.append(
            StageResult(
                stage="grade",
                status="skip",
                detail="Context holds no evidence; the grade cannot be faulted.",
                data=r.grade,
            )
        )
    elif r.grade["relevant"]:
        stages.append(
            StageResult(
                stage="grade",
                status="ok",
                detail="Accepted a context that holds the evidence.",
                data=r.grade,
            )
        )
    else:
        stages.append(
            StageResult(
                stage="grade",
                status="fail",
                detail="False rejection: the context held the evidence, the grader said not relevant.",
                data=r.grade,
            )
        )

    ok = r.metrics.get("answer_ok") == 1.0
    if r.abstained:
        outcome = "false_abstention"
    elif ok:
        outcome = "correct"
    else:
        outcome = "incorrect"
    faith = r.metrics.get("faithfulness")
    if outcome == "incorrect" and faith is not None and faith < 1.0:
        flags.append("unsupported_claims")

    grade_ok = r.grade is None or r.grade["relevant"]
    if not has_evidence:
        gen = StageResult(
            stage="generation",
            status="skip",
            detail="Context holds no evidence; the generator cannot be faulted.",
        )
        if ok:
            flags.append("correct_without_evidence")
    elif not grade_ok:
        gen = StageResult(
            stage="generation", status="skip", detail="Not reached (abstained after the grade)."
        )
    elif ok:
        gen = StageResult(
            stage="generation", status="ok", detail="Correct answer from good context."
        )
    elif r.abstained:
        gen = StageResult(
            stage="generation",
            status="fail",
            detail="False abstention: the evidence was in the context.",
        )
    else:
        why = (
            f" Faithfulness {faith:.2f} — it asserted things the context does not say."
            if faith is not None and faith < 1
            else ""
        )
        gen = StageResult(
            stage="generation", status="fail", detail="Incorrect answer from good context." + why
        )
    stages.append(gen)

    if outcome == "correct":
        blame = "none"
    else:
        blame = next((s.stage for s in stages if s.status == "fail"), None) or next(
            (s.stage for s in stages if s.status == "warn"), "generation"
        )
    return stages, blame, outcome, flags


# --- a whole run ---------------------------------------------------------------------------


def aggregate_gen(
    results: list[GenItemResult], resamples: int = 2000
) -> dict[str, dict[str, float]]:
    ok = [r for r in results if r.error is None]
    names = sorted({m for r in ok for m in r.metrics})
    out: dict[str, dict[str, float]] = {}
    for name in names:
        values = [r.metrics[name] for r in ok if name in r.metrics]
        if not values:
            continue
        interval = mean_ci(values, resamples=resamples)
        out[name] = {
            "mean": interval.mean,
            "low": interval.low if len(values) > 1 else interval.mean,
            "high": interval.high if len(values) > 1 else interval.mean,
            "n": interval.n,
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return out


def components_summary(results: list[GenItemResult]) -> dict:
    ok = [r for r in results if r.error is None]
    per_stage = {s: Counter() for s in STAGES}
    for r in ok:
        for st in r.stages:
            per_stage.setdefault(st.stage, Counter())[st.status] += 1
    answerable = [r for r in ok if r.answerable]

    def count(pred) -> int:
        return sum(1 for r in answerable if pred(r))

    def stage(r, name):
        return next((s for s in r.stages if s.stage == name), None)

    funnel = [
        {"step": "answerable questions", "n": len(answerable)},
        {
            "step": "evidence indexed",
            "n": count(lambda r: stage(r, "ingestion").status in ("ok", "warn")),
        },
        {
            "step": "evidence in candidates",
            "n": count(lambda r: any(c.evidence_units for c in r.candidates)),
        },
        {
            "step": "evidence in context",
            "n": count(lambda r: any(c.evidence_units for c in r.contexts)),
        },
        {
            "step": "context accepted",
            "n": count(
                lambda r: (
                    any(c.evidence_units for c in r.contexts)
                    and (r.grade is None or r.grade["relevant"])
                )
            ),
        },
        {"step": "answered correctly", "n": count(lambda r: r.outcome == "correct")},
    ]
    blame = Counter(r.blame for r in ok)
    by_trap: dict[str, Counter] = {}
    for r in ok:
        by_trap.setdefault(r.trap, Counter())[r.blame] += 1
    return {
        "stages": {s: dict(c) for s, c in per_stage.items()},
        "blame": dict(blame),
        "blame_by_trap": {t: dict(c) for t, c in sorted(by_trap.items())},
        "outcomes": dict(Counter(r.outcome for r in ok)),
        "funnel": funnel,
        "flags": dict(Counter(f for r in ok for f in r.flags)),
    }


def run_generation(
    kh,
    llm: LLM,
    store,
    bundle: Bundle,
    spec: PipelineSpec,
    split: str,
    judge: gm.MetricJudge,
    item_ids: list[str] | None = None,
    limit: int | None = None,
    llm_context_metrics: bool = True,
    job: Job | None = None,
    on_start: Callable[[GenRun], None] | None = None,
) -> GenRun:
    state = bundle.indexes.get("off")
    if state is None:
        raise RuntimeError("corpus has no 'off' index — index it in Golden Sets → Index")
    selected = [i for i in bundle.items if split == "all" or i.split == split]
    if item_ids:
        wanted = set(item_ids)
        selected = [i for i in selected if i.id in wanted]
    if limit:
        selected = selected[:limit]
    run = GenRun(
        id=f"gen_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}",
        dataset_id=bundle.meta.id,
        dataset_version=bundle.manifest.version,
        dataset_hash=bundle.current_hash,
        dataset_status=bundle.manifest.status,
        corpus_hash=bundle.corpus_hash("off"),
        spec=spec,
        split=split,  # type: ignore[arg-type]
        item_ids=[i.id for i in selected],
        llm_context_metrics=llm_context_metrics,
        judge=judge.config,
        judge_fingerprint=judge.fingerprint,
    )
    store.save_gen_run(run)
    if on_start:
        on_start(run)
    if job:
        job.progress(0, len(selected), f"{spec.name} on {split}")
    try:
        for n, item in enumerate(selected, start=1):
            try:
                res = run_item(
                    kh, llm, judge, bundle, item, spec, state.owner_id, llm_context_metrics
                )
            except Exception as exc:  # noqa: BLE001 — recorded, excluded from aggregates
                res = GenItemResult(
                    item_id=item.id,
                    query=item.query,
                    answerable=item.answerable,
                    ground_truth=item.ground_truth,
                    trap=item.trap,
                    difficulty=item.difficulty,
                    error=f"{type(exc).__name__}: {exc}",
                )
            run.results.append(res)
            if job:
                job.progress(n, message=f"{item.id}: {res.outcome or res.error}")
            if n % 5 == 0:
                store.save_gen_run(run)
    except Exception as exc:
        run.status = "failed"
        run.error = f"{type(exc).__name__}: {exc}"
        run.finished_at = now_iso()
        store.save_gen_run(run)
        raise
    run.aggregates = aggregate_gen(run.results)
    run.components = components_summary(run.results)
    ok = [r for r in run.results if r.error is None]
    run.diagnostics = {
        "errors": float(len(run.results) - len(ok)),
        "mean_latency_ms": statistics.fmean(r.latency_ms for r in ok) if ok else 0.0,
        "judge_calls": float(judge.calls),
    }
    run.status = "done"
    run.finished_at = now_iso()
    store.save_gen_run(run)
    return run


def fault_matrix(runs: list[GenRun]) -> dict:
    """Injected fault × blamed stage. The diagonal is attribution doing its job."""
    rows = []
    for run in runs:
        ok = [r for r in run.results if r.error is None and r.answerable]
        bad = [r for r in ok if r.outcome != "correct"]
        blamed = Counter(r.blame for r in bad)
        fault = run.spec.fault
        rows.append(
            {
                "run_id": run.id,
                "fault": fault,
                "items": len(ok),
                "failed": len(bad),
                "masked": len(ok) - len(bad),
                "blamed": dict(blamed),
                "accuracy": (blamed.get(fault, 0) / len(bad)) if bad and fault != "none" else None,
            }
        )
    return {"stages": list(STAGES), "rows": rows}
