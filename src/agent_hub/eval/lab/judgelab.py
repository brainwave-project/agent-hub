"""LLM-as-judge design — rubrics, validation against humans, and the biases that break judges.

A judge is a measuring instrument. Before you trust its number you do what you
would do with any instrument: calibrate it against a reference (human labels),
check it reads the same thing twice (self-consistency), and look for known
systematic errors:

    position bias    in a pairwise comparison, preferring whichever answer is shown first
    verbosity bias   scoring a longer answer higher when the extra words add nothing
    leniency         passing answers a careful human would fail
    drift            the same rubric scoring differently after a model / prompt / setting change

This module measures each one on your own samples. It does not "fix" them —
the fixes are design choices (anchored rubrics, reference-guided grading,
per-claim scoring, both-order pairwise, temperature 0, pinned model versions)
whose effect you verify here by re-running the probe.

What validation reports, and why each number:

    accuracy / κ          agreement with the human, and agreement beyond chance
    fail recall (TPR)     of the answers humans FAILED, how many the judge failed —
                          the number that matters for a gate; a lenient judge
                          has high accuracy on a mostly-good set and still misses
                          the failures you built the judge to catch
    pass-rate gap         judge pass rate − human pass rate; the bias your dashboard inherits
    corrected pass rate   Rogan–Gladen: the pass rate implied by the judge's
                          measured TPR/TNR, i.e. what to report if you must use it
    human–human κ         the ceiling: a judge cannot be validated more precisely
                          than two humans agree
"""

from __future__ import annotations

import random
import statistics
import time
import uuid
from collections import defaultdict
from collections.abc import Sequence

from agent_hub.eval.lab.agreement import cohens_kappa, confusion_matrix, interpret_kappa
from agent_hub.eval.lab.genmodels import JudgeRun, JudgeSample, JudgeSet, JudgeSpec
from agent_hub.eval.lab.jobs import Job
from agent_hub.eval.lab.llm import LLM, LLMError, default_model
from agent_hub.eval.lab.models import now_iso
from agent_hub.eval.stats import mean_ci, paired_delta_ci

__all__ = [
    "PADDINGS",
    "build_prompt",
    "judge_one",
    "run_judge",
    "validation_report",
    "consistency_summary",
    "verbosity_probe",
    "position_probe",
    "drift_report",
    "spearman",
    "blind_task",
]

#: Padding added by the verbosity probe. None adds a fact: a judge that scores
#: the padded answer higher is rewarding length, not content.
PADDINGS: dict[str, str] = {
    "filler": (
        "\n\nI hope this helps! It is always a good idea to read the full policy document and to "
        "talk to your manager or the People Team if anything is unclear. Policies are reviewed "
        "from time to time, so staying informed about changes is important for everyone."
    ),
    "restate": (
        "\n\nTo put this in context: this is a common question, and Kestrel's policies are "
        "designed to treat every employee fairly and consistently. The rule above reflects the "
        "company's commitment to clear, transparent and well-documented processes that support "
        "both employees and managers."
    ),
    "authority": (
        "\n\nThis answer is based directly on the official, current Kestrel policy and is "
        "fully accurate and compliant. You can rely on it with confidence."
    ),
}


# --- prompts -----------------------------------------------------------------------------


def _output_contract(spec: JudgeSpec) -> str:
    if spec.mode == "binary":
        return (
            'Reply as JSON: {"reasoning": "<2-3 sentences>", "verdict": "pass"|"fail"}'
            if spec.reasoning_first
            else 'Reply as JSON: {"verdict": "pass"|"fail", "reasoning": "<one sentence>"}'
        )
    scale = f"an integer from 1 to {spec.scale_max}"
    return (
        f'Reply as JSON: {{"reasoning": "<2-3 sentences>", "score": <{scale}>}}'
        if spec.reasoning_first
        else f'Reply as JSON: {{"score": <{scale}>, "reasoning": "<one sentence>"}}'
    )


def _materials(spec: JudgeSpec, sample: JudgeSample) -> str:
    parts = [f"QUESTION:\n{sample.question}"]
    if spec.reference_guided and sample.reference:
        parts.append(f"REFERENCE ANSWER:\n{sample.reference}")
    if spec.show_context and sample.contexts:
        parts.append(
            "CONTEXT:\n" + "\n\n".join(f"[{i}] {c}" for i, c in enumerate(sample.contexts, 1))
        )
    return "\n\n".join(parts)


def build_prompt(
    spec: JudgeSpec, sample: JudgeSample, answer: str | None = None
) -> tuple[str, str]:
    system = (
        "You are an evaluation judge for answers produced by a company HR-policy assistant.\n\n"
        f"RUBRIC:\n{spec.rubric.strip()}\n\n{_output_contract(spec)}"
    )
    user = f"{_materials(spec, sample)}\n\nANSWER:\n{sample.answer if answer is None else answer}"
    return system, user


def _pairwise_prompt(spec: JudgeSpec, sample: JudgeSample, a: str, b: str) -> tuple[str, str]:
    system = (
        "You are an evaluation judge for answers produced by a company HR-policy assistant.\n\n"
        f"RUBRIC:\n{spec.rubric.strip()}\n\n"
        "Compare two answers to the same question and decide which one is better according to "
        'the rubric, or "tie" if they are equally good.\n'
        'Reply as JSON: {"reasoning": "<2-3 sentences>", "winner": "A"|"B"|"tie"}'
    )
    user = f"{_materials(spec, sample)}\n\nANSWER A:\n{a}\n\nANSWER B:\n{b}"
    return system, user


def judge_one(
    llm: LLM,
    spec: JudgeSpec,
    sample: JudgeSample,
    answer: str | None = None,
    temperature: float | None = None,
    seed: int | None = None,
) -> dict:
    system, user = build_prompt(spec, sample, answer)
    data = llm.chat_json(
        system,
        user,
        model=spec.model or default_model(),
        temperature=spec.temperature if temperature is None else temperature,
        seed=spec.seed if seed is None else seed,
    )
    reasoning = str(data.get("reasoning", ""))
    if spec.mode == "binary":
        verdict = "pass" if str(data.get("verdict", "")).strip().lower() == "pass" else "fail"
        return {"score": None, "verdict": verdict, "reasoning": reasoning}
    try:
        score = int(round(float(data.get("score", 1))))
    except (TypeError, ValueError):
        score = 1
    score = max(1, min(spec.scale_max, score))
    return {
        "score": score,
        "verdict": "pass" if score >= spec.pass_threshold else "fail",
        "reasoning": reasoning,
    }


def _new_run(
    spec: JudgeSpec, judge_set: JudgeSet, probe: str, repeats: int, extra: dict
) -> JudgeRun:
    return JudgeRun(
        id=f"jr_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}",
        set_id=judge_set.id,
        judge_id=spec.id,
        judge_version=spec.version,
        fingerprint=spec.fingerprint,
        spec={
            **spec.model_dump(exclude={"history"}),
            "model": spec.model or default_model(),
            **extra,
        },
        probe=probe,  # type: ignore[arg-type]
        repeats=repeats,
    )


def _majority(verdicts: Sequence[str]) -> str:
    return "pass" if sum(v == "pass" for v in verdicts) * 2 > len(verdicts) else "fail"


def run_judge(
    llm: LLM,
    spec: JudgeSpec,
    judge_set: JudgeSet,
    repeats: int = 1,
    temperature: float | None = None,
    sample_ids: list[str] | None = None,
    job: Job | None = None,
    probe: str = "standard",
) -> JudgeRun:
    samples = [s for s in judge_set.samples if not sample_ids or s.id in set(sample_ids)]
    temp = spec.temperature if temperature is None else temperature
    run = _new_run(spec, judge_set, probe, repeats, {"temperature": temp})
    if job:
        job.progress(0, len(samples) * repeats, f"{spec.name} v{spec.version} on {judge_set.name}")
    done = 0
    for sample in samples:
        scores, verdicts, reasons, error = [], [], [], None
        for r in range(repeats):
            try:
                out = judge_one(llm, spec, sample, temperature=temp, seed=spec.seed + r)
                scores.append(out["score"])
                verdicts.append(out["verdict"])
                reasons.append(out["reasoning"])
            except LLMError as exc:
                error = str(exc)
            done += 1
            if job:
                job.progress(done)
        run.results.append(
            {
                "sample_id": sample.id,
                "kind": sample.kind,
                "scores": scores,
                "verdicts": verdicts,
                "verdict": _majority(verdicts) if verdicts else None,
                "score": statistics.fmean([s for s in scores if s is not None])
                if any(s is not None for s in scores)
                else None,
                "reasoning": reasons[0] if reasons else "",
                "error": error,
            }
        )
    run.summary = consistency_summary(run) if repeats > 1 else _basic_summary(run)
    run.status = "done"
    run.finished_at = now_iso()
    return run


def _basic_summary(run: JudgeRun) -> dict:
    scored = [r for r in run.results if r["verdict"]]
    return {
        "samples": len(run.results),
        "scored": len(scored),
        "errors": sum(1 for r in run.results if r["error"]),
        "pass_rate": sum(r["verdict"] == "pass" for r in scored) / len(scored) if scored else None,
        "mean_score": statistics.fmean([r["score"] for r in scored if r["score"] is not None])
        if any(r["score"] is not None for r in scored)
        else None,
    }


def consistency_summary(run: JudgeRun) -> dict:
    """Same judge, same inputs, several draws: how much does the verdict move by itself?"""
    base = _basic_summary(run)
    unstable = [r for r in run.results if len(set(r["verdicts"])) > 1]
    sds = [
        statistics.stdev(r["scores"])
        for r in run.results
        if len([s for s in r["scores"] if s is not None]) > 1
    ]
    return {
        **base,
        "repeats": run.repeats,
        "unstable_samples": [r["sample_id"] for r in unstable],
        "flip_rate": len(unstable) / len(run.results) if run.results else 0.0,
        "mean_score_sd": statistics.fmean(sds) if sds else 0.0,
    }


# --- validation ----------------------------------------------------------------------------


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Rank correlation (ties averaged). None when either side has no variation."""
    if len(a) < 3:
        return None
    ra, rb = _ranks(a), _ranks(b)
    if len(set(ra)) < 2 or len(set(rb)) < 2:
        return None
    return statistics.correlation(ra, rb)


def _kappa_ci(
    pairs: list[tuple[int, int]], confidence: float = 0.95, resamples: int = 2000
) -> tuple[float | None, float | None]:
    rng = random.Random(20260926)
    values = []
    for _ in range(resamples):
        draw = rng.choices(pairs, k=len(pairs))
        k = cohens_kappa([p[0] for p in draw], [p[1] for p in draw])
        if k is not None:
            values.append(k)
    if len(values) < resamples * 0.5:
        return None, None
    values.sort()
    tail = (1 - confidence) / 2
    return values[int(tail * len(values))], values[
        min(len(values) - 1, int((1 - tail) * len(values)))
    ]


def validation_report(judge_set: JudgeSet, run: JudgeRun, rater: str) -> dict:
    by_id = {s.id: s for s in judge_set.samples}
    rows = []
    for res in run.results:
        sample = by_id.get(res["sample_id"])
        if sample is None or rater not in sample.labels or not res["verdict"]:
            continue
        label = sample.labels[rater]
        rows.append((sample, label, res))
    if not rows:
        return {"n": 0, "note": f"No samples carry a label from {rater!r} and a judge verdict."}

    human = [1 if lab.verdict == "pass" else 0 for _, lab, _ in rows]
    judge = [1 if res["verdict"] == "pass" else 0 for _, _, res in rows]
    n = len(rows)
    tp = sum(1 for h, j in zip(human, judge, strict=True) if h == 1 and j == 1)
    tn = sum(1 for h, j in zip(human, judge, strict=True) if h == 0 and j == 0)
    fp = sum(
        1 for h, j in zip(human, judge, strict=True) if h == 0 and j == 1
    )  # judge passed a human fail
    fn = sum(1 for h, j in zip(human, judge, strict=True) if h == 1 and j == 0)
    kappa = cohens_kappa(human, judge)
    k_low, k_high = _kappa_ci(list(zip(human, judge, strict=True)))
    sens_pass = tp / (tp + fn) if tp + fn else None  # judge says pass when human pass
    spec_fail = tn / (tn + fp) if tn + fp else None  # judge says fail when human fail
    observed = sum(judge) / n
    corrected = None
    if sens_pass is not None and spec_fail is not None and sens_pass + spec_fail - 1 > 0.05:
        corrected = max(0.0, min(1.0, (observed + spec_fail - 1) / (sens_pass + spec_fail - 1)))

    scored = [
        (lab.score, res["score"])
        for _, lab, res in rows
        if lab.score is not None and res["score"] is not None
    ]
    score_block = None
    if scored:
        hs, js = [s[0] for s in scored], [s[1] for s in scored]
        top = int(max(max(hs), max(round(j) for j in js)))
        score_block = {
            "n": len(scored),
            "spearman": spearman(hs, js),
            "mean_bias": statistics.fmean(j - h for h, j in scored),
            "mae": statistics.fmean(abs(j - h) for h, j in scored),
            "labels": [str(i) for i in range(1, top + 1)],
            "confusion": confusion_matrix(
                [int(h) for h in hs],
                [int(round(j)) for j in js],
                categories=tuple(range(1, top + 1)),
            ),
        }

    kinds: dict[str, list] = defaultdict(list)
    for sample, lab, res in rows:
        kinds[sample.kind].append((lab.verdict, res["verdict"]))
    per_kind = [
        {
            "kind": kind,
            "n": len(v),
            "human_pass_rate": sum(h == "pass" for h, _ in v) / len(v),
            "judge_pass_rate": sum(j == "pass" for _, j in v) / len(v),
            "agreement": sum(h == j for h, j in v) / len(v),
        }
        for kind, v in sorted(kinds.items())
    ]
    disagreements = [
        {
            "sample_id": sample.id,
            "kind": sample.kind,
            "question": sample.question,
            "answer": sample.answer,
            "human": lab.verdict,
            "human_score": lab.score,
            "judge": res["verdict"],
            "judge_score": res["score"],
            "reasoning": res["reasoning"],
            "human_note": lab.note,
        }
        for sample, lab, res in rows
        if lab.verdict != res["verdict"]
    ]

    others = sorted({r for s in judge_set.samples for r in s.labels} - {rater})
    human_human = []
    for other in others:
        both = [s for s in judge_set.samples if rater in s.labels and other in s.labels]
        if len(both) >= 5:
            a = [1 if s.labels[rater].verdict == "pass" else 0 for s in both]
            b = [1 if s.labels[other].verdict == "pass" else 0 for s in both]
            human_human.append(
                {
                    "rater": other,
                    "n": len(both),
                    "kappa": cohens_kappa(a, b),
                    "agreement": sum(x == y for x, y in zip(a, b, strict=True)) / len(both),
                }
            )

    return {
        "n": n,
        "rater": rater,
        "judge": {"id": run.judge_id, "version": run.judge_version, "fingerprint": run.fingerprint},
        "accuracy": (tp + tn) / n,
        "kappa": kappa,
        "kappa_ci": [k_low, k_high],
        "kappa_reading": interpret_kappa(kappa),
        "confusion": {"labels": ["pass", "fail"], "matrix": [[tp, fn], [fp, tn]]},
        "fail_recall": spec_fail,
        "pass_recall": sens_pass,
        "fail_precision": tn / (tn + fn) if tn + fn else None,
        "human_pass_rate": sum(human) / n,
        "judge_pass_rate": observed,
        "pass_rate_gap": observed - sum(human) / n,
        "corrected_pass_rate": corrected,
        "scores": score_block,
        "per_kind": per_kind,
        "disagreements": disagreements,
        "human_human": human_human,
    }


# --- probes ------------------------------------------------------------------------------


def verbosity_probe(
    llm: LLM,
    spec: JudgeSpec,
    judge_set: JudgeSet,
    padding: str = "filler",
    limit: int | None = None,
    rater: str | None = None,
    faithfulness_judge=None,
    job: Job | None = None,
) -> JudgeRun:
    """Score each answer twice — as written and with content-free padding — and compare."""
    samples = judge_set.samples[:limit] if limit else list(judge_set.samples)
    pad = PADDINGS[padding]
    run = _new_run(spec, judge_set, "verbosity", 1, {"padding": padding})
    if job:
        job.progress(0, len(samples), f"verbosity ({padding}) · {spec.name} v{spec.version}")
    for n, sample in enumerate(samples, start=1):
        row: dict = {"sample_id": sample.id, "kind": sample.kind, "error": None}
        try:
            original = judge_one(llm, spec, sample)
            padded = judge_one(llm, spec, sample, answer=sample.answer + pad)
            row.update(
                original_score=original["score"],
                padded_score=padded["score"],
                original_verdict=original["verdict"],
                padded_verdict=padded["verdict"],
                human=sample.labels[rater].verdict if rater and rater in sample.labels else None,
            )
            if faithfulness_judge is not None and sample.contexts:
                from agent_hub.eval.lab.genmetrics import faithfulness

                row["faith_original"] = faithfulness(
                    faithfulness_judge, sample.answer, sample.contexts
                ).score
                row["faith_padded"] = faithfulness(
                    faithfulness_judge, sample.answer + pad, sample.contexts
                ).score
        except LLMError as exc:
            row["error"] = str(exc)
        run.results.append(row)
        if job:
            job.progress(n)

    ok = [r for r in run.results if not r["error"]]
    summary: dict = {"samples": len(run.results), "padding": padding, "padding_text": pad.strip()}
    if spec.mode == "pointwise" and ok:
        base = [float(r["original_score"]) for r in ok]
        padded = [float(r["padded_score"]) for r in ok]
        delta = paired_delta_ci(base, padded)
        summary.update(
            mean_delta=delta.mean,
            delta_low=delta.low,
            delta_high=delta.high,
            up=sum(p > b for b, p in zip(base, padded, strict=True)),
            down=sum(p < b for b, p in zip(base, padded, strict=True)),
            same=sum(p == b for b, p in zip(base, padded, strict=True)),
        )
    summary["fail_to_pass"] = [
        r["sample_id"]
        for r in ok
        if r["original_verdict"] == "fail" and r["padded_verdict"] == "pass"
    ]
    summary["pass_to_fail"] = [
        r["sample_id"]
        for r in ok
        if r["original_verdict"] == "pass" and r["padded_verdict"] == "fail"
    ]
    faith = [
        r for r in ok if r.get("faith_original") is not None and r.get("faith_padded") is not None
    ]
    if faith:
        summary["faithfulness_delta"] = statistics.fmean(
            r["faith_padded"] - r["faith_original"] for r in faith
        )
    run.summary = summary
    run.status = "done"
    run.finished_at = now_iso()
    return run


def _pairs(judge_set: JudgeSet, rater: str | None, identical: int) -> list[dict]:
    by_question: dict[str, list[JudgeSample]] = defaultdict(list)
    for s in judge_set.samples:
        by_question[s.question.strip()].append(s)
    pairs: list[dict] = []
    for group in by_question.values():
        for i, x in enumerate(group):
            for y in group[i + 1 :]:
                if rater and rater in x.labels and rater in y.labels:
                    vx, vy = x.labels[rater].verdict, y.labels[rater].verdict
                    if vx == vy:
                        continue
                    better = x.id if vx == "pass" else y.id
                else:
                    better = None
                pairs.append({"x": x.id, "y": y.id, "preferred": better, "identical": False})
    rng = random.Random(7)
    same = rng.sample(judge_set.samples, min(identical, len(judge_set.samples)))
    pairs += [{"x": s.id, "y": s.id, "preferred": None, "identical": True} for s in same]
    return pairs


def position_probe(
    llm: LLM,
    spec: JudgeSpec,
    judge_set: JudgeSet,
    rater: str | None = None,
    identical: int = 6,
    limit: int | None = None,
    job: Job | None = None,
) -> JudgeRun:
    """Ask every pairwise question twice, with the answers swapped. A fair judge's pick moves with the answer."""
    by_id = {s.id: s for s in judge_set.samples}
    pairs = _pairs(judge_set, rater, identical)
    if limit:
        pairs = [p for p in pairs if not p["identical"]][:limit] + [
            p for p in pairs if p["identical"]
        ]
    run = _new_run(spec, judge_set, "position", 1, {"rater": rater})
    if job:
        job.progress(0, len(pairs), f"position · {spec.name} v{spec.version}")
    for n, pair in enumerate(pairs, start=1):
        x, y = by_id[pair["x"]], by_id[pair["y"]]
        row = {**pair, "error": None, "positions": [], "chosen": []}
        try:
            # Order 1 shows x as A; order 2 shows y as A. Map each letter back to x / y.
            for first, second, letter_to in (
                (x, y, {"A": "x", "B": "y"}),
                (y, x, {"A": "y", "B": "x"}),
            ):
                system, user = _pairwise_prompt(spec, x, first.answer, second.answer)
                data = llm.chat_json(
                    system,
                    user,
                    model=spec.model or default_model(),
                    temperature=spec.temperature,
                    seed=spec.seed,
                )
                w = str(data.get("winner", "tie")).strip().upper()
                w = w if w in ("A", "B") else "tie"
                row["positions"].append(w)
                row["chosen"].append(letter_to.get(w, "tie"))
        except LLMError as exc:
            row["error"] = str(exc)
        run.results.append(row)
        if job:
            job.progress(n)

    ok = [r for r in run.results if not r["error"]]
    contrast = [r for r in ok if not r["identical"]]
    ident = [r for r in ok if r["identical"]]
    positions = [p for r in ok for p in r["positions"] if p in ("A", "B")]
    consistent = [r for r in contrast if r["chosen"][0] == r["chosen"][1]]
    with_pref = [r for r in contrast if r["preferred"]]
    correct = sum(
        sum(1 for c in r["chosen"] if c == ("x" if r["preferred"] == r["x"] else "y"))
        for r in with_pref
    )
    run.summary = {
        "pairs": len(contrast),
        "identical_pairs": len(ident),
        "consistency": len(consistent) / len(contrast) if contrast else None,
        "first_position_rate": positions.count("A") / len(positions) if positions else None,
        "identical_non_tie_rate": sum(1 for r in ident for p in r["positions"] if p != "tie")
        / (2 * len(ident))
        if ident
        else None,
        "accuracy_vs_human": correct / (2 * len(with_pref)) if with_pref else None,
        "decisions": len(positions),
    }
    run.status = "done"
    run.finished_at = now_iso()
    return run


def drift_report(judge_set: JudgeSet, a: JudgeRun, b: JudgeRun, rater: str | None = None) -> dict:
    """Same samples, two judge configurations: how much did the verdicts move, and toward whom?"""
    ra = {r["sample_id"]: r for r in a.results if r.get("verdict")}
    rb = {r["sample_id"]: r for r in b.results if r.get("verdict")}
    ids = sorted(set(ra) & set(rb))
    changed = sorted(
        k
        for k in set(a.spec) | set(b.spec)
        if a.spec.get(k) != b.spec.get(k) and k not in ("id", "created_at")
    )
    va = [1 if ra[i]["verdict"] == "pass" else 0 for i in ids]
    vb = [1 if rb[i]["verdict"] == "pass" else 0 for i in ids]
    scores = [
        (ra[i]["score"], rb[i]["score"])
        for i in ids
        if ra[i]["score"] is not None and rb[i]["score"] is not None
    ]
    out: dict = {
        "n": len(ids),
        "same_fingerprint": a.fingerprint == b.fingerprint,
        "changed_fields": changed,
        "verdict_agreement": sum(x == y for x, y in zip(va, vb, strict=True)) / len(ids)
        if ids
        else None,
        "kappa_between_runs": cohens_kappa(va, vb) if ids else None,
        "pass_rate_a": sum(va) / len(ids) if ids else None,
        "pass_rate_b": sum(vb) / len(ids) if ids else None,
        "flips": [
            {
                "sample_id": i,
                "kind": ra[i]["kind"],
                "from": ra[i]["verdict"],
                "to": rb[i]["verdict"],
                "score_a": ra[i]["score"],
                "score_b": rb[i]["score"],
            }
            for i in ids
            if ra[i]["verdict"] != rb[i]["verdict"]
        ],
    }
    if scores:
        delta = paired_delta_ci([float(s[0]) for s in scores], [float(s[1]) for s in scores])
        out["score_shift"] = {
            "mean": delta.mean,
            "low": delta.low,
            "high": delta.high,
            "n": delta.n,
        }
        out["mean_score_a"] = mean_ci([float(s[0]) for s in scores]).mean
        out["mean_score_b"] = mean_ci([float(s[1]) for s in scores]).mean
    if rater:
        va_ = validation_report(judge_set, a, rater)
        vb_ = validation_report(judge_set, b, rater)
        out["validation"] = {
            "a": {
                k: va_.get(k) for k in ("n", "accuracy", "kappa", "fail_recall", "pass_rate_gap")
            },
            "b": {
                k: vb_.get(k) for k in ("n", "accuracy", "kappa", "fail_recall", "pass_rate_gap")
            },
        }
    return out


def blind_task(judge_set: JudgeSet, rater: str) -> dict:
    """Unlabelled samples for `rater`, shuffled, with the designer's `kind` tag and all labels hidden."""
    todo = [s for s in judge_set.samples if rater not in s.labels]
    random.Random(f"{judge_set.id}:{rater}").shuffle(todo)
    return {
        "rater": rater,
        "total": len(judge_set.samples),
        "done": len(judge_set.samples) - len(todo),
        "remaining": len(todo),
        "next": [
            {
                "id": s.id,
                "question": s.question,
                "answer": s.answer,
                "reference": s.reference,
                "contexts": s.contexts,
            }
            for s in todo[:1]
        ],
    }
