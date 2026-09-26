"""Release gate — a written policy that decides whether a candidate may ship.

The offline harness's `gate.py` asks one question: "did a mean drop by more
than 0.05?". This gate makes the policy explicit data (`GatePolicy`) and asks
three kinds of question, in order:

1. **Can the two runs be compared at all?** (preconditions → BLOCK if not)
   Same dataset content hash, a frozen dataset, the same judge fingerprint for
   judged metrics, enough items, few errors. A gate that compares a run scored
   by judge v2 against a baseline scored by judge v1 is measuring the judge.

2. **Does each rule hold?** Four rule kinds, each with a different honesty:

       floor         candidate mean ≥ threshold          absolute quality bar
       ceiling       candidate mean ≤ threshold          for "bad" rates (hallucination)
       max_drop      candidate − baseline ≥ −threshold   point estimates — cheap, noisy
       non_inferior  lower CI bound of the paired Δ ≥ −threshold   noise-aware

   `max_drop` is what most teams wire first and what cries wolf: on 26
   questions a mean moves several points by chance. `non_inferior` asks the
   question you actually care about — "can I rule out a loss bigger than the
   margin?" — and on a small set it often answers "no, not with this data",
   which is also worth knowing before a release.

3. **Can this rule even resolve its threshold at this n?** (advisories)
   For every comparative rule the gate computes the smallest effect this many
   paired questions can detect. When the threshold is below it, a `max_drop`
   rule will flap on noise and a `non_inferior` rule can never pass — the
   policy, not the model, needs fixing.

Decision: BLOCK if any precondition or `block` rule fails, WARN if only `warn`
rules fail, PASS otherwise. Advisories are shown next to their rule but never
change the decision — they are a note about the policy, not about the
candidate. An override needs a written reason and is recorded; it never
rewrites the verdict it overrode.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from agent_hub.eval.lab.genmodels import GatePolicy, GateRule, GenRun
from agent_hub.eval.lab.models import Run, now_iso
from agent_hub.eval.stats import detectable_effect, mean_ci, paired_delta_ci

__all__ = [
    "LOWER_IS_BETTER",
    "RunView",
    "view_of",
    "evaluate_gate",
    "default_policy",
    "render_markdown",
]

#: Metrics where a DECREASE is an improvement. Everything else: higher is better.
LOWER_IS_BETTER = {"false_abstention", "answered_unanswerable"}


@dataclass
class RunView:
    """What the gate needs from a run, whichever half of the lab produced it."""

    id: str
    kind: str  # retrieval | generation
    name: str
    dataset_id: str
    dataset_version: int
    dataset_hash: str
    dataset_status: str
    split: str
    judge_fingerprint: str | None
    status: str
    n: int
    errors: int
    scores: dict[str, dict[str, float]]  # item_id -> metric -> value
    created_at: str


def view_of(run: Run | GenRun | dict) -> RunView:
    if isinstance(run, dict):
        run = (
            GenRun.model_validate(run)
            if run.get("kind") == "generation"
            else Run.model_validate(run)
        )
    if isinstance(run, GenRun):
        ok = [r for r in run.results if r.error is None]
        return RunView(
            id=run.id,
            kind="generation",
            name=run.spec.name,
            dataset_id=run.dataset_id,
            dataset_version=run.dataset_version,
            dataset_hash=run.dataset_hash,
            dataset_status=run.dataset_status,
            split=run.split,
            judge_fingerprint=run.judge_fingerprint,
            status=run.status,
            n=len(run.results),
            errors=len(run.results) - len(ok),
            scores={r.item_id: dict(r.metrics) for r in ok},
            created_at=run.created_at,
        )
    ok = [r for r in run.results if r.error is None and r.metrics]
    return RunView(
        id=run.id,
        kind="retrieval",
        name=run.variant.name,
        dataset_id=run.dataset_id,
        dataset_version=run.dataset_version,
        dataset_hash=run.dataset_hash,
        dataset_status=run.dataset_status,
        split=run.split,
        judge_fingerprint=None,
        status=run.status,
        n=len(run.results),
        errors=sum(1 for r in run.results if r.error is not None),
        scores={r.item_id: dict(r.metrics) for r in ok},
        created_at=run.created_at,
    )


def _values(view: RunView, metric: str) -> dict[str, float]:
    return {i: m[metric] for i, m in view.scores.items() if metric in m}


def _check(ok: bool, title: str, detail: str) -> dict:
    return {"ok": ok, "title": title, "detail": detail}


def _preconditions(policy: GatePolicy, base: RunView, cand: RunView) -> list[dict]:
    checks = [
        _check(
            base.kind == cand.kind,
            "Same kind of run",
            f"baseline is {base.kind}, candidate is {cand.kind}.",
        ),
        _check(
            base.status == "done" and cand.status == "done",
            "Both runs finished",
            f"baseline {base.status}, candidate {cand.status}.",
        ),
        _check(
            base.split == cand.split,
            "Same split",
            f"baseline {base.split}, candidate {cand.split}.",
        ),
    ]
    if policy.require_same_dataset:
        checks.append(
            _check(
                base.dataset_hash == cand.dataset_hash,
                "Same dataset content",
                f"hash {base.dataset_hash[:10]} vs {cand.dataset_hash[:10]} — different questions, labels or "
                "documents make the numbers incomparable.",
            )
        )
    if policy.require_frozen_dataset:
        checks.append(
            _check(
                cand.dataset_status == "frozen",
                "Frozen dataset",
                f"candidate scored on a {cand.dataset_status} version — a draft can change under the number.",
            )
        )
    if policy.require_same_judge and cand.kind == "generation":
        checks.append(
            _check(
                base.judge_fingerprint == cand.judge_fingerprint,
                "Same judge",
                f"judge {base.judge_fingerprint} vs {cand.judge_fingerprint}. A different judge model, "
                "temperature or prompt measures differently: the delta would be judge drift, not a change "
                "in the system.",
            )
        )
    shared = len(set(base.scores) & set(cand.scores))
    checks.append(
        _check(
            shared >= policy.min_items,
            f"At least {policy.min_items} paired items",
            f"{shared} items scored by both runs.",
        )
    )
    rate = cand.errors / cand.n if cand.n else 1.0
    checks.append(
        _check(
            rate <= policy.max_error_rate,
            f"Error rate ≤ {policy.max_error_rate:.0%}",
            f"{cand.errors} of {cand.n} candidate items errored ({rate:.0%}). Errors are excluded from "
            "scores, so many of them silently shrink the set.",
        )
    )
    return checks


def _rule(rule: GateRule, base: RunView, cand: RunView, confidence: float) -> dict:
    lower_better = rule.metric in LOWER_IS_BETTER
    b, c = _values(base, rule.metric), _values(cand, rule.metric)
    out: dict = {**rule.model_dump(), "lower_is_better": lower_better}
    if not c:
        return {
            **out,
            "status": "fail",
            "detail": f"{rule.metric} is missing from the candidate run (no item in it measures this metric).",
        }
    cvals = list(c.values())
    ci = mean_ci(cvals, confidence)
    out["candidate"] = {
        "mean": ci.mean,
        "low": ci.low if ci.n > 1 else ci.mean,
        "high": ci.high if ci.n > 1 else ci.mean,
        "n": ci.n,
    }
    if b:
        out["baseline"] = {"mean": statistics.fmean(b.values()), "n": len(b)}
    ids = sorted(set(b) & set(c))
    delta = None
    if ids:
        base_v = [b[i] for i in ids]
        cand_v = [c[i] for i in ids]
        d = paired_delta_ci(base_v, cand_v, confidence)
        diffs = [y - x for x, y in zip(base_v, cand_v, strict=True)]
        sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        # Orient so that "improvement" is always positive.
        sign = -1.0 if lower_better else 1.0
        low, high = (d.low, d.high) if not lower_better else (-d.high, -d.low)
        delta = {
            "mean": sign * d.mean,
            "low": low if d.n > 1 else sign * d.mean,
            "high": high if d.n > 1 else sign * d.mean,
            "n": d.n,
            "sd": sd,
            "wins": sum(1 for x in diffs if sign * x > 0),
            "losses": sum(1 for x in diffs if sign * x < 0),
            "detectable": detectable_effect(len(ids), sd, confidence) if sd > 0 else 0.0,
        }
        out["delta"] = delta

    t = rule.threshold
    if rule.kind == "floor":
        ok = ci.mean >= t
        out["detail"] = f"candidate {ci.mean:.3f} {'≥' if ok else '<'} floor {t:.3f}"
        if ok and ci.n > 1 and ci.low < t:
            out["advisory"] = (
                f"Passes on the point estimate, but the CI [{ci.low:.3f}, {ci.high:.3f}] reaches below the floor."
            )
    elif rule.kind == "ceiling":
        ok = ci.mean <= t
        out["detail"] = f"candidate {ci.mean:.3f} {'≤' if ok else '>'} ceiling {t:.3f}"
        if ok and ci.n > 1 and ci.high > t:
            out["advisory"] = (
                f"Passes on the point estimate, but the CI [{ci.low:.3f}, {ci.high:.3f}] reaches above the ceiling."
            )
    elif delta is None:
        ok = False
        out["detail"] = "no paired items with the baseline"
    elif rule.kind == "max_drop":
        ok = delta["mean"] >= -t
        out["detail"] = f"Δ {delta['mean']:+.3f} {'≥' if ok else '<'} −{t:.3f} (point estimate)"
        if delta["detectable"] > t:
            out["advisory"] = (
                f"n={delta['n']} can only resolve changes of about ±{delta['detectable']:.3f}; a tolerance of "
                f"{t:.3f} is inside the noise, so this rule will flap on chance."
            )
    else:  # non_inferior
        ok = delta["low"] >= -t
        out["detail"] = f"CI lower bound {delta['low']:+.3f} {'≥' if ok else '<'} −{t:.3f} (margin)"
        if not ok and delta["high"] >= -t and delta["mean"] >= 0:
            out["advisory"] = (
                "Not a measured regression — the interval is just too wide to rule one out. More paired "
                f"items (or a wider margin than {t:.3f}) are needed for this rule to ever pass."
            )
        elif delta["detectable"] > t:
            out["advisory"] = (
                f"n={delta['n']} resolves about ±{delta['detectable']:.3f}; the margin {t:.3f} is tighter than that."
            )
    out["status"] = "pass" if ok else "fail"
    return out


def evaluate_gate(
    policy: GatePolicy, baseline: Run | GenRun | dict, candidate: Run | GenRun | dict
) -> dict:
    base, cand = view_of(baseline), view_of(candidate)
    pre = _preconditions(policy, base, cand)
    rules = [
        _rule(r, base, cand, policy.confidence) for r in policy.rules if r.applies_to == cand.kind
    ]
    blocked_pre = [p for p in pre if not p["ok"]]
    blocking = [r for r in rules if r["status"] == "fail" and r["severity"] == "block"]
    warning = [r for r in rules if r["status"] == "fail" and r["severity"] == "warn"]
    if blocked_pre or blocking:
        decision = "block"
    elif warning:
        decision = "warn"
    else:
        decision = "pass"
    reasons = [f"precondition: {p['title']} — {p['detail']}" for p in blocked_pre]
    reasons += [f"{r['id']}: {r['metric']} {r['detail']}" for r in blocking + warning]
    return {
        "decision": decision,
        "policy": {"id": policy.id, "version": policy.version, "name": policy.name},
        "baseline": _meta(base),
        "candidate": _meta(cand),
        "preconditions": pre,
        "rules": rules,
        "reasons": reasons,
        "evaluated_at": now_iso(),
        "exit_code": 1 if decision == "block" else 0,
    }


def _meta(v: RunView) -> dict:
    return {
        "id": v.id,
        "kind": v.kind,
        "name": v.name,
        "dataset": f"{v.dataset_id} v{v.dataset_version}",
        "dataset_hash": v.dataset_hash,
        "split": v.split,
        "judge_fingerprint": v.judge_fingerprint,
        "n": v.n,
        "errors": v.errors,
        "created_at": v.created_at,
    }


def default_policy() -> GatePolicy:
    """A starting policy — the guide asks you to criticise and change it."""
    return GatePolicy.model_validate(
        {
            "id": "kestrel-release",
            "name": "Kestrel assistant — release policy",
            "description": "What must hold before a RAG change ships. Blocks on safety and on measured, "
            "noise-aware regressions; warns on the rest.",
            "min_items": 20,
            "rules": [
                {
                    "id": "faithfulness-floor",
                    "metric": "faithfulness",
                    "kind": "floor",
                    "threshold": 0.8,
                    "severity": "block",
                    "applies_to": "generation",
                    "rationale": "Unsupported claims are the failure users cannot detect themselves.",
                },
                {
                    "id": "answer-ok-noninferior",
                    "metric": "answer_ok",
                    "kind": "non_inferior",
                    "threshold": 0.2,
                    "severity": "block",
                    "applies_to": "generation",
                    "rationale": "Correct-answer rate must not be measurably worse than the baseline by more "
                    "than 20 points. 20, not 10: ~27 dev questions resolve only about ±0.15–0.25, and a "
                    "tighter margin blocks unchanged reruns. Grow the dataset to tighten it.",
                },
                {
                    "id": "unanswerable-ceiling",
                    "metric": "answered_unanswerable",
                    "kind": "ceiling",
                    "threshold": 0.0,
                    "severity": "block",
                    "applies_to": "generation",
                    "rationale": "Answering a question no document covers is a fabrication.",
                },
                {
                    "id": "false-abstention-drop",
                    "metric": "false_abstention",
                    "kind": "max_drop",
                    "threshold": 0.05,
                    "severity": "warn",
                    "applies_to": "generation",
                    "rationale": "More refusals on answerable questions hurt usefulness, not safety.",
                },
                {
                    "id": "relevance-drop",
                    "metric": "answer_relevance",
                    "kind": "max_drop",
                    "threshold": 0.05,
                    "severity": "warn",
                    "applies_to": "generation",
                    "rationale": "Style regression — worth a look.",
                },
                {
                    "id": "recall-noninferior",
                    "metric": "recall@k",
                    "kind": "non_inferior",
                    "threshold": 0.05,
                    "severity": "block",
                    "applies_to": "retrieval",
                    "rationale": "The generator cannot cite what retrieval never returned.",
                },
                {
                    "id": "mrr-drop",
                    "metric": "mrr",
                    "kind": "max_drop",
                    "threshold": 0.05,
                    "severity": "warn",
                    "applies_to": "retrieval",
                    "rationale": "Ranking quality.",
                },
            ],
        }
    )


def render_markdown(result: dict) -> str:
    """Summary for a CI log or $GITHUB_STEP_SUMMARY."""
    icon = {"pass": "✅ PASS", "warn": "⚠️ PASS WITH WARNINGS", "block": "⛔ BLOCK"}[
        result["decision"]
    ]
    lines = [
        f"## Eval release gate — {icon}",
        "",
        f"Policy **{result['policy']['name']}** (v{result['policy']['version']}) · "
        f"baseline `{result['baseline']['id']}` ({result['baseline']['name']}) → "
        f"candidate `{result['candidate']['id']}` ({result['candidate']['name']})",
        "",
        "| Precondition | OK | Detail |",
        "|---|---|---|",
    ]
    lines += [
        f"| {p['title']} | {'✅' if p['ok'] else '⛔'} | {p['detail']} |"
        for p in result["preconditions"]
    ]
    lines += [
        "",
        "| Rule | Metric | Kind | Severity | Result | Detail |",
        "|---|---|---|---|---|---|",
    ]
    for r in result["rules"]:
        mark = "✅" if r["status"] == "pass" else ("⛔" if r["severity"] == "block" else "⚠️")
        detail = r.get("detail", "") + (f" — _{r['advisory']}_" if r.get("advisory") else "")
        lines.append(
            f"| {r['id']} | {r['metric']} | {r['kind']} | {r['severity']} | {mark} | {detail} |"
        )
    return "\n".join(lines) + "\n"
