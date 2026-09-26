"""HTTP routes for the generation half: metrics playground, pipeline runs, components, Judge Lab.

Mounted by `api.create_app`. Long work (pipeline runs, judge runs, probes) is a
background job, like indexing; the playground and reports are synchronous.
The holdout guard is the same as for retrieval runs: ad-hoc generation runs
may not read the holdout split.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError

from agent_hub.eval.lab import genmetrics as gm
from agent_hub.eval.lab import judgelab as jl
from agent_hub.eval.lab.genmodels import (
    FAULTS,
    STAGES,
    HumanLabel,
    JudgeRun,
    JudgeSample,
    JudgeSet,
    JudgeSpec,
    PipelineSpec,
)
from agent_hub.eval.lab.jobs import JobRegistry
from agent_hub.eval.lab.llm import LLM, default_model
from agent_hub.eval.lab.models import SLUG, now_iso
from agent_hub.eval.lab.pipeline import GENERATOR_PROMPTS, fault_matrix, run_generation
from agent_hub.eval.lab.scenarios import SCENARIOS
from agent_hub.eval.lab.store import Conflict, LabStore, NotFound
from agent_hub.eval.stats import paired_delta_ci

__all__ = ["build_router"]


class PlaygroundBody(BaseModel):
    question: str = Field(min_length=1)
    contexts: list[str] = Field(default_factory=list, max_length=20)
    answer: str = ""
    reference: str = ""
    metrics: list[str] | None = None
    model: str | None = None
    embedding_relevance: bool = True


class GenRunBody(BaseModel):
    dataset_id: str
    version: int
    spec: PipelineSpec = Field(default_factory=PipelineSpec)
    split: Literal["dev", "holdout", "all"] = "dev"
    item_ids: list[str] | None = None
    limit: int | None = Field(default=None, ge=1, le=500)
    llm_context_metrics: bool = True
    judge_model: str | None = None


class CompareGenBody(BaseModel):
    run_a: str
    run_b: str
    confidence: float = Field(default=0.95, ge=0.5, le=0.999)


class FaultSuiteBody(BaseModel):
    dataset_id: str
    version: int
    spec: PipelineSpec = Field(default_factory=PipelineSpec)
    faults: list[str] = Field(default_factory=lambda: list(FAULTS))
    item_ids: list[str] | None = None
    limit: int | None = Field(default=12, ge=1, le=500)
    judge_model: str | None = None


class FaultMatrixBody(BaseModel):
    run_ids: list[str] = Field(min_length=1)


class JudgeBody(BaseModel):
    id: str | None = None
    name: str = Field(min_length=1)
    model: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=1.5)
    seed: int = 7
    mode: Literal["pointwise", "binary"] = "pointwise"
    rubric: str = Field(min_length=10)
    reference_guided: bool = True
    show_context: bool = False
    reasoning_first: bool = True
    scale_max: int = Field(default=5, ge=2, le=10)
    pass_threshold: int = Field(default=4, ge=1, le=10)
    notes: str = ""


class TryBody(BaseModel):
    question: str
    answer: str
    reference: str = ""
    contexts: list[str] = Field(default_factory=list)


class JudgeSetBody(BaseModel):
    name: str = Field(min_length=1)
    id: str | None = None
    description: str = ""
    jsonl: str = ""


class ImportSamplesBody(BaseModel):
    jsonl: str
    mode: Literal["append", "replace"] = "append"


class FromGenRunBody(BaseModel):
    run_id: str
    name: str = Field(min_length=1)
    id: str | None = None
    limit: int = Field(default=20, ge=1, le=200)
    seed: int = 7


class JudgeRunBody(BaseModel):
    judge_id: str
    set_id: str
    repeats: int = Field(default=1, ge=1, le=10)
    temperature: float | None = Field(default=None, ge=0.0, le=1.5)
    probe: Literal["standard", "consistency"] = "standard"


class VerbosityBody(BaseModel):
    judge_id: str
    set_id: str
    padding: Literal["filler", "restate", "authority"] = "filler"
    limit: int | None = Field(default=None, ge=1, le=200)
    rater: str | None = None
    with_faithfulness: bool = False


class PositionBody(BaseModel):
    judge_id: str
    set_id: str
    rater: str | None = None
    identical: int = Field(default=6, ge=0, le=40)
    limit: int | None = Field(default=None, ge=1, le=200)


def _slug(text: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip().lower()).strip("-.")[:100] or "item"


def _parse_samples(jsonl: str) -> list[JudgeSample]:
    out, errors = [], []
    for n, line in enumerate(jsonl.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            out.append(JudgeSample.model_validate(json.loads(line)))
        except (ValueError, ValidationError) as exc:
            errors.append(f"line {n}: {exc}")
    if errors:
        raise HTTPException(422, {"code": "invalid_jsonl", "errors": errors[:20]})
    ids = [s.id for s in out]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise HTTPException(
            422, {"code": "invalid_jsonl", "errors": [f"duplicate ids: {', '.join(dupes)}"]}
        )
    return out


def build_router(store: LabStore, kh, llm: LLM, jobs: JobRegistry) -> APIRouter:
    router = APIRouter(prefix="/api/eval")

    def metric_judge(model: str | None = None) -> gm.MetricJudge:
        return gm.MetricJudge(llm, model=model or default_model())

    # --- config & playground ---------------------------------------------------------

    @router.get("/gen/config")
    def gen_config() -> dict:
        judge = metric_judge()
        return {
            "stages": list(STAGES),
            "faults": FAULTS,
            "generator_prompts": GENERATOR_PROMPTS,
            "rerank_modes": ["none", "lexical", "llm"],
            "metrics": list(gm.GEN_METRICS),
            "failure_modes": gm.FAILURE_MODES,
            "metric_prompts": gm.PROMPTS,
            "prompt_version": gm.METRIC_PROMPT_VERSION,
            "judge": {**judge.config, "fingerprint": judge.fingerprint},
            "default_model": default_model(),
            "models": llm.models(),
            "paddings": jl.PADDINGS,
        }

    @router.get("/gen/scenarios")
    def scenarios() -> list[dict]:
        return SCENARIOS

    @router.post("/gen/metrics/playground")
    def playground(body: PlaygroundBody) -> dict:
        judge = metric_judge(body.model)
        wanted = set(
            body.metrics
            or [
                "faithfulness",
                "answer_relevance",
                "context_precision",
                "context_recall",
                "answer_correctness",
            ]
        )
        results: dict[str, dict] = {}
        if "faithfulness" in wanted:
            results["faithfulness"] = gm.faithfulness(judge, body.answer, body.contexts).as_dict()
        if "answer_relevance" in wanted:
            results["answer_relevance"] = gm.answer_relevance(
                judge, body.question, body.answer
            ).as_dict()
            if body.embedding_relevance:
                results["answer_relevance_embedding"] = gm.answer_relevance_embedding(
                    judge, body.question, body.answer
                ).as_dict()
        if "context_precision" in wanted:
            results["context_precision"] = gm.context_precision(
                judge, body.question, body.contexts, body.reference
            ).as_dict()
        if "context_recall" in wanted:
            results["context_recall"] = gm.context_recall(
                judge, body.contexts, body.reference
            ).as_dict()
        if "answer_correctness" in wanted:
            results["answer_correctness"] = gm.answer_correctness(
                judge, body.question, body.answer, body.reference
            ).as_dict()
            results["answer_verdict"] = gm.answer_verdict(
                judge, body.question, body.answer, body.reference
            ).as_dict()
        return {
            "results": results,
            "abstention": gm.is_abstention(body.answer),
            "judge": {**judge.config, "fingerprint": judge.fingerprint},
            "calls": judge.calls,
        }

    # --- pipeline runs ---------------------------------------------------------------

    def _guard_holdout(bundle, split: str) -> None:
        has_holdout = any(i.split == "holdout" for i in bundle.items)
        if split == "holdout" or (split == "all" and has_holdout):
            raise Conflict(
                "Ad-hoc generation runs cannot read holdout items — the holdout is for one "
                "pre-registered look. Use the dev split."
            )

    @router.post("/gen/runs")
    def start_gen_run(body: GenRunBody) -> dict:
        bundle = store.bundle(body.dataset_id, body.version)
        _guard_holdout(bundle, body.split)
        if "off" not in bundle.indexes:
            raise Conflict("index the corpus (Golden Sets → Index, 'off') before a generation run")
        judge = metric_judge(body.judge_model)
        job = jobs.submit(
            "gen_run",
            f"generation · {body.spec.name} · {body.split}",
            lambda job: {
                "run_id": run_generation(
                    kh,
                    llm,
                    store,
                    bundle,
                    body.spec,
                    body.split,
                    judge,
                    item_ids=body.item_ids,
                    limit=body.limit,
                    llm_context_metrics=body.llm_context_metrics,
                    job=job,
                ).id
            },
        )
        return job.as_dict()

    @router.get("/gen/runs")
    def list_gen_runs(dataset_id: str | None = None) -> list[dict]:
        return store.list_gen_runs(dataset_id)

    @router.get("/gen/runs/{run_id}")
    def get_gen_run(run_id: str) -> dict:
        return store.get_gen_run(run_id).model_dump()

    @router.delete("/gen/runs/{run_id}")
    def delete_gen_run(run_id: str) -> dict:
        store.delete_doc_json("gen_runs", run_id)
        return {"deleted": run_id}

    @router.post("/gen/compare")
    def compare(body: CompareGenBody) -> dict:
        a, b = store.get_gen_run(body.run_a), store.get_gen_run(body.run_b)
        ra = {r.item_id: r for r in a.results if r.error is None}
        rb = {r.item_id: r for r in b.results if r.error is None}
        ids = sorted(set(ra) & set(rb))
        metrics = sorted(
            {m for r in ra.values() for m in r.metrics}
            & {m for r in rb.values() for m in r.metrics}
        )
        rows = []
        for m in metrics:
            pair = [
                (ra[i].metrics[m], rb[i].metrics[m])
                for i in ids
                if m in ra[i].metrics and m in rb[i].metrics
            ]
            if not pair:
                continue
            d = paired_delta_ci([p[0] for p in pair], [p[1] for p in pair], body.confidence)
            diffs = [y - x for x, y in pair]
            rows.append(
                {
                    "metric": m,
                    "n": d.n,
                    "a": statistics.fmean(p[0] for p in pair),
                    "b": statistics.fmean(p[1] for p in pair),
                    "delta": d.mean,
                    "low": d.low if d.n > 1 else None,
                    "high": d.high if d.n > 1 else None,
                    "wins": sum(1 for x in diffs if x > 0),
                    "losses": sum(1 for x in diffs if x < 0),
                    "ties": sum(1 for x in diffs if x == 0),
                }
            )
        transitions = Counter((ra[i].outcome, rb[i].outcome) for i in ids)
        warnings = []
        if a.judge_fingerprint != b.judge_fingerprint:
            warnings.append(
                "The two runs were scored by different judges (fingerprints differ): a judged-metric "
                "delta mixes the system change with judge drift."
            )
        if a.dataset_hash != b.dataset_hash:
            warnings.append("The runs used different dataset content.")
        changed = [
            k
            for k in PipelineSpec.model_fields
            if k != "name" and getattr(a.spec, k) != getattr(b.spec, k)
        ]
        if len(changed) > 1:
            warnings.append(
                f"The specs differ in {len(changed)} fields ({', '.join(changed)}): an effect cannot be "
                "attributed to one of them."
            )
        return {
            "n": len(ids),
            "metrics": rows,
            "spec_diff": changed,
            "transitions": [
                {"from": f, "to": t, "n": n} for (f, t), n in sorted(transitions.items())
            ],
            "blame_a": a.components.get("blame", {}),
            "blame_b": b.components.get("blame", {}),
            "items": [
                {
                    "item_id": i,
                    "a": ra[i].outcome,
                    "b": rb[i].outcome,
                    "blame_a": ra[i].blame,
                    "blame_b": rb[i].blame,
                }
                for i in ids
                if ra[i].outcome != rb[i].outcome
            ],
            "warnings": warnings,
        }

    @router.post("/gen/fault-suite")
    def fault_suite(body: FaultSuiteBody) -> dict:
        bundle = store.bundle(body.dataset_id, body.version)
        unknown = [f for f in body.faults if f not in FAULTS]
        if unknown:
            raise HTTPException(422, f"unknown faults: {', '.join(unknown)}")
        judge = metric_judge(body.judge_model)

        def work(job) -> dict:
            produced = []
            for fault in body.faults:
                spec = body.spec.model_copy(
                    update={"fault": fault, "name": f"{body.spec.name}+{fault}"}
                )
                run = run_generation(
                    kh,
                    llm,
                    store,
                    bundle,
                    spec,
                    "dev",
                    judge,
                    item_ids=body.item_ids,
                    limit=body.limit,
                    llm_context_metrics=False,
                    job=job,
                )
                produced.append(run.id)
            return {"run_ids": produced}

        return jobs.submit("fault_suite", f"fault suite · {len(body.faults)} runs", work).as_dict()

    @router.post("/gen/fault-matrix")
    def matrix(body: FaultMatrixBody) -> dict:
        return fault_matrix([store.get_gen_run(r) for r in body.run_ids])

    # --- judges ------------------------------------------------------------------------

    def get_judge(judge_id: str) -> JudgeSpec:
        return JudgeSpec.model_validate(store.get_doc_json("judges", judge_id))

    def judge_json(spec: JudgeSpec) -> dict:
        return {
            **spec.model_dump(),
            "fingerprint": spec.fingerprint,
            "resolved_model": spec.model or default_model(),
        }

    @router.get("/judges")
    def list_judges() -> list[dict]:
        return [judge_json(JudgeSpec.model_validate(j)) for j in store.list_doc_json("judges")]

    @router.post("/judges")
    def create_judge(body: JudgeBody) -> dict:
        judge_id = body.id or _slug(body.name)
        try:
            store.get_doc_json("judges", judge_id)
            raise Conflict(f"judge {judge_id} already exists")
        except NotFound:
            pass
        spec = JudgeSpec(id=judge_id, **body.model_dump(exclude={"id"}))
        store.put_doc_json("judges", judge_id, spec.model_dump())
        return judge_json(spec)

    @router.get("/judges/{judge_id}")
    def read_judge(judge_id: str) -> dict:
        return judge_json(get_judge(judge_id))

    @router.put("/judges/{judge_id}")
    def update_judge(judge_id: str, body: JudgeBody) -> dict:
        """Every change to the judge's contract is a new version; the old one stays in `history`."""
        current = get_judge(judge_id)
        updated = JudgeSpec(id=judge_id, **body.model_dump(exclude={"id"}))
        if updated.contract() != current.contract():
            updated.version = current.version + 1
            updated.history = [
                *current.history,
                {
                    "version": current.version,
                    "fingerprint": current.fingerprint,
                    **current.contract(),
                    "retired_at": now_iso(),
                },
            ]
        else:
            updated.version, updated.history, updated.created_at = (
                current.version,
                current.history,
                current.created_at,
            )
        store.put_doc_json("judges", judge_id, updated.model_dump())
        return judge_json(updated)

    @router.delete("/judges/{judge_id}")
    def delete_judge(judge_id: str) -> dict:
        store.delete_doc_json("judges", judge_id)
        return {"deleted": judge_id}

    @router.post("/judges/{judge_id}/try")
    def try_judge(judge_id: str, body: TryBody) -> dict:
        spec = get_judge(judge_id)
        sample = JudgeSample(id="try", **body.model_dump())
        system, user = jl.build_prompt(spec, sample)
        return {"system": system, "user": user, "result": jl.judge_one(llm, spec, sample)}

    # --- judge sets (samples + human labels) ---------------------------------------------

    def get_set(set_id: str) -> JudgeSet:
        return JudgeSet.model_validate(store.get_doc_json("judge_sets", set_id))

    def save_set(judge_set: JudgeSet) -> None:
        store.put_doc_json("judge_sets", judge_set.id, judge_set.model_dump())

    def set_summary(s: JudgeSet) -> dict:
        raters = Counter(r for sample in s.samples for r in sample.labels)
        return {
            "id": s.id,
            "name": s.name,
            "description": s.description,
            "created_at": s.created_at,
            "samples": len(s.samples),
            "raters": dict(raters),
            "kinds": dict(Counter(sample.kind for sample in s.samples)),
        }

    @router.get("/judge-sets")
    def list_sets() -> list[dict]:
        return [set_summary(JudgeSet.model_validate(s)) for s in store.list_doc_json("judge_sets")]

    @router.post("/judge-sets")
    def create_set(body: JudgeSetBody) -> dict:
        set_id = body.id or _slug(body.name)
        import re

        if not re.match(SLUG, set_id):
            raise HTTPException(422, f"invalid set id {set_id!r}")
        try:
            store.get_doc_json("judge_sets", set_id)
            raise Conflict(f"judge set {set_id} already exists")
        except NotFound:
            pass
        judge_set = JudgeSet(
            id=set_id,
            name=body.name,
            description=body.description,
            samples=_parse_samples(body.jsonl) if body.jsonl.strip() else [],
        )
        save_set(judge_set)
        return judge_set.model_dump()

    @router.get("/judge-sets/{set_id}")
    def read_set(set_id: str) -> dict:
        return get_set(set_id).model_dump()

    @router.delete("/judge-sets/{set_id}")
    def delete_set(set_id: str) -> dict:
        store.delete_doc_json("judge_sets", set_id)
        return {"deleted": set_id}

    @router.post("/judge-sets/{set_id}/import")
    def import_samples(set_id: str, body: ImportSamplesBody) -> dict:
        judge_set = get_set(set_id)
        parsed = _parse_samples(body.jsonl)
        existing = [] if body.mode == "replace" else judge_set.samples
        clash = {s.id for s in existing} & {s.id for s in parsed}
        if clash:
            raise Conflict(f"duplicate sample ids: {', '.join(sorted(clash))}")
        judge_set.samples = existing + parsed
        save_set(judge_set)
        return set_summary(judge_set)

    @router.post("/judge-sets/from-gen-run")
    def from_gen_run(body: FromGenRunBody) -> dict:
        run = store.get_gen_run(body.run_id)
        results = [r for r in run.results if r.error is None and r.answer]
        random.Random(body.seed).shuffle(results)
        samples = [
            JudgeSample(
                id=f"{r.item_id}",
                question=r.query,
                answer=r.answer,
                reference=r.ground_truth,
                contexts=[c.text for c in r.contexts],
                kind=r.outcome or "unknown",
                source=f"gen_run:{run.id}",
            )
            for r in results[: body.limit]
        ]
        set_id = body.id or _slug(body.name)
        judge_set = JudgeSet(
            id=set_id,
            name=body.name,
            description=f"{len(samples)} answers sampled from {run.id} ({run.spec.name}).",
            samples=samples,
        )
        save_set(judge_set)
        return set_summary(judge_set)

    @router.get("/judge-sets/{set_id}/task")
    def task(set_id: str, rater: str) -> dict:
        return jl.blind_task(get_set(set_id), rater)

    @router.put("/judge-sets/{set_id}/labels/{rater}/{sample_id}")
    def label(set_id: str, rater: str, sample_id: str, body: HumanLabel) -> dict:
        judge_set = get_set(set_id)
        sample = next((s for s in judge_set.samples if s.id == sample_id), None)
        if sample is None:
            raise NotFound(f"sample {sample_id}")
        sample.labels[rater] = body.model_copy(update={"at": now_iso()})
        save_set(judge_set)
        return jl.blind_task(judge_set, rater)

    @router.delete("/judge-sets/{set_id}/labels/{rater}")
    def clear_labels(set_id: str, rater: str) -> dict:
        judge_set = get_set(set_id)
        for s in judge_set.samples:
            s.labels.pop(rater, None)
        save_set(judge_set)
        return set_summary(judge_set)

    # --- judge runs & probes ---------------------------------------------------------------

    def save_judge_run(run: JudgeRun) -> None:
        store.put_doc_json("judge_runs", run.id, run.model_dump())

    @router.post("/judge-runs")
    def start_judge_run(body: JudgeRunBody) -> dict:
        spec, judge_set = get_judge(body.judge_id), get_set(body.set_id)
        repeats = body.repeats if body.probe == "consistency" else max(1, body.repeats)

        def work(job) -> dict:
            run = jl.run_judge(
                llm,
                spec,
                judge_set,
                repeats=repeats,
                temperature=body.temperature,
                job=job,
                probe=body.probe,
            )
            save_judge_run(run)
            return {"judge_run_id": run.id}

        return jobs.submit(
            "judge_run", f"{spec.name} v{spec.version} · {judge_set.name}", work
        ).as_dict()

    @router.post("/judge-runs/verbosity")
    def verbosity(body: VerbosityBody) -> dict:
        spec, judge_set = get_judge(body.judge_id), get_set(body.set_id)
        faith = metric_judge() if body.with_faithfulness else None

        def work(job) -> dict:
            run = jl.verbosity_probe(
                llm, spec, judge_set, body.padding, body.limit, body.rater, faith, job
            )
            save_judge_run(run)
            return {"judge_run_id": run.id}

        return jobs.submit("judge_probe", f"verbosity probe · {spec.name}", work).as_dict()

    @router.post("/judge-runs/position")
    def position(body: PositionBody) -> dict:
        spec, judge_set = get_judge(body.judge_id), get_set(body.set_id)

        def work(job) -> dict:
            run = jl.position_probe(
                llm, spec, judge_set, body.rater, body.identical, body.limit, job
            )
            save_judge_run(run)
            return {"judge_run_id": run.id}

        return jobs.submit("judge_probe", f"position probe · {spec.name}", work).as_dict()

    @router.get("/judge-runs")
    def list_judge_runs(set_id: str | None = None) -> list[dict]:
        out = []
        for raw in store.list_doc_json("judge_runs"):
            if set_id and raw.get("set_id") != set_id:
                continue
            raw.pop("results", None)
            out.append(raw)
        return sorted(out, key=lambda r: r.get("created_at", ""), reverse=True)

    @router.get("/judge-runs/drift")
    def drift(a: str, b: str, rater: str | None = None) -> dict:
        run_a = JudgeRun.model_validate(store.get_doc_json("judge_runs", a))
        run_b = JudgeRun.model_validate(store.get_doc_json("judge_runs", b))
        if run_a.set_id != run_b.set_id:
            raise Conflict("drift compares two runs over the SAME judge set")
        return jl.drift_report(get_set(run_a.set_id), run_a, run_b, rater)

    @router.get("/judge-runs/{run_id}")
    def read_judge_run(run_id: str) -> dict:
        return store.get_doc_json("judge_runs", run_id)

    @router.get("/judge-sets/{set_id}/validation")
    def validation(set_id: str, run_id: str, rater: str) -> dict:
        run = JudgeRun.model_validate(store.get_doc_json("judge_runs", run_id))
        if run.set_id != set_id:
            raise Conflict("the judge run belongs to a different set")
        return jl.validation_report(get_set(set_id), run, rater)

    return router
