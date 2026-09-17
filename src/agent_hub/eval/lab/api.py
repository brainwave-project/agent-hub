"""Eval Lab HTTP API — the backend of the UI's Evaluation section.

Mounted at /api/eval on its own port (3005). Not routed through Kong: like the
eval-runner it talks to know-hub with synthetic headers, so it belongs only on
the lean eval stack, never on a shared or exposed network.

Guards that encode experiment discipline rather than validation:

* Frozen dataset versions reject every write (409) — fork instead.
* Ad-hoc runs may not touch the holdout split (409). The holdout is evaluated
  only through a registered experiment, where every access is logged.
* Registration requires a frozen, split, blocker-free dataset and records its
  hash; a registered experiment's design can no longer be edited.
* A second holdout evaluation requires a written reason, which is kept.
"""

from __future__ import annotations

import os
import pathlib
import re
import threading
import time
from typing import Literal

import httpx
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from agent_hub.eval.datasets import EVAL_DIR, load_corpus, load_golden
from agent_hub.eval.lab import analysis
from agent_hub.eval.lab.chunking import (
    KNOWHUB_MAX_TOKENS,
    KNOWHUB_OVERLAP,
    chunk_text,
    locate_evidence,
)
from agent_hub.eval.lab.jobs import JobRegistry
from agent_hub.eval.lab.models import (
    LAB_METRICS,
    SLUG,
    CorpusDoc,
    DatasetMeta,
    Experiment,
    GoldenItem,
    HoldoutAccess,
    Normalization,
    SplitConfig,
    SuccessCriteria,
    VariantSpec,
    now_iso,
)
from agent_hub.eval.lab.readiness import assess
from agent_hub.eval.lab.runner import (
    IndexNotReady,
    KnowHubLab,
    index_corpus,
    index_is_current,
    run_split,
    search_once,
)
from agent_hub.eval.lab.splits import assign_split, summarize
from agent_hub.eval.lab.store import Bundle, Conflict, Immutable, LabStore, NotFound

__all__ = ["create_app", "variant_diff"]

DEFAULT_LAB_DIR = EVAL_DIR / "lab_store"


# --- request bodies ------------------------------------------------------------


class CreateDataset(BaseModel):
    name: str = Field(min_length=1)
    id: str | None = None
    description: str = ""
    lang: str = "en"


class ImportBuiltin(BaseModel):
    lang: Literal["en", "ar"] = "en"
    version: str = "v1"


class PatchMeta(BaseModel):
    name: str | None = None
    description: str | None = None


class PatchVersion(BaseModel):
    changelog: str = ""


class ImportItems(BaseModel):
    jsonl: str
    mode: Literal["append", "replace"] = "append"


class ForkBody(BaseModel):
    changelog: str = ""


class IndexBody(BaseModel):
    normalization: Normalization = "off"


class SearchBody(BaseModel):
    dataset_id: str
    version: int
    variant: VariantSpec = Field(default_factory=VariantSpec)
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=50)
    item_id: str | None = None
    relevant: dict[str, int] | None = None


class AdhocRunBody(BaseModel):
    dataset_id: str
    version: int
    variant: VariantSpec = Field(default_factory=VariantSpec)
    split: Literal["dev", "holdout", "all"] = "dev"
    k: int = Field(default=5, ge=1, le=50)


class CompareBody(BaseModel):
    run_a: str
    run_b: str
    confidence: float = Field(default=0.95, ge=0.5, le=0.999)


class ExperimentBody(BaseModel):
    name: str = Field(min_length=1)
    id: str | None = None
    hypothesis: str = ""
    rationale: str = ""
    dataset_id: str
    dataset_version: int
    baseline: VariantSpec = Field(default_factory=lambda: VariantSpec(name="baseline"))
    treatment: VariantSpec = Field(default_factory=lambda: VariantSpec(name="treatment"))
    criteria: SuccessCriteria = Field(default_factory=SuccessCriteria)
    secondary_metrics: list[str] = Field(default_factory=list)


class ExperimentRunBody(BaseModel):
    split: Literal["dev", "holdout"] = "dev"
    arms: list[Literal["baseline", "treatment"]] = Field(
        default_factory=lambda: ["baseline", "treatment"]
    )
    reason: str = ""


class ConcludeBody(BaseModel):
    note: str = ""


# --- helpers ---------------------------------------------------------------------


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip().lower()).strip("-.")
    return (slug or f"item-{int(time.time())}")[:100]


def variant_diff(a: VariantSpec, b: VariantSpec) -> list[str]:
    return [f for f in VariantSpec.model_fields if f != "name" and getattr(a, f) != getattr(b, f)]


def _bundle_json(bundle: Bundle) -> dict:
    relevant_for: dict[str, list[str]] = {}
    for item in bundle.items:
        for doc_id in item.relevant:
            relevant_for.setdefault(doc_id, []).append(item.id)
    indexes = {}
    for norm, state in bundle.indexes.items():
        current, reason = index_is_current(bundle, norm)
        statuses: dict[str, int] = {}
        for entry in state.documents.values():
            statuses[entry.status] = statuses.get(entry.status, 0) + 1
        indexes[norm] = {
            **state.model_dump(),
            "current": current,
            "reason": reason,
            "statuses": statuses,
        }
    return {
        "meta": bundle.meta.model_dump(),
        "manifest": bundle.manifest.model_dump(),
        "current_hash": bundle.current_hash,
        "corpus_hashes": {n: bundle.corpus_hash(n) for n in ("off", "conservative", "aggressive")},
        "corpus": [
            {
                **d.model_dump(),
                "words": len(d.body.split()),
                "relevant_for": relevant_for.get(d.doc_id, []),
            }
            for d in sorted(bundle.corpus.values(), key=lambda d: d.doc_id)
        ],
        "items": [i.model_dump() for i in bundle.items],
        "indexes": indexes,
        "split_summary": summarize(bundle.items, bundle.manifest.split_config).as_dict(),
    }


def _parse_item(raw: dict) -> GoldenItem:
    """Accept both the lab schema and the offline harness's golden_v1 schema."""
    data = dict(raw)
    if "relevant_doc_ids" in data and "relevant" not in data:
        data["relevant"] = {d: 3 for d in data.pop("relevant_doc_ids")}
    if "holdout" in data and "split" not in data:
        data["split"] = "holdout" if data.pop("holdout") else "dev"
    known = set(GoldenItem.model_fields)
    return GoldenItem.model_validate({k: v for k, v in data.items() if k in known})


# --- app ---------------------------------------------------------------------------


def create_app(store: LabStore | None = None, knowhub: KnowHubLab | None = None) -> FastAPI:
    store = store or LabStore(pathlib.Path(os.environ.get("EVAL_LAB_DIR", str(DEFAULT_LAB_DIR))))
    kh = knowhub or KnowHubLab(
        base_url=os.environ.get("KNOW_HUB_URL", "http://know-hub:3002"),
        grpc_host=os.environ.get("KNOW_HUB_GRPC_HOST", "know-hub"),
        grpc_port=int(os.environ.get("KNOW_HUB_GRPC_PORT", "50052")),
    )
    jobs = JobRegistry()
    experiment_lock = threading.Lock()
    report_cache: dict[tuple, dict] = {}

    app = FastAPI(
        title="Brainwave Eval Lab",
        description="Dataset construction, data readiness, retrieval evaluation and experiments.",
        docs_url="/api/eval/docs",
        openapi_url="/api/eval/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("EVAL_LAB_CORS", "http://localhost:5173").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _error(status: int, code: str, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=status, content={"code": code, "detail": str(exc)})

    @app.exception_handler(NotFound)
    async def _not_found(_: Request, exc: NotFound):
        return _error(404, "not_found", exc)

    @app.exception_handler(Immutable)
    async def _immutable(_: Request, exc: Immutable):
        return _error(409, "immutable", exc)

    @app.exception_handler(Conflict)
    async def _conflict(_: Request, exc: Conflict):
        return _error(409, "conflict", exc)

    @app.exception_handler(IndexNotReady)
    async def _index(_: Request, exc: IndexNotReady):
        return _error(409, "index_not_ready", exc)

    @app.exception_handler(ValidationError)
    async def _invalid(_: Request, exc: ValidationError):
        return JSONResponse(status_code=422, content={"code": "invalid", "detail": exc.errors()})

    @app.exception_handler(httpx.HTTPError)
    async def _upstream(_: Request, exc: httpx.HTTPError):
        return _error(502, "know_hub_unreachable", exc)

    api = APIRouter(prefix="/api/eval")

    @app.get("/health/live")
    def live() -> dict:
        return {"status": "ok"}

    # --- status & jobs ----------------------------------------------------------

    @api.get("/status")
    def status() -> dict:
        try:
            know_hub = kh.ready()
        except Exception:  # noqa: BLE001
            know_hub = False
        return {
            "eval_api": True,
            "know_hub": know_hub,
            "know_hub_grpc": kh.grpc_ready() if know_hub else False,
            "know_hub_url": getattr(kh, "base_url", ""),
            "lab_dir": str(store.root),
            "metrics": list(LAB_METRICS),
            "chunking": {"max_tokens": KNOWHUB_MAX_TOKENS, "overlap": KNOWHUB_OVERLAP},
            "time": now_iso(),
        }

    @api.get("/jobs")
    def list_jobs() -> list[dict]:
        return [j.as_dict() for j in jobs.list()[:50]]

    @api.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found (jobs are in-memory and reset on restart)")
        return job.as_dict()

    # --- datasets -----------------------------------------------------------------

    @api.get("/datasets")
    def list_datasets() -> list[dict]:
        return store.list_datasets()

    @api.post("/datasets")
    def create_dataset(body: CreateDataset) -> dict:
        dataset_id = body.id or _slugify(body.name)
        if not re.match(SLUG, dataset_id):
            raise HTTPException(422, f"invalid dataset id {dataset_id!r}")
        store.create_dataset(
            DatasetMeta(id=dataset_id, name=body.name, description=body.description, lang=body.lang)
        )
        return _bundle_json(store.bundle(dataset_id, 1))

    @api.post("/datasets/import-builtin")
    def import_builtin(body: ImportBuiltin) -> dict:
        dataset_id = f"harness-{body.lang}-golden-{body.version}"
        store.create_dataset(
            DatasetMeta(
                id=dataset_id,
                name=f"Harness golden {body.version} ({body.lang})",
                description="Imported from agent_hub/eval/datasets — the offline harness's golden set.",
                lang=body.lang,
            )
        )
        for doc_id, text in load_corpus(body.lang).items():
            store.put_doc(
                dataset_id,
                1,
                CorpusDoc(
                    doc_id=doc_id,
                    lang=body.lang,
                    body=text,
                    source=f"eval/corpus/{body.lang}/{doc_id}.md",
                ),
            )
        records = load_golden(body.lang, body.version, include_holdout=True)
        store.replace_items(
            dataset_id,
            1,
            [
                GoldenItem(
                    id=r.id,
                    query=r.query,
                    lang=r.lang,
                    relevant={d: 3 for d in r.relevant_doc_ids},
                    ground_truth=r.ground_truth,
                    difficulty=r.difficulty
                    if r.difficulty in ("easy", "medium", "hard")
                    else "medium",
                    trap=r.trap,
                    split="holdout" if r.holdout else "dev",
                    origin="imported",
                )
                for r in records
            ],
        )
        return _bundle_json(store.bundle(dataset_id, 1))

    @api.patch("/datasets/{dataset_id}")
    def patch_dataset(dataset_id: str, body: PatchMeta) -> dict:
        return store.update_meta(dataset_id, body.name, body.description).model_dump()

    @api.get("/datasets/{dataset_id}/versions/{version}")
    def get_version(dataset_id: str, version: int) -> dict:
        return _bundle_json(store.bundle(dataset_id, version))

    @api.patch("/datasets/{dataset_id}/versions/{version}")
    def patch_version(dataset_id: str, version: int, body: PatchVersion) -> dict:
        return store.set_changelog(dataset_id, version, body.changelog).model_dump()

    @api.put("/datasets/{dataset_id}/versions/{version}/corpus/{doc_id}")
    def put_doc(dataset_id: str, version: int, doc_id: str, doc: CorpusDoc) -> dict:
        if doc.doc_id != doc_id:
            raise HTTPException(422, "doc_id in body must match the path")
        store.put_doc(dataset_id, version, doc)
        return _bundle_json(store.bundle(dataset_id, version))

    @api.delete("/datasets/{dataset_id}/versions/{version}/corpus/{doc_id}")
    def delete_doc(dataset_id: str, version: int, doc_id: str) -> dict:
        store.delete_doc(dataset_id, version, doc_id)
        return _bundle_json(store.bundle(dataset_id, version))

    @api.put("/datasets/{dataset_id}/versions/{version}/items/{item_id}")
    def put_item(dataset_id: str, version: int, item_id: str, item: GoldenItem) -> dict:
        if item.id != item_id:
            raise HTTPException(422, "item id in body must match the path")
        store.put_item(dataset_id, version, item)
        return _bundle_json(store.bundle(dataset_id, version))

    @api.delete("/datasets/{dataset_id}/versions/{version}/items/{item_id}")
    def delete_item(dataset_id: str, version: int, item_id: str) -> dict:
        store.delete_item(dataset_id, version, item_id)
        return _bundle_json(store.bundle(dataset_id, version))

    @api.post("/datasets/{dataset_id}/versions/{version}/items/import")
    def import_items(dataset_id: str, version: int, body: ImportItems) -> dict:
        import json

        parsed: list[GoldenItem] = []
        errors: list[str] = []
        for n, line in enumerate(body.jsonl.splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                parsed.append(_parse_item(json.loads(line)))
            except (ValueError, ValidationError) as exc:
                errors.append(f"line {n}: {exc}")
        if errors:
            raise HTTPException(422, {"code": "invalid_jsonl", "errors": errors[:20]})
        existing = [] if body.mode == "replace" else store.items(dataset_id, version)
        store.replace_items(dataset_id, version, existing + parsed)
        return _bundle_json(store.bundle(dataset_id, version))

    @api.post("/datasets/{dataset_id}/versions/{version}/split")
    def split(dataset_id: str, version: int, config: SplitConfig) -> dict:
        manifest = store.manifest(dataset_id, version)
        items, summary = assign_split(store.items(dataset_id, version), config)
        manifest.split_config = config
        store.set_split(dataset_id, version, items, manifest)
        return {
            "summary": summary.as_dict(),
            "bundle": _bundle_json(store.bundle(dataset_id, version)),
        }

    @api.post("/datasets/{dataset_id}/versions/{version}/freeze")
    def freeze(dataset_id: str, version: int) -> dict:
        store.freeze(dataset_id, version)
        return _bundle_json(store.bundle(dataset_id, version))

    @api.post("/datasets/{dataset_id}/versions/{version}/fork")
    def fork(dataset_id: str, version: int, body: ForkBody) -> dict:
        manifest = store.fork(dataset_id, version, body.changelog)
        return _bundle_json(store.bundle(dataset_id, manifest.version))

    @api.get("/datasets/{dataset_id}/versions/{version}/export")
    def export(dataset_id: str, version: int) -> dict:
        bundle = store.bundle(dataset_id, version)
        return {
            "manifest": bundle.manifest.model_dump(),
            "content_hash": bundle.current_hash,
            "golden_jsonl": "".join(i.model_dump_json() + "\n" for i in bundle.items),
            "corpus": [d.model_dump() for d in bundle.corpus.values()],
        }

    @api.get("/datasets/{dataset_id}/versions/{version}/chunks/{doc_id}")
    def chunks(
        dataset_id: str,
        version: int,
        doc_id: str,
        max_tokens: int = Query(KNOWHUB_MAX_TOKENS, ge=16, le=4096),
        overlap: int = Query(KNOWHUB_OVERLAP, ge=0, le=1024),
    ) -> dict:
        bundle = store.bundle(dataset_id, version)
        doc = bundle.corpus.get(doc_id)
        if doc is None:
            raise NotFound(f"document {doc_id}")
        pieces = chunk_text(doc.body, max_tokens, overlap)
        evidence = [
            {"item_id": item.id, "text": ev.text, **locate_evidence(doc.body, ev.text, pieces)}
            for item in bundle.items
            for ev in item.evidence
            if ev.doc_id == doc_id
        ]
        return {
            "doc_id": doc_id,
            "config": {"max_tokens": max_tokens, "overlap": overlap},
            "is_knowhub_config": (max_tokens, overlap) == (KNOWHUB_MAX_TOKENS, KNOWHUB_OVERLAP),
            "words": len(doc.body.split()),
            "chunks": [c.__dict__ for c in pieces],
            "evidence": evidence,
        }

    @api.post("/datasets/{dataset_id}/versions/{version}/index")
    def index(dataset_id: str, version: int, body: IndexBody) -> dict:
        bundle = store.bundle(dataset_id, version)
        if not bundle.corpus:
            raise Conflict("corpus is empty")
        job = jobs.submit(
            "index",
            f"index {dataset_id} v{version} ({body.normalization})",
            lambda job: index_corpus(kh, store, bundle, body.normalization, job).model_dump(),
        )
        return job.as_dict()

    @api.get("/datasets/{dataset_id}/versions/{version}/readiness")
    def readiness(
        dataset_id: str,
        version: int,
        k: int = Query(5, ge=1, le=50),
        score_sd: float | None = Query(None, gt=0, le=1),
    ) -> dict:
        return assess(
            store.bundle(dataset_id, version),
            k=k,
            score_sd=score_sd,
            pool=store.artifact(dataset_id, version, "pool"),
            annotations=store.artifact(dataset_id, version, "annotations"),
        )

    # --- retrieval ------------------------------------------------------------------

    @api.post("/search")
    def search(body: SearchBody) -> dict:
        bundle = store.bundle(body.dataset_id, body.version)
        item = (
            next((i for i in bundle.items if i.id == body.item_id), None) if body.item_id else None
        )
        if body.item_id and item is None:
            raise NotFound(f"item {body.item_id}")
        result, explanation = search_once(
            kh, bundle, body.variant, body.query, body.k, item, body.relevant
        )

        notes = []
        if body.variant.level == "chunk":
            notes.append(
                "Chunk level uses gRPC SemanticSearch: vector-only, and min_score 0 falls back "
                "to know-hub's SIMILARITY_THRESHOLD (0.0 on the eval stack, 0.60 on dev)."
            )
        if result.query_sent != body.query:
            notes.append("The query was normalised before embedding — see query_sent.")
        if item and item.split == "holdout":
            notes.append(
                "This is a HOLDOUT item. Looking at its results and then changing the system turns "
                "it into a dev item. Prefer dev items while iterating."
            )
        if not result.relevant_units:
            notes.append(
                "No relevance judgements for this query — metrics are undefined. Mark results as "
                "relevant to score them."
            )
        return {"result": result.model_dump(), "explain": explanation, "notes": notes}

    @api.get("/runs")
    def list_runs(dataset_id: str | None = None) -> list[dict]:
        return store.list_runs(dataset_id)

    @api.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        return store.get_run(run_id).model_dump()

    @api.post("/runs")
    def adhoc_run(body: AdhocRunBody) -> dict:
        bundle = store.bundle(body.dataset_id, body.version)
        has_holdout = any(i.split == "holdout" for i in bundle.items)
        if body.split == "holdout" or (body.split == "all" and has_holdout):
            raise Conflict(
                "Ad-hoc runs cannot evaluate holdout items. Register an experiment — holdout access "
                "is only allowed there, where every look is logged."
            )
        ok, why = index_is_current(bundle, body.variant.corpus_index)
        if not ok:
            raise IndexNotReady(why)
        job = jobs.submit(
            "run",
            f"ad-hoc {body.variant.name} on {body.split}",
            lambda job: {
                "run_id": run_split(kh, store, bundle, body.variant, body.split, body.k, job).id
            },
        )
        return job.as_dict()

    @api.post("/compare")
    def compare(body: CompareBody) -> dict:
        run_a, run_b = store.get_run(body.run_a), store.get_run(body.run_b)
        items = store.items(run_a.dataset_id, run_a.dataset_version)
        return analysis.compare_runs(run_a, run_b, items, body.confidence)

    # --- experiments ---------------------------------------------------------------

    def _experiment_json(experiment: Experiment) -> dict:
        return {
            **experiment.model_dump(),
            "variant_diff": variant_diff(experiment.baseline, experiment.treatment),
        }

    @api.get("/experiments")
    def list_experiments() -> list[dict]:
        return [_experiment_json(e) for e in store.list_experiments()]

    @api.post("/experiments")
    def create_experiment(body: ExperimentBody) -> dict:
        experiment_id = body.id or _slugify(body.name)
        try:
            store.get_experiment(experiment_id)
            raise Conflict(f"experiment {experiment_id} already exists")
        except NotFound:
            pass
        store.manifest(body.dataset_id, body.dataset_version)  # 404 if absent
        experiment = Experiment(id=experiment_id, **body.model_dump(exclude={"id"}))
        store.save_experiment(experiment)
        return _experiment_json(experiment)

    @api.get("/experiments/{experiment_id}")
    def get_experiment(experiment_id: str) -> dict:
        return _experiment_json(store.get_experiment(experiment_id))

    @api.put("/experiments/{experiment_id}")
    def update_experiment(experiment_id: str, body: ExperimentBody) -> dict:
        with experiment_lock:
            experiment = store.get_experiment(experiment_id)
            if experiment.status != "draft":
                raise Immutable(
                    "registered experiments cannot be edited — the design was fixed before the data was "
                    "seen. Create a new experiment instead."
                )
            updated = experiment.model_copy(update=body.model_dump(exclude={"id"}))
            store.save_experiment(Experiment.model_validate(updated.model_dump()))
            return _experiment_json(store.get_experiment(experiment_id))

    @api.delete("/experiments/{experiment_id}")
    def delete_experiment(experiment_id: str) -> dict:
        store.delete_experiment(experiment_id)
        return {"deleted": experiment_id}

    @api.post("/experiments/{experiment_id}/register")
    def register(experiment_id: str) -> dict:
        with experiment_lock:
            experiment = store.get_experiment(experiment_id)
            if experiment.status != "draft":
                raise Conflict(f"experiment is already {experiment.status}")
            bundle = store.bundle(experiment.dataset_id, experiment.dataset_version)
            problems: list[str] = []
            if bundle.manifest.status != "frozen":
                problems.append(
                    "The dataset version must be frozen — results must be tied to a content hash."
                )
            if not experiment.hypothesis.strip():
                problems.append("Write the hypothesis down before seeing any result.")
            if experiment.criteria.primary_metric not in LAB_METRICS:
                problems.append(f"Unknown primary metric {experiment.criteria.primary_metric!r}.")
            if not variant_diff(experiment.baseline, experiment.treatment):
                problems.append("Baseline and treatment are identical — there is nothing to test.")
            report = assess(bundle, k=experiment.criteria.k)
            blockers = [
                c["title"]
                for c in report["checks"]
                if c["severity"] == "blocker" and c["category"] != "Index"
            ]
            problems += [f"Data Lab blocker: {b}" for b in blockers]
            if report["stats"]["holdout"] == 0:
                problems.append("The holdout split has no answerable items.")
            if problems:
                raise HTTPException(409, {"code": "not_registrable", "problems": problems})
            experiment.status = "registered"
            experiment.registered_at = now_iso()
            experiment.dataset_hash = bundle.current_hash
            store.save_experiment(experiment)
            return _experiment_json(experiment)

    @api.post("/experiments/{experiment_id}/runs")
    def run_experiment(experiment_id: str, body: ExperimentRunBody) -> dict:
        with experiment_lock:
            experiment = store.get_experiment(experiment_id)
            if experiment.status == "concluded":
                raise Conflict("experiment is concluded — its record is final")
            bundle = store.bundle(experiment.dataset_id, experiment.dataset_version)
            access_index: int | None = None
            if body.split == "holdout":
                if experiment.status != "registered":
                    raise Conflict("register the experiment before evaluating on the holdout split")
                if bundle.current_hash != experiment.dataset_hash:
                    raise Conflict(
                        "dataset content no longer matches the hash recorded at registration"
                    )
                if experiment.holdout_access and not body.reason.strip():
                    raise HTTPException(
                        422,
                        {
                            "code": "reason_required",
                            "detail": f"The holdout has already been evaluated {len(experiment.holdout_access)} "
                            "time(s). State why you are looking again — it will be kept in the record.",
                        },
                    )
                experiment.holdout_access.append(HoldoutAccess(reason=body.reason))
                access_index = len(experiment.holdout_access) - 1
                store.save_experiment(experiment)
            for arm in body.arms:
                variant = experiment.baseline if arm == "baseline" else experiment.treatment
                ok, why = index_is_current(bundle, variant.corpus_index)
                if not ok:
                    raise IndexNotReady(f"{arm}: {why}")

        def record(run) -> None:
            with experiment_lock:
                current = store.get_experiment(experiment_id)
                current.runs.append(run.id)
                if access_index is not None:
                    current.holdout_access[access_index].run_ids.append(run.id)
                store.save_experiment(current)

        def work(job) -> dict:
            produced = []
            for arm in body.arms:
                variant = experiment.baseline if arm == "baseline" else experiment.treatment
                run = run_split(
                    kh,
                    store,
                    bundle,
                    variant,
                    body.split,
                    experiment.criteria.k,
                    job,
                    experiment_id=experiment_id,
                    arm=arm,
                    on_start=record,
                )
                produced.append(run.id)
            return {"run_ids": produced}

        job = jobs.submit(
            "experiment", f"{experiment_id} · {body.split} · {'+'.join(body.arms)}", work
        )
        return job.as_dict()

    @api.get("/experiments/{experiment_id}/report")
    def experiment_report(experiment_id: str, split: Literal["dev", "holdout"] = "dev") -> dict:
        experiment = store.get_experiment(experiment_id)
        runs = []
        for run_id in experiment.runs:
            try:
                runs.append(store.get_run(run_id))
            except NotFound:
                continue
        bundle = store.bundle(experiment.dataset_id, experiment.dataset_version)
        key = (
            experiment_id,
            split,
            tuple((r.id, r.status) for r in runs),
            experiment.model_dump_json(),
            bundle.current_hash,
        )
        if key not in report_cache:
            report = analysis.experiment_report(
                experiment, runs, bundle.items, bundle.current_hash, split
            )
            changed = variant_diff(experiment.baseline, experiment.treatment)
            report["variant_diff"] = changed
            if len(changed) > 1:
                report["warnings"].append(
                    {
                        "severity": "warning",
                        "title": f"Confounded design: {len(changed)} variables differ",
                        "detail": f"Arms differ in {', '.join(changed)}. Any effect cannot be attributed to one "
                        "of them. Change one variable per experiment.",
                    }
                )
            report_cache.clear()
            report_cache[key] = report
        return report_cache[key]

    @api.post("/experiments/{experiment_id}/conclude")
    def conclude(experiment_id: str, body: ConcludeBody) -> dict:
        with experiment_lock:
            experiment = store.get_experiment(experiment_id)
            if experiment.status != "registered":
                raise Conflict("only a registered experiment can be concluded")
        report = experiment_report(experiment_id, "holdout")
        if report.get("status") != "ok":
            raise Conflict("run both arms on the holdout split before concluding")
        with experiment_lock:
            experiment = store.get_experiment(experiment_id)
            experiment.status = "concluded"
            experiment.concluded_at = now_iso()
            experiment.conclusion_note = body.note
            experiment.conclusion = {
                "decision": report["decision"],
                "confirmatory": report["confirmatory"],
                "primary": report["metrics"][experiment.criteria.primary_metric]["delta"]
                | {"per_query": None},
                "baseline_run": report["baseline_run"],
                "treatment_run": report["treatment_run"],
                "holdout_access_count": len(experiment.holdout_access),
                "warnings": report["warnings"],
            }
            store.save_experiment(experiment)
            return _experiment_json(experiment)

    app.include_router(api)
    # Imported here: both modules import from this package's runner/store, and
    # keeping them out of module scope keeps api.py importable on its own.
    from agent_hub.eval.lab import routes_insight, routes_labeling

    app.include_router(routes_labeling.build_router(store, kh, jobs, _bundle_json))
    app.include_router(routes_insight.build_router(store))
    return app
