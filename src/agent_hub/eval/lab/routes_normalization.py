"""Controlled passage-retrieval experiments, separate from know-hub indexes.

The benchmark has immutable imported inputs and its own explicit metrics. It
does not accept Golden Set IDs or expose an alternative route to their holdouts.
Large public corpora are prepared with the CLI, not uploaded through a browser.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from agent_hub.eval.lab.jobs import Job, JobRegistry
from agent_hub.eval.lab.store import LabStore
from agent_hub.eval.normalization import BenchmarkStore, RunConfig, options
from agent_hub.eval.normalization.retrievers import document_text
from agent_hub.eval.normalization.text import normalize


class PassageInput(BaseModel):
    passage_id: str = Field(min_length=1, max_length=300)
    title: str = Field(default="", max_length=2000)
    text: str = Field(min_length=1, max_length=100_000)


class QueryInput(BaseModel):
    qid: str = Field(min_length=1, max_length=300)
    query: str = Field(min_length=1, max_length=10_000)
    tags: list[str] = Field(default_factory=list, max_length=30)


class QrelInput(BaseModel):
    qid: str = Field(min_length=1, max_length=300)
    passage_id: str = Field(min_length=1, max_length=300)
    relevance: Literal[0, 1]


class ImportBody(BaseModel):
    # The portable fixture's human-readable ID is provenance, not a filesystem path.
    id: str | None = Field(default=None, max_length=300)
    name: str = Field(min_length=1, max_length=200)
    source: str = Field(default="User import", max_length=2000)
    scope: str = Field(default="custom", max_length=80)
    corpus: list[PassageInput] = Field(min_length=1, max_length=10_000)
    queries: list[QueryInput] = Field(min_length=1, max_length=3000)
    qrels: list[QrelInput] = Field(min_length=1, max_length=100_000)


class ConfigInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    retrievers: list[Literal["bm25", "dense"]] = Field(
        default_factory=lambda: ["bm25"], min_length=1, max_length=2
    )
    presets: list[Literal["raw", "diacritics", "tatweel", "alef", "conservative", "aggressive"]] = (
        Field(
            default_factory=lambda: [
                "raw",
                "diacritics",
                "tatweel",
                "alef",
                "conservative",
                "aggressive",
            ],
            min_length=1,
            max_length=6,
        )
    )
    normalization_side: Literal["both", "query"] = "both"
    bootstrap_samples: int = Field(default=1000, ge=100, le=20_000)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    dense_model: str | None = Field(default=None, min_length=1, max_length=300)
    dense_revision: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{40}$")
    dense_batch_size: int = Field(default=32, ge=1, le=256)
    dense_max_seq_length: int = Field(default=256, ge=8, le=8192)
    query_prefix: str = Field(default="", max_length=200)
    document_prefix: str = Field(default="", max_length=200)
    top_k: Literal[100] = 100


class RunBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    config: ConfigInput = Field(default_factory=ConfigInput)


def build_router(store: LabStore, jobs: JobRegistry) -> APIRouter:
    router = APIRouter(prefix="/api/eval/normalization", tags=["Arabic normalization"])
    benchmark = BenchmarkStore(store.root / "normalization")
    run_lock = threading.Lock()

    def checked(call, *args, **kwargs):
        try:
            return call(*args, **kwargs)
        except FileNotFoundError as exc:
            raise HTTPException(404, "Normalization dataset or run not found") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/options")
    def get_options() -> dict:
        return options()

    @router.get("/datasets")
    def list_datasets() -> list[dict]:
        return benchmark.list_datasets()

    @router.post("/datasets/import", status_code=201)
    def import_dataset(body: ImportBody) -> dict:
        return checked(
            benchmark.import_records,
            body.name,
            [row.model_dump() for row in body.corpus],
            [row.model_dump() for row in body.queries],
            [row.model_dump() for row in body.qrels],
            provenance={
                "source": body.source,
                "scope": body.scope,
                "source_id": body.id,
                "method": "browser_import",
                "benchmark_comparable": False,
            },
        )

    @router.get("/runs")
    def list_runs(dataset_id: str | None = None) -> list[dict]:
        return [
            run
            for run in benchmark.list_runs()
            if dataset_id is None or run["dataset_id"] == dataset_id
        ]

    @router.post("/runs", status_code=202)
    def start_run(body: RunBody) -> dict:
        checked(benchmark.get_dataset, body.dataset_id)
        config = checked(RunConfig, **body.config.model_dump())
        checked(config.validate)
        if not run_lock.acquire(blocking=False):
            raise HTTPException(
                409, "A normalization run is already running; wait for it to finish"
            )

        def work(job: Job) -> dict:
            try:

                def progress(event: dict) -> None:
                    job.progress(
                        event.get("completed", 0),
                        event.get("total", 0),
                        event.get("detail") or event.get("stage", "Running"),
                    )

                report = benchmark.run(body.dataset_id, config, progress=progress)
                return {"run_id": report["run_id"]}
            finally:
                run_lock.release()

        try:
            return jobs.submit("normalization", "Arabic normalization experiment", work).as_dict()
        except Exception:
            run_lock.release()
            raise

    @router.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        return checked(benchmark.get_summary, run_id)

    @router.get("/runs/{run_id}/queries")
    def get_queries(
        run_id: str,
        condition: int = Query(default=0, ge=0),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=20, ge=1, le=100),
        ranking_limit: int = Query(default=10, ge=1, le=100),
        qid: str | None = None,
    ) -> dict:
        report = checked(benchmark.get_run, run_id)
        if condition >= len(report["conditions"]):
            raise HTTPException(422, "Condition index is outside this run")
        rows = report["conditions"][condition]["per_query"]
        if qid is not None:
            rows = [row for row in rows if row["qid"] == qid]
        return {
            "total": len(rows),
            "offset": offset,
            "limit": limit,
            "items": [
                {**row, "ranking": row["ranking"][:ranking_limit]}
                for row in rows[offset : offset + limit]
            ],
        }

    @router.get("/runs/{run_id}/export")
    def export_run(run_id: str) -> FileResponse:
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise HTTPException(422, "Invalid normalization run ID")
        path = benchmark.root / "runs" / f"{run_id}.json"
        if not path.is_file():
            raise HTTPException(404, "Normalization run not found")
        return FileResponse(
            path, media_type="application/json", filename=f"normalization-{run_id}.json"
        )

    @router.get("/runs/{run_id}/passage")
    def get_passage(
        run_id: str,
        passage_id: str = Query(min_length=1, max_length=300),
        condition: int = Query(default=0, ge=0),
    ) -> dict:
        report = checked(benchmark.get_summary, run_id)
        if condition >= len(report["conditions"]):
            raise HTTPException(422, "Condition index is outside this run")
        dataset = checked(benchmark.get_dataset, report["dataset_id"])
        corpus = benchmark.root / "datasets" / dataset["dataset_id"] / "corpus.sqlite"
        # The dataset ID comes from the validated store, and IDs within the corpus
        # are SQL parameters: original MIRACL IDs (including '#') remain untouched.
        connection = sqlite3.connect(f"{corpus.resolve().as_uri()}?mode=ro", uri=True)
        try:
            row = connection.execute(
                "SELECT title, text FROM corpus WHERE passage_id=?", (passage_id,)
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise HTTPException(404, "Passage not found in this run's corpus")
        title, text = row
        original = document_text(title, text)
        preset = (
            report["conditions"][condition]["preset"]
            if report["config"]["normalization_side"] == "both"
            else "raw"
        )
        return {
            "passage_id": passage_id,
            "title": title,
            "text": text,
            "retrieval_input": original,
            "normalized_input": normalize(original, preset),
            "preset": preset,
            "normalization_side": report["config"]["normalization_side"],
        }

    return router
