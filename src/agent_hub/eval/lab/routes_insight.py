"""HTTP routes for insight: Stats Lab simulations, run analysis, and the Learning Path.

Simulations are synchronous and bounded by `simulate.MAX_WORK`, so a request
takes seconds, not minutes. They are stateless: the same body always returns
the same result (every simulation is seeded).
"""

from __future__ import annotations

import random
import statistics

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from agent_hub.eval.lab.learning import evaluate, update_progress
from agent_hub.eval.lab.runanalysis import TAXONOMY, failure_analysis, k_sweep, threshold_sweep
from agent_hub.eval.lab.simulate import (
    MAX_WORK,
    PopulationSpec,
    bootstrap_demo,
    clt_sim,
    coverage_sim,
    phacking_sim,
    population_values,
    power_sim,
)
from agent_hub.eval.lab.store import LabStore, NotFound

__all__ = ["build_router"]


class BootstrapBody(BaseModel):
    population: PopulationSpec = Field(default_factory=PopulationSpec)
    #: Sample size to draw from a synthetic population. A run population is used as-is.
    n: int | None = Field(default=None, ge=2, le=1000)
    resamples: int = Field(default=2000, ge=100, le=20000)
    confidence: float = Field(default=0.95, ge=0.5, le=0.999)
    seed: int = 7


class CoverageBody(BaseModel):
    population: PopulationSpec = Field(default_factory=PopulationSpec)
    n: int = Field(default=10, ge=2, le=500)
    sims: int = Field(default=300, ge=50, le=2000)
    resamples: int = Field(default=500, ge=100, le=5000)
    confidence: float = Field(default=0.95, ge=0.5, le=0.999)
    seed: int = 7


class CltBody(BaseModel):
    population: PopulationSpec = Field(default_factory=lambda: PopulationSpec(kind="skewed"))
    ns: list[int] = Field(default_factory=lambda: [2, 5, 30, 100], min_length=1, max_length=6)
    sims: int = Field(default=2000, ge=100, le=10000)
    seed: int = 7


class PowerBody(BaseModel):
    n: int = Field(default=50, ge=5, le=2000)
    effect: float = Field(default=0.1, ge=-1.0, le=1.0)
    discordance: float = Field(default=0.3, ge=0.0, le=1.0)
    sims: int = Field(default=300, ge=50, le=2000)
    resamples: int = Field(default=500, ge=100, le=5000)
    confidence: float = Field(default=0.95, ge=0.5, le=0.999)
    seed: int = 7


class PhackingBody(BaseModel):
    n: int = Field(default=40, ge=5, le=1000)
    metrics: int = Field(default=7, ge=1, le=20)
    sims: int = Field(default=500, ge=50, le=3000)
    confidence: float = Field(default=0.95, ge=0.5, le=0.999)
    discordance: float = Field(default=0.3, ge=0.01, le=1.0)
    correlation: float = Field(default=0.5, ge=0.0, le=1.0)
    batch: int = Field(default=10, ge=1, le=500)
    seed: int = 7


class TagBody(BaseModel):
    tags: list[str] = Field(default_factory=list, max_length=8)
    note: str = Field(default="", max_length=1000)


class ProgressBody(BaseModel):
    done: bool = False
    reflection: str = Field(default="", max_length=5000)


def _guard(work: int) -> None:
    if work > MAX_WORK:
        raise HTTPException(
            422,
            f"simulation too large ({work:,} resample draws > {MAX_WORK:,}) — lower sims, resamples or n",
        )


def build_router(store: LabStore) -> APIRouter:
    router = APIRouter(prefix="/api/eval")

    def values_for(spec: PopulationSpec) -> tuple[list[float], str]:
        try:
            return population_values(spec, store)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    # --- Stats Lab -------------------------------------------------------------------

    @router.post("/stats/bootstrap")
    def bootstrap(body: BootstrapBody) -> dict:
        values, description = values_for(body.population)
        population_mean = statistics.fmean(values)
        if body.population.kind == "run" and body.n is None:
            sample = values
        else:
            sample = random.Random(body.seed).choices(values, k=body.n or 20)
        _guard(body.resamples * len(sample))
        return {
            **bootstrap_demo(sample, body.resamples, body.confidence, body.seed),
            "population": description,
            "population_mean": population_mean,
            "sampled": sample is not values,
            "sample": sample if len(sample) <= 200 else None,
        }

    @router.post("/stats/coverage")
    def coverage(body: CoverageBody) -> dict:
        values, description = values_for(body.population)
        _guard(body.sims * body.resamples * body.n)
        return {
            **coverage_sim(values, body.n, body.sims, body.resamples, body.confidence, body.seed),
            "population": description,
        }

    @router.post("/stats/clt")
    def clt(body: CltBody) -> dict:
        values, description = values_for(body.population)
        if any(n < 1 or n > 2000 for n in body.ns):
            raise HTTPException(422, "each n must be between 1 and 2000")
        _guard(body.sims * sum(body.ns))
        return {
            **clt_sim(values, sorted(set(body.ns)), body.sims, body.seed),
            "population": description,
        }

    @router.post("/stats/power")
    def power(body: PowerBody) -> dict:
        _guard(body.sims * body.resamples * body.n)
        try:
            return power_sim(
                body.n,
                body.effect,
                body.discordance,
                body.sims,
                body.resamples,
                body.confidence,
                body.seed,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/stats/phacking")
    def phacking(body: PhackingBody) -> dict:
        _guard(body.sims * body.n * (body.metrics + max(1, body.n // body.batch)))
        return phacking_sim(
            body.n,
            body.metrics,
            body.sims,
            body.confidence,
            body.discordance,
            body.correlation,
            body.batch,
            body.seed,
        )

    # --- run analysis ----------------------------------------------------------------

    @router.get("/runs/{run_id}/sweeps")
    def sweeps(run_id: str, max_k: int = Query(20, ge=1, le=50)) -> dict:
        run = store.get_run(run_id)
        return {"k": k_sweep(run, max_k), "threshold": threshold_sweep(run)}

    @router.get("/runs/{run_id}/failures")
    def failures(
        run_id: str, metric: str = "recall@k", threshold: float = Query(1.0, ge=0.0, le=1.0)
    ) -> dict:
        run = store.get_run(run_id)
        try:
            bundle = store.bundle(run.dataset_id, run.dataset_version)
        except NotFound:
            bundle = None
        return failure_analysis(run, bundle, store.run_annotations(run_id), metric, threshold)

    @router.put("/runs/{run_id}/failures/{item_id}")
    def tag_failure(run_id: str, item_id: str, body: TagBody) -> dict:
        run = store.get_run(run_id)
        if not any(r.item_id == item_id for r in run.results):
            raise NotFound(f"item {item_id} is not in run {run_id}")
        unknown = [t for t in body.tags if t not in TAXONOMY and not t.startswith("custom:")]
        if unknown:
            raise HTTPException(
                422, f"unknown tag(s): {', '.join(unknown)} (prefix custom: for your own)"
            )
        data = store.run_annotations(run_id)
        from agent_hub.eval.lab.models import now_iso

        data[item_id] = {"tags": body.tags, "note": body.note, "updated_at": now_iso()}
        store.save_run_annotations(run_id, data)
        return data[item_id]

    # --- Learning Path -----------------------------------------------------------------

    @router.get("/learning")
    def learning() -> dict:
        return evaluate(store)

    @router.put("/learning/{mission_id}")
    def learning_progress(mission_id: str, body: ProgressBody) -> dict:
        try:
            return update_progress(store, mission_id, body.done, body.reflection)
        except KeyError as exc:
            raise HTTPException(404, f"unknown mission {mission_id}") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return router
