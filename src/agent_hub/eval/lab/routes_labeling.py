"""HTTP routes for labelling work: pooled judging, label audit, synthetic generation, diffs.

Mounted by `api.create_app`. Everything here writes either golden items (drafts
only — the store enforces immutability) or per-version artefacts, which stay
writable on frozen versions because auditing a set does not change the set.
"""

from __future__ import annotations

import os
import threading
from collections import Counter
from collections.abc import Callable
from typing import Literal

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field

from agent_hub.eval.lab.agreement import GOLD, agreement_report, build_task, simulate_annotations
from agent_hub.eval.lab.chunking import chunk_text, locate_evidence
from agent_hub.eval.lab.diff import diff_versions
from agent_hub.eval.lab.generate import OllamaGenerator, generate_candidates
from agent_hub.eval.lab.jobs import JobRegistry
from agent_hub.eval.lab.models import Evidence, GoldenItem, VariantSpec, now_iso
from agent_hub.eval.lab.pooling import build_pool, pool_view, simulate_pool_judgements
from agent_hub.eval.lab.runner import KnowHubLab
from agent_hub.eval.lab.store import Bundle, Conflict, LabStore, NotFound

__all__ = ["build_router"]

BASE = "/datasets/{dataset_id}/versions/{version}"
DEFAULT_GEN_MODEL = "llama3.1:8b-instruct-q4_K_M"


class PoolBody(BaseModel):
    item_ids: list[str] | None = None
    variants: list[VariantSpec] | None = None
    depth: int = Field(default=10, ge=3, le=30)


class JudgeBody(BaseModel):
    item_id: str
    doc_id: str
    #: 0 = judged not relevant, 1-3 = relevant grade. `clear` removes any judgement.
    grade: int = Field(default=0, ge=0, le=3)
    clear: bool = False


class AnnotationBody(BaseModel):
    grades: dict[str, int]
    seconds: float = Field(default=0.0, ge=0.0)


class SimulateJudgeBody(BaseModel):
    item_id: str | None = None


class SimulateAuditBody(BaseModel):
    annotator: str = Field(min_length=1, max_length=40)
    sample: int = Field(default=10, ge=1, le=50)
    agreement: float = Field(default=0.8, ge=0.0, le=1.0)
    seed: int = 7


class RenameAnnotatorBody(BaseModel):
    name: str = Field(min_length=1, max_length=40)


class GenerateBody(BaseModel):
    doc_ids: list[str] | None = None
    per_doc: int = Field(default=2, ge=1, le=5)
    style: Literal["user", "keyword", "hard"] = "user"
    temperature: float = Field(default=0.7, ge=0.0, le=1.5)
    seed: int = 7
    model: str | None = None


class AcceptBody(BaseModel):
    query: str | None = None
    evidence: str | None = None
    difficulty: Literal["easy", "medium", "hard"] | None = None
    split: Literal["dev", "holdout"] | None = None


class RejectBody(BaseModel):
    reason: str = Field(min_length=3, max_length=200)


def _candidates_view(candidates: list[dict]) -> dict:
    total = len(candidates)
    reviewed = [c for c in candidates if c["status"] != "pending"]
    accepted = [c for c in candidates if c["status"] == "accepted"]
    return {
        "candidates": sorted(
            candidates, key=lambda c: (c["status"] != "pending", c["doc_id"], c["id"])
        ),
        "summary": {
            "total": total,
            "pending": total - len(reviewed),
            "accepted": len(accepted),
            "rejected": len(reviewed) - len(accepted),
            "acceptance_rate": len(accepted) / len(reviewed) if reviewed else None,
            "evidence_found_rate": sum(c["evidence_found"] for c in candidates) / total
            if total
            else None,
            "mean_overlap": sum(c["lexical_overlap"] for c in candidates) / total
            if total
            else None,
            "copied": sum(c["copied"] for c in candidates),
            "reject_reasons": dict(
                Counter(c["reject_reason"] for c in candidates if c["status"] == "rejected")
            ),
        },
    }


def build_router(
    store: LabStore, kh: KnowHubLab, jobs: JobRegistry, bundle_json: Callable[[Bundle], dict]
) -> APIRouter:
    router = APIRouter(prefix="/api/eval")
    artifact_lock = threading.Lock()

    # --- pooled judging ----------------------------------------------------------

    @router.post(BASE + "/pool")
    def start_pool(dataset_id: str, version: int, body: PoolBody) -> dict:
        bundle = store.bundle(dataset_id, version)

        def work(job) -> dict:
            pool = build_pool(kh, bundle, body.item_ids, body.variants, body.depth, job)
            store.save_artifact(dataset_id, version, "pool", pool)
            return pool_view(bundle, pool)["summary"]

        return jobs.submit(
            "pool", f"pool {dataset_id} v{version} (depth {body.depth})", work
        ).as_dict()

    @router.get(BASE + "/pool")
    def get_pool(dataset_id: str, version: int) -> dict:
        bundle = store.bundle(dataset_id, version)
        return pool_view(bundle, store.artifact(dataset_id, version, "pool"))

    @router.post(BASE + "/judge")
    def judge(dataset_id: str, version: int, body: JudgeBody) -> dict:
        bundle = store.bundle(dataset_id, version)
        item = bundle.item(body.item_id)
        if item is None:
            raise NotFound(f"item {body.item_id}")
        if body.doc_id not in bundle.corpus:
            raise NotFound(f"document {body.doc_id}")
        relevant = {d: g for d, g in item.relevant.items() if d != body.doc_id}
        nonrelevant = [d for d in item.judged_nonrelevant if d != body.doc_id]
        evidence = item.evidence
        if not body.clear:
            if body.grade == 0:
                nonrelevant.append(body.doc_id)
            else:
                relevant[body.doc_id] = body.grade
        if body.doc_id not in relevant:
            evidence = [e for e in evidence if e.doc_id != body.doc_id]
        updated = item.model_copy(
            update={
                "relevant": relevant,
                "judged_nonrelevant": sorted(nonrelevant),
                "evidence": evidence,
            }
        )
        store.put_item(dataset_id, version, GoldenItem.model_validate(updated.model_dump()))
        return bundle_json(store.bundle(dataset_id, version))

    @router.post(BASE + "/judge/simulate")
    def simulate_judge(
        dataset_id: str, version: int, body: SimulateJudgeBody = Body(default_factory=SimulateJudgeBody)
    ) -> dict:
        bundle = store.bundle(dataset_id, version)
        try:
            items, _filled = simulate_pool_judgements(
                bundle, store.artifact(dataset_id, version, "pool"), body.item_id
            )
        except ValueError as exc:
            raise Conflict(str(exc)) from exc
        store.replace_items(dataset_id, version, items)
        return bundle_json(store.bundle(dataset_id, version))

    # --- label audit -------------------------------------------------------------

    @router.get(BASE + "/annotations")
    def annotations(dataset_id: str, version: int) -> dict:
        store.manifest(dataset_id, version)
        ann = store.artifact(dataset_id, version, "annotations", {}) or {}
        return _annotation_index(ann)

    @router.patch(BASE + "/annotations/{annotator}")
    def rename_annotator(
        dataset_id: str, version: int, annotator: str, body: RenameAnnotatorBody
    ) -> dict:
        new = body.name.strip()
        if annotator == GOLD or new == GOLD:
            raise HTTPException(422, "'gold' is reserved for the dataset's own labels")
        if not new:
            raise HTTPException(422, "name is empty")
        store.manifest(dataset_id, version)
        with artifact_lock:
            ann = store.artifact(dataset_id, version, "annotations", {}) or {}
            buckets = ann.setdefault("annotators", {})
            if annotator not in buckets:
                raise NotFound(f"annotator {annotator}")
            if new != annotator and new in buckets:
                raise Conflict(f"annotator '{new}' already exists")
            if new != annotator:
                buckets[new] = buckets.pop(annotator)
                simulated = ann.setdefault("simulated", {})
                if annotator in simulated:
                    simulated[new] = simulated.pop(annotator)
                store.save_artifact(dataset_id, version, "annotations", ann)
            return {**_annotation_index(ann), "annotator": new}

    @router.post(BASE + "/annotations/simulate")
    def simulate_audit(dataset_id: str, version: int, body: SimulateAuditBody) -> dict:
        if body.annotator == GOLD:
            raise HTTPException(422, "'gold' is reserved for the dataset's own labels")
        bundle = store.bundle(dataset_id, version)
        with artifact_lock:
            ann = store.artifact(dataset_id, version, "annotations", {}) or {}
            pool = store.artifact(dataset_id, version, "pool")
            updated, n = simulate_annotations(
                bundle, ann, body.annotator, body.sample, body.agreement, body.seed, pool
            )
            if n == 0:
                raise Conflict(f"no remaining items for {body.annotator}")
            store.save_artifact(dataset_id, version, "annotations", updated)
        return {"annotator": body.annotator, "annotated": n, "simulated": True}

    @router.get(BASE + "/annotations/task")
    def annotation_task(
        dataset_id: str,
        version: int,
        annotator: str = Query(min_length=1, max_length=40),
        sample: int = Query(10, ge=1, le=50),
        seed: int = Query(7),
    ) -> dict:
        if annotator == GOLD:
            raise HTTPException(422, "'gold' is reserved for the dataset's own labels")
        bundle = store.bundle(dataset_id, version)
        ann = store.artifact(dataset_id, version, "annotations", {}) or {}
        pool = store.artifact(dataset_id, version, "pool")
        return build_task(bundle, ann, annotator, sample, seed, pool)

    @router.put(BASE + "/annotations/{annotator}/{item_id}")
    def annotate(
        dataset_id: str, version: int, annotator: str, item_id: str, body: AnnotationBody
    ) -> dict:
        if annotator == GOLD:
            raise HTTPException(422, "'gold' is reserved for the dataset's own labels")
        bundle = store.bundle(dataset_id, version)
        if bundle.item(item_id) is None:
            raise NotFound(f"item {item_id}")
        bad = [d for d, g in body.grades.items() if g not in (0, 1, 2, 3) or d not in bundle.corpus]
        if bad:
            raise HTTPException(422, f"invalid grades for: {', '.join(bad)}")
        with artifact_lock:
            ann = store.artifact(dataset_id, version, "annotations", {}) or {}
            ann.setdefault("annotators", {}).setdefault(annotator, {})[item_id] = {
                "grades": body.grades,
                "seconds": body.seconds,
                "created_at": now_iso(),
            }
            store.save_artifact(dataset_id, version, "annotations", ann)
        return {"annotator": annotator, "annotated": len(ann["annotators"][annotator])}

    @router.get(BASE + "/agreement")
    def agreement(
        dataset_id: str, version: int, a: str = GOLD, b: str = Query(min_length=1)
    ) -> dict:
        bundle = store.bundle(dataset_id, version)
        ann = store.artifact(dataset_id, version, "annotations", {}) or {}
        try:
            return agreement_report(bundle, ann, a, b)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    # --- synthetic generation ------------------------------------------------------

    @router.post(BASE + "/generate")
    def generate(dataset_id: str, version: int, body: GenerateBody) -> dict:
        bundle = store.bundle(dataset_id, version)
        if not bundle.corpus:
            raise Conflict("corpus is empty")
        generator = OllamaGenerator(
            os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434"),
            body.model or os.environ.get("EVAL_LAB_GEN_MODEL", DEFAULT_GEN_MODEL),
        )

        def work(job) -> dict:
            existing = store.artifact(dataset_id, version, "candidates", []) or []
            out = generate_candidates(
                generator,
                bundle,
                body.doc_ids,
                body.per_doc,
                body.style,
                body.temperature,
                body.seed,
                existing,
                job,
            )
            with artifact_lock:
                current = store.artifact(dataset_id, version, "candidates", []) or []
                known = {c["id"] for c in current}
                store.save_artifact(
                    dataset_id,
                    version,
                    "candidates",
                    current + [c for c in out["created"] if c["id"] not in known],
                )
            if not out["created"] and out["failures"]:
                raise RuntimeError("generation failed for every chunk: " + out["failures"][0])
            return {"created": len(out["created"]), "failures": out["failures"][:10]}

        return jobs.submit(
            "generate", f"generate questions · {dataset_id} v{version}", work
        ).as_dict()

    @router.get(BASE + "/candidates")
    def candidates(dataset_id: str, version: int) -> dict:
        store.manifest(dataset_id, version)
        return _candidates_view(store.artifact(dataset_id, version, "candidates", []) or [])

    @router.post(BASE + "/candidates/{candidate_id}/accept")
    def accept(dataset_id: str, version: int, candidate_id: str, body: AcceptBody) -> dict:
        with artifact_lock:
            cands = store.artifact(dataset_id, version, "candidates", []) or []
            cand = next((c for c in cands if c["id"] == candidate_id), None)
            if cand is None:
                raise NotFound(f"candidate {candidate_id}")
            if cand["status"] != "pending":
                raise Conflict(f"candidate is already {cand['status']}")
            bundle = store.bundle(dataset_id, version)
            doc = bundle.corpus.get(cand["doc_id"])
            if doc is None:
                raise NotFound(f"document {cand['doc_id']} no longer in corpus")
            query = (body.query or cand["query"]).strip()
            evidence_text = (
                body.evidence if body.evidence is not None else cand["evidence"]
            ).strip()
            evidence = []
            if (
                len(evidence_text) >= 3
                and locate_evidence(doc.body, evidence_text, chunk_text(doc.body))["found_in_doc"]
            ):
                evidence = [Evidence(doc_id=doc.doc_id, text=evidence_text)]
            n = 1
            existing_ids = {i.id for i in bundle.items}
            while f"syn-{n:03d}" in existing_ids:
                n += 1
            item = GoldenItem(
                id=f"syn-{n:03d}",
                query=query,
                lang=doc.lang,
                relevant={doc.doc_id: 3},
                evidence=evidence,
                ground_truth=cand.get("answer", ""),
                difficulty=body.difficulty
                or (
                    cand["difficulty"]
                    if cand["difficulty"] in ("easy", "medium", "hard")
                    else "medium"
                ),
                tags=["synthetic", cand["style"]],
                split=body.split,
                pinned=body.split is not None,
                origin="synthetic",
                notes=f"generated by {cand['model']} ({cand['prompt_version']}, T={cand['temperature']}, "
                f"seed={cand['seed']}); edited={query != cand['query']}",
            )
            store.put_item(dataset_id, version, item)
            cand.update({"status": "accepted", "item_id": item.id, "query": query})
            store.save_artifact(dataset_id, version, "candidates", cands)
        return bundle_json(store.bundle(dataset_id, version))

    @router.post(BASE + "/candidates/{candidate_id}/reject")
    def reject(dataset_id: str, version: int, candidate_id: str, body: RejectBody) -> dict:
        with artifact_lock:
            cands = store.artifact(dataset_id, version, "candidates", []) or []
            cand = next((c for c in cands if c["id"] == candidate_id), None)
            if cand is None:
                raise NotFound(f"candidate {candidate_id}")
            if cand["status"] != "pending":
                raise Conflict(f"candidate is already {cand['status']}")
            cand.update({"status": "rejected", "reject_reason": body.reason})
            store.save_artifact(dataset_id, version, "candidates", cands)
        return _candidates_view(cands)

    # --- version diff ---------------------------------------------------------------

    @router.get("/datasets/{dataset_id}/diff")
    def diff(
        dataset_id: str,
        from_version: int = Query(alias="from", ge=1),
        to_version: int = Query(alias="to", ge=1),
    ) -> dict:
        return diff_versions(
            store.bundle(dataset_id, from_version), store.bundle(dataset_id, to_version)
        )

    return router


def _annotation_index(ann: dict) -> dict:
    counts = {name: len(entries) for name, entries in ann.get("annotators", {}).items()}
    return {"annotators": counts, "raters": [GOLD, *sorted(counts)]}
