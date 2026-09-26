"""HTTP routes for the release gate — policies, evaluation, recorded decisions, CI export.

The UI's decision and the CLI's exit code come from the same `evaluate_gate`
call. `POST /gate/export` writes the policy and the two runs into
`eval/release/` inside the repository (the eval-api container mounts it), which
is exactly what `.github/workflows/eval-gate.yml` gates on — so "what will CI
say?" can be answered before you push.
"""

from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agent_hub.eval.datasets import EVAL_DIR
from agent_hub.eval.lab.genmodels import GatePolicy
from agent_hub.eval.lab.models import now_iso
from agent_hub.eval.lab.release_gate import default_policy, evaluate_gate, render_markdown
from agent_hub.eval.lab.store import LabStore, NotFound

__all__ = ["build_router", "RELEASE_DIR"]

RELEASE_DIR = EVAL_DIR / "release"


class EvaluateBody(BaseModel):
    policy_id: str
    candidate_run: str
    baseline_run: str | None = None


class DecisionBody(EvaluateBody):
    override_reason: str = ""
    author: str = ""


class BaselineBody(BaseModel):
    run_id: str
    note: str = ""


def build_router(store: LabStore) -> APIRouter:
    router = APIRouter(prefix="/api/eval/gate")

    def load_run(run_id: str) -> dict:
        for kind in ("gen_runs", None):
            try:
                if kind:
                    return store.get_doc_json(kind, run_id)
                return store.get_run(run_id).model_dump()
            except NotFound:
                continue
        raise NotFound(f"run {run_id}")

    def get_policy(policy_id: str) -> GatePolicy:
        return GatePolicy.model_validate(store.get_doc_json("gate_policies", policy_id))

    def ensure_default() -> None:
        if not store.list_doc_json("gate_policies"):
            policy = default_policy()
            store.put_doc_json("gate_policies", policy.id, policy.model_dump())

    @router.get("/policies")
    def list_policies() -> list[dict]:
        ensure_default()
        return store.list_doc_json("gate_policies")

    @router.get("/policies/{policy_id}")
    def read_policy(policy_id: str) -> dict:
        ensure_default()
        return get_policy(policy_id).model_dump()

    @router.put("/policies/{policy_id}")
    def save_policy(policy_id: str, policy: GatePolicy) -> dict:
        if policy.id != policy_id:
            raise HTTPException(422, "policy id in body must match the path")
        try:
            current = get_policy(policy_id)
            same = current.model_dump(exclude={"version", "updated_at"}) == policy.model_dump(
                exclude={"version", "updated_at"}
            )
            policy.version = current.version if same else current.version + 1
        except NotFound:
            policy.version = 1
        policy.updated_at = now_iso()
        store.put_doc_json("gate_policies", policy_id, policy.model_dump())
        return policy.model_dump()

    @router.post("/policies/{policy_id}/reset")
    def reset_policy(policy_id: str) -> dict:
        policy = default_policy()
        policy.id = policy_id
        try:
            policy.version = get_policy(policy_id).version + 1
        except NotFound:
            pass
        store.put_doc_json("gate_policies", policy_id, policy.model_dump())
        return policy.model_dump()

    @router.get("/runs")
    def runs(dataset_id: str | None = None) -> list[dict]:
        """Both kinds of run in one list, for the gate's pickers."""
        out = []
        for r in store.list_runs(dataset_id):
            out.append(
                {
                    "id": r["id"],
                    "kind": "retrieval",
                    "name": r["variant"]["name"],
                    "split": r["split"],
                    "status": r["status"],
                    "dataset_id": r["dataset_id"],
                    "dataset_version": r["dataset_version"],
                    "created_at": r["created_at"],
                    "arm": r.get("arm"),
                }
            )
        for r in store.list_gen_runs(dataset_id):
            out.append(
                {
                    "id": r["id"],
                    "kind": "generation",
                    "name": r["spec"]["name"],
                    "split": r["split"],
                    "status": r["status"],
                    "dataset_id": r["dataset_id"],
                    "dataset_version": r["dataset_version"],
                    "created_at": r["created_at"],
                    "fault": r["spec"].get("fault", "none"),
                }
            )
        return sorted(out, key=lambda r: r["created_at"], reverse=True)

    @router.get("/baselines")
    def baselines() -> dict:
        return store.gate_baselines()

    @router.put("/baselines/{policy_id}")
    def promote(policy_id: str, body: BaselineBody) -> dict:
        run = load_run(body.run_id)
        kind = "generation" if run.get("kind") == "generation" else "retrieval"
        data = store.gate_baselines()
        data.setdefault(policy_id, {})[kind] = {
            "run_id": body.run_id,
            "note": body.note,
            "promoted_at": now_iso(),
        }
        store.save_gate_baselines(data)
        return data

    def resolve(body: EvaluateBody) -> tuple[GatePolicy, dict, dict]:
        policy = get_policy(body.policy_id)
        candidate = load_run(body.candidate_run)
        baseline_id = body.baseline_run
        if not baseline_id:
            kind = "generation" if candidate.get("kind") == "generation" else "retrieval"
            baseline_id = store.gate_baselines().get(body.policy_id, {}).get(kind, {}).get("run_id")
            if not baseline_id:
                raise HTTPException(
                    409,
                    {
                        "code": "no_baseline",
                        "detail": f"No promoted {kind} baseline for {body.policy_id}. "
                        "Pick a baseline run or promote one.",
                    },
                )
        return policy, load_run(baseline_id), candidate

    @router.post("/evaluate")
    def evaluate(body: EvaluateBody) -> dict:
        policy, baseline, candidate = resolve(body)
        result = evaluate_gate(policy, baseline, candidate)
        result["markdown"] = render_markdown(result)
        return result

    @router.post("/decisions")
    def record(body: DecisionBody) -> dict:
        policy, baseline, candidate = resolve(body)
        result = evaluate_gate(policy, baseline, candidate)
        override = None
        final = result["decision"]
        if body.override_reason.strip():
            if result["decision"] != "block":
                raise HTTPException(422, "only a BLOCK can be overridden")
            if len(body.override_reason.strip()) < 30:
                raise HTTPException(422, "an override needs a real reason (at least 30 characters)")
            override = {
                "reason": body.override_reason.strip(),
                "author": body.author or "unknown",
                "at": now_iso(),
            }
            final = "pass_overridden"
        decision = {
            "id": f"gd_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}",
            "recorded_at": now_iso(),
            "policy_id": policy.id,
            "policy_version": policy.version,
            "baseline_run": result["baseline"]["id"],
            "candidate_run": result["candidate"]["id"],
            "decision": result["decision"],
            "final": final,
            "override": override,
            "reasons": result["reasons"],
            "result": result,
        }
        store.put_doc_json("gate_decisions", decision["id"], decision)
        return decision

    @router.get("/decisions")
    def decisions() -> list[dict]:
        rows = store.list_doc_json("gate_decisions")
        for r in rows:
            r.pop("result", None)
        return sorted(rows, key=lambda r: r["recorded_at"], reverse=True)

    def slim(run: dict) -> dict:
        """What the gate reads, plus what a reviewer wants to see — not the full trace.

        Retrieved chunk texts and metric working make a run ~1 MB; the gate needs
        only per-item metrics, outcomes and errors.
        """
        if run.get("kind") != "generation":
            return {**run, "results": [{**r, "ranked": r.get("ranked", [])[:10]} for r in run.get("results", [])]}
        keep = ("item_id", "query", "answerable", "trap", "difficulty", "answer", "abstained",
                "metrics", "blame", "outcome", "flags", "error")
        return {**run, "results": [{k: r.get(k) for k in keep} for r in run.get("results", [])]}

    @router.post("/export")
    def export(body: EvaluateBody) -> dict:
        policy, baseline, candidate = resolve(body)
        baseline, candidate = slim(baseline), slim(candidate)
        RELEASE_DIR.mkdir(parents=True, exist_ok=True)
        for name, data in (
            ("policy", policy.model_dump()),
            ("baseline", baseline),
            ("candidate", candidate),
        ):
            (RELEASE_DIR / f"{name}.json").write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return snapshot()

    @router.get("/export")
    def snapshot() -> dict:
        files = {n: RELEASE_DIR / f"{n}.json" for n in ("policy", "baseline", "candidate")}
        present = {n: p.exists() for n, p in files.items()}
        out: dict = {"dir": str(RELEASE_DIR), "present": present}
        if all(present.values()):
            data = {n: json.loads(p.read_text(encoding="utf-8")) for n, p in files.items()}
            result = evaluate_gate(
                GatePolicy.model_validate(data["policy"]), data["baseline"], data["candidate"]
            )
            out["result"] = {
                k: result[k]
                for k in ("decision", "exit_code", "reasons", "baseline", "candidate", "policy")
            }
            out["markdown"] = render_markdown(result)
        return out

    return router
