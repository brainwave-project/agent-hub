"""Data model for the generation half of the Eval Lab — pipeline runs, judges, gates.

Kept apart from `models.py` so the retrieval half's schema (and the content
hashes built from it) never moves when this half changes.

Three distinctions this module makes structural:

* **A stage trace, not only a score.** A generation run stores what every
  stage saw and did — candidates, kept context, grade, answer — so a failure
  can be attributed to the stage that caused it (`blame`), not only counted.
* **A judge is a versioned artefact.** Rubric, model, temperature and mode are
  hashed into a fingerprint. Two scores are comparable only when their judges'
  fingerprints match; edit a rubric and you get a new version, not a silently
  different judge.
* **A gate is a written policy.** What blocks a release is data you can review
  in a diff, evaluated the same way in the UI, the CLI and CI.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field

from agent_hub.eval.lab.models import SLUG, now_iso

__all__ = [
    "STAGES",
    "FAULTS",
    "PipelineSpec",
    "CtxChunk",
    "StageResult",
    "GenItemResult",
    "GenRun",
    "JudgeSpec",
    "HumanLabel",
    "JudgeSample",
    "JudgeSet",
    "JudgeRun",
    "GateRule",
    "GatePolicy",
]

#: Pipeline stages in execution order. Blame is the FIRST stage that failed.
STAGES: tuple[str, ...] = ("ingestion", "retrieval", "rerank", "grade", "generation")

#: Fault injection: each breaks exactly one stage, on purpose, so you can check
#: that component attribution blames the stage you broke.
FAULTS: dict[str, str] = {
    "none": "No fault. The pipeline as configured.",
    "ingestion": "Simulated: chunks holding the evidence were never indexed (lost at ingestion).",
    "retrieval": "Evidence chunks are in the index but removed from the retrieved candidates.",
    "rerank": "The reranked list is reversed before the top-n cut, so the best chunks fall off.",
    "grade": "The relevance grader rejects every context, so CRAG abstains.",
    "generation": "The generator gets no context and a 'always answer' prompt, so it guesses.",
}


class PipelineSpec(BaseModel):
    """One complete RAG configuration — what a generation run varies."""

    name: str = "baseline"
    #: Stage 1: chunks fetched from know-hub (gRPC SemanticSearch, the CRAG path).
    candidate_k: int = Field(default=10, ge=1, le=50)
    min_score: float = Field(default=0.0, ge=0.0, le=0.99)
    #: Stage 2: reorder the candidates, then keep top_n as the context.
    rerank: Literal["none", "lexical", "llm"] = "none"
    top_n: int = Field(default=4, ge=1, le=20)
    #: CRAG's relevance grader (whole-context, like the service). Off = always generate.
    grader: bool = True
    #: Generator prompt: strict (CRAG's), lenient (hallucination-prone), verbose.
    generator: Literal["strict", "lenient", "verbose"] = "strict"
    generate_temperature: float = Field(default=0.0, ge=0.0, le=1.5)
    #: CRAG's post-generation grounding check with one regeneration.
    grounding: bool = False
    fault: Literal["none", "ingestion", "retrieval", "rerank", "grade", "generation"] = "none"
    model: str | None = None  # generator model; None = the lab default


class CtxChunk(BaseModel):
    rank: int
    doc_id: str
    chunk_id: str = ""
    score: float = 0.0
    text: str
    #: Indexes into the item's evidence units that this chunk contains.
    evidence_units: list[int] = Field(default_factory=list)


class StageResult(BaseModel):
    stage: str
    status: Literal["ok", "warn", "fail", "skip"]
    detail: str = ""
    data: dict = Field(default_factory=dict)


class GenItemResult(BaseModel):
    item_id: str
    query: str
    answerable: bool
    ground_truth: str = ""
    trap: str = "none"
    difficulty: str = "medium"
    units: list[str] = Field(default_factory=list)  # human-readable evidence units
    candidates: list[CtxChunk] = Field(default_factory=list)
    contexts: list[CtxChunk] = Field(default_factory=list)
    grade: dict | None = None
    answer: str = ""
    abstained: bool = False
    grounding: dict | None = None
    #: Per-item metric values. Binary outcome metrics (answer_ok, false_abstention,
    #: answered_unanswerable) are here too, so rates get paired CIs like any metric.
    metrics: dict[str, float] = Field(default_factory=dict)
    details: dict[str, dict] = Field(default_factory=dict)
    stages: list[StageResult] = Field(default_factory=list)
    blame: str = "none"
    outcome: str = ""
    flags: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None


class GenRun(BaseModel):
    kind: Literal["generation"] = "generation"
    id: str
    dataset_id: str
    dataset_version: int
    dataset_hash: str
    dataset_status: Literal["draft", "frozen"]
    corpus_hash: str
    spec: PipelineSpec
    split: Literal["dev", "holdout", "all"]
    item_ids: list[str] = Field(default_factory=list)
    llm_context_metrics: bool = True
    judge: dict = Field(default_factory=dict)
    judge_fingerprint: str = ""
    status: Literal["running", "done", "failed"] = "running"
    created_at: str = Field(default_factory=now_iso)
    finished_at: str | None = None
    results: list[GenItemResult] = Field(default_factory=list)
    aggregates: dict[str, dict[str, float]] = Field(default_factory=dict)
    components: dict = Field(default_factory=dict)
    diagnostics: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


# --- LLM-as-judge ------------------------------------------------------------------------


class JudgeSpec(BaseModel):
    """A judge = rubric + model + decoding + output contract. Versioned; hashed."""

    id: str = Field(pattern=SLUG)
    name: str
    version: int = 1
    model: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=1.5)
    seed: int = 7
    #: pointwise: a 1..scale_max score; binary: pass/fail only.
    mode: Literal["pointwise", "binary"] = "pointwise"
    rubric: str = Field(min_length=10)
    #: Show the judge the reference answer. Without it, it can only judge style.
    reference_guided: bool = True
    #: Show the judge the retrieved context.
    show_context: bool = False
    #: Ask for the reasoning BEFORE the score (a cheap chain-of-thought).
    reasoning_first: bool = True
    scale_max: int = Field(default=5, ge=2, le=10)
    #: Pointwise scores >= this count as "pass".
    pass_threshold: int = Field(default=4, ge=1, le=10)
    notes: str = ""
    created_at: str = Field(default_factory=now_iso)
    history: list[dict] = Field(default_factory=list)

    def contract(self) -> dict:
        return self.model_dump(exclude={"id", "name", "version", "notes", "created_at", "history"})

    @property
    def fingerprint(self) -> str:
        from agent_hub.eval.lab.llm import default_model

        payload = {**self.contract(), "model": self.model or default_model()}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class HumanLabel(BaseModel):
    verdict: Literal["pass", "fail"]
    score: int | None = Field(default=None, ge=1, le=10)
    note: str = ""
    at: str = Field(default_factory=now_iso)


class JudgeSample(BaseModel):
    """One (question, answer) to judge. `kind` is the designer's tag, hidden while labelling."""

    id: str = Field(pattern=SLUG)
    question: str
    answer: str
    reference: str = ""
    contexts: list[str] = Field(default_factory=list)
    kind: str = "unknown"
    source: str = "imported"
    labels: dict[str, HumanLabel] = Field(default_factory=dict)


class JudgeSet(BaseModel):
    id: str = Field(pattern=SLUG)
    name: str
    description: str = ""
    created_at: str = Field(default_factory=now_iso)
    samples: list[JudgeSample] = Field(default_factory=list)


class JudgeRun(BaseModel):
    """One judge applied to one set. `probe` says which experiment produced it."""

    id: str
    set_id: str
    judge_id: str
    judge_version: int
    fingerprint: str
    spec: dict
    probe: Literal["standard", "consistency", "verbosity", "position"] = "standard"
    repeats: int = 1
    status: Literal["running", "done", "failed"] = "running"
    created_at: str = Field(default_factory=now_iso)
    finished_at: str | None = None
    results: list[dict] = Field(default_factory=list)
    summary: dict = Field(default_factory=dict)
    error: str | None = None


# --- release gate ------------------------------------------------------------------------


class GateRule(BaseModel):
    id: str = Field(pattern=SLUG)
    metric: str
    #: floor: candidate mean >= threshold · ceiling: candidate mean <= threshold ·
    #: max_drop: candidate − baseline >= −threshold (point estimates) ·
    #: non_inferior: lower CI bound of the paired delta >= −threshold.
    kind: Literal["floor", "ceiling", "max_drop", "non_inferior"]
    threshold: float = Field(ge=0.0, le=1.0)
    severity: Literal["block", "warn"] = "block"
    applies_to: Literal["retrieval", "generation"] = "generation"
    rationale: str = ""


class GatePolicy(BaseModel):
    id: str = Field(pattern=SLUG)
    name: str
    description: str = ""
    version: int = 1
    confidence: float = Field(default=0.95, ge=0.5, le=0.999)
    #: Preconditions. Failing one is a BLOCK: the gate cannot certify what it cannot compare.
    require_same_dataset: bool = True
    require_frozen_dataset: bool = True
    require_same_judge: bool = True
    min_items: int = Field(default=20, ge=1)
    max_error_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    rules: list[GateRule] = Field(default_factory=list)
    updated_at: str = Field(default_factory=now_iso)
