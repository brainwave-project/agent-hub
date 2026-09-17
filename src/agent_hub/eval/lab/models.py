"""Eval Lab data model — datasets, golden items, experiments, runs.

Three vocabulary distinctions this module makes structural rather than verbal:

* **Corpus vs dataset.** The corpus is what gets indexed (documents). The
  dataset is corpus + golden items (questions with relevance judgements). One
  corpus can back many datasets; an index is a function of the corpus alone,
  which is why editing a question never forces a reindex but editing a document
  always does.
* **Draft vs frozen.** A frozen version is immutable and carries a content
  hash. Every run records the hash it was scored against, so two numbers are
  comparable only when their hashes match. To change a frozen dataset you fork
  a new version; you never edit history.
* **Dev vs holdout.** Items carry a split. Dev is for iteration; holdout is for
  the one confirmatory number. Experiments log every holdout access.

And one distinction that most eval sets leave implicit:

* **Judged vs unjudged.** A document can be judged relevant (a grade), judged
  NOT relevant (`judged_nonrelevant`), or never judged at all. Metrics treat
  unjudged documents as non-relevant. That is an assumption, and pooled judging
  plus the judged@k diagnostic exist to measure how much it matters.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "SLUG",
    "LAB_METRICS",
    "Split",
    "Normalization",
    "CorpusDoc",
    "Evidence",
    "GoldenItem",
    "SplitConfig",
    "IndexedDoc",
    "IndexState",
    "DatasetManifest",
    "DatasetMeta",
    "VariantSpec",
    "Guardrail",
    "SuccessCriteria",
    "HoldoutAccess",
    "Experiment",
    "RankedHit",
    "QueryResult",
    "Run",
    "Candidate",
    "now_iso",
]

#: Ids are used as filenames and as know-hub document titles.
SLUG = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$"

#: Every per-query metric the lab reports. The first six are the harness's
#: `METRIC_NAMES`; graded NDCG is added because the lab supports graded labels.
LAB_METRICS: tuple[str, ...] = (
    "hit_rate",
    "recall@k",
    "precision@k",
    "mrr",
    "ndcg@k",
    "graded_ndcg@k",
    "map",
)

Split = Literal["dev", "holdout"]
Normalization = Literal["off", "conservative", "aggressive"]


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


# --- dataset ---------------------------------------------------------------------


class CorpusDoc(BaseModel):
    doc_id: str = Field(pattern=SLUG)
    lang: str = "en"
    body: str
    source: str = ""  # provenance: where the text came from


class Evidence(BaseModel):
    """A verbatim span inside a relevant document that answers the question.

    Evidence is what makes chunk-level scoring survive rechunking. Chunk ids
    change every time the chunker changes; the sentence that answers the
    question does not. A chunk counts as relevant if it contains a span.
    """

    doc_id: str = Field(pattern=SLUG)
    text: str = Field(min_length=3)


class GoldenItem(BaseModel):
    id: str = Field(pattern=SLUG)
    query: str = Field(min_length=1)
    lang: str = "en"
    #: doc_id -> grade. 3 = fully answers, 2 = substantially, 1 = partially.
    relevant: dict[str, int] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    ground_truth: str = ""
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    trap: str = "none"
    tags: list[str] = Field(default_factory=list)
    split: Split | None = None
    #: A pinned item keeps its split when the split is regenerated.
    pinned: bool = False
    notes: str = ""
    #: Documents explicitly judged NOT relevant (grade 0). A document in neither
    #: this list nor `relevant` is UNJUDGED — metrics assume it is non-relevant.
    judged_nonrelevant: list[str] = Field(default_factory=list)
    #: Provenance of the question: human | synthetic | imported.
    origin: str = "human"

    @field_validator("relevant")
    @classmethod
    def _grades_in_range(cls, value: dict[str, int]) -> dict[str, int]:
        for doc_id, grade in value.items():
            if grade not in (1, 2, 3):
                raise ValueError(f"grade for {doc_id} must be 1, 2 or 3 (got {grade})")
        return value

    @property
    def answerable(self) -> bool:
        return bool(self.relevant)

    @property
    def judged_docs(self) -> set[str]:
        return set(self.relevant) | set(self.judged_nonrelevant)

    @property
    def primary_doc(self) -> str:
        """Highest-graded relevant doc (ties broken alphabetically) — the cluster key."""
        if not self.relevant:
            return f"__unanswerable__:{self.id}"
        return sorted(self.relevant.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


class SplitConfig(BaseModel):
    holdout_fraction: float = Field(default=0.3, ge=0.1, le=0.5)
    seed: int = 20260915
    stratify_by: list[Literal["difficulty", "trap", "lang"]] = Field(
        default_factory=lambda: ["difficulty"]
    )
    #: primary_doc keeps every question about one document on the same side.
    group_by: Literal["none", "primary_doc"] = "primary_doc"


class IndexedDoc(BaseModel):
    document_id: str = ""
    status: str = "PENDING"


class IndexState(BaseModel):
    normalization: Normalization = "off"
    corpus_hash: str
    owner_id: str
    documents: dict[str, IndexedDoc] = Field(default_factory=dict)
    started_at: str = Field(default_factory=now_iso)
    finished_at: str | None = None


class DatasetMeta(BaseModel):
    id: str = Field(pattern=SLUG)
    name: str
    description: str = ""
    lang: str = "en"
    created_at: str = Field(default_factory=now_iso)


class DatasetManifest(BaseModel):
    dataset_id: str
    version: int
    status: Literal["draft", "frozen"] = "draft"
    parent_version: int | None = None
    created_at: str = Field(default_factory=now_iso)
    frozen_at: str | None = None
    content_hash: str | None = None
    split_config: SplitConfig | None = None
    split_assigned_at: str | None = None
    changelog: str = ""


class Candidate(BaseModel):
    """A machine-generated question awaiting human review. Never scored until accepted."""

    id: str
    doc_id: str
    chunk_index: int = 0
    query: str
    evidence: str = ""
    answer: str = ""
    difficulty: str = "medium"
    style: str = "user"
    model: str = ""
    prompt_version: str = ""
    temperature: float = 0.7
    seed: int = 0
    evidence_found: bool = False
    lexical_overlap: float = 0.0
    copied: bool = False
    status: Literal["pending", "accepted", "rejected"] = "pending"
    reject_reason: str = ""
    item_id: str | None = None
    created_at: str = Field(default_factory=now_iso)


# --- experiments -----------------------------------------------------------------


class VariantSpec(BaseModel):
    """One complete retrieval configuration — the thing an experiment varies."""

    name: str = "baseline"
    level: Literal["document", "chunk"] = "document"
    #: Document level only. The gRPC chunk path is vector-only by contract.
    hybrid: bool = True
    #: Which index copy to search: the corpus as written, or normalised at ingestion.
    corpus_index: Normalization = "off"
    query_normalization: Normalization = "off"
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    fetch: int = Field(default=50, ge=1, le=50)


class Guardrail(BaseModel):
    metric: str
    #: Largest tolerated drop in the metric (CI lower bound must stay above -max_drop).
    max_drop: float = Field(default=0.05, ge=0.0, le=1.0)


class SuccessCriteria(BaseModel):
    primary_metric: str = "recall@k"
    k: int = Field(default=5, ge=1, le=50)
    direction: Literal["increase", "decrease", "non_inferior"] = "increase"
    #: Smallest effect worth acting on. Doubles as the non-inferiority margin.
    min_effect: float = Field(default=0.05, gt=0.0, le=1.0)
    confidence: float = Field(default=0.95, ge=0.5, le=0.999)
    guardrails: list[Guardrail] = Field(default_factory=list)


class HoldoutAccess(BaseModel):
    at: str = Field(default_factory=now_iso)
    reason: str = ""
    run_ids: list[str] = Field(default_factory=list)


class Experiment(BaseModel):
    id: str = Field(pattern=SLUG)
    name: str
    hypothesis: str = ""
    rationale: str = ""
    dataset_id: str
    dataset_version: int
    dataset_hash: str | None = None
    baseline: VariantSpec = Field(default_factory=lambda: VariantSpec(name="baseline"))
    treatment: VariantSpec = Field(default_factory=lambda: VariantSpec(name="treatment"))
    criteria: SuccessCriteria = Field(default_factory=SuccessCriteria)
    secondary_metrics: list[str] = Field(default_factory=list)
    status: Literal["draft", "registered", "concluded"] = "draft"
    created_at: str = Field(default_factory=now_iso)
    registered_at: str | None = None
    concluded_at: str | None = None
    #: Every run ever made for this experiment, in order. Never pruned: the
    #: report lists all of them, which is the structural guard against
    #: selective reporting.
    runs: list[str] = Field(default_factory=list)
    holdout_access: list[HoldoutAccess] = Field(default_factory=list)
    conclusion: dict | None = None
    conclusion_note: str = ""


# --- runs ------------------------------------------------------------------------


class RankedHit(BaseModel):
    rank: int
    unit_id: str  # the id the metrics actually saw
    doc_id: str
    document_id: str = ""
    chunk_id: str = ""
    score: float = 0.0
    rrf_score: float = 0.0
    excerpt: str = ""
    grade: int = 0  # 0 = not relevant
    novel: bool = True  # False: relevant, but its evidence was already credited


class QueryResult(BaseModel):
    item_id: str
    query: str
    query_sent: str
    ranked: list[RankedHit] = Field(default_factory=list)
    relevant_units: dict[str, int] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    #: Share of the top-k results whose document carries ANY judgement. The rest
    #: were scored as non-relevant by assumption.
    judged_at_k: float | None = None
    latency_ms: float = 0.0
    error: str | None = None


class Run(BaseModel):
    id: str
    experiment_id: str | None = None
    arm: Literal["baseline", "treatment", "adhoc"] = "adhoc"
    dataset_id: str
    dataset_version: int
    dataset_hash: str
    dataset_status: Literal["draft", "frozen"]
    corpus_hash: str
    variant: VariantSpec
    split: Literal["dev", "holdout", "all"]
    k: int
    status: Literal["running", "done", "failed"] = "running"
    created_at: str = Field(default_factory=now_iso)
    finished_at: str | None = None
    results: list[QueryResult] = Field(default_factory=list)
    skipped_unanswerable: list[str] = Field(default_factory=list)
    aggregates: dict[str, dict[str, float]] = Field(default_factory=dict)
    #: Run-level diagnostics that are NOT quality metrics: judged@k, latency, empties.
    diagnostics: dict[str, float] = Field(default_factory=dict)
    error: str | None = None
