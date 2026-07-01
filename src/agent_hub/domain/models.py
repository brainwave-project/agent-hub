"""Pure domain types — no framework or I/O dependencies (hexagonal core).

Structured-output schemas (Grade, Groundedness, DocClass) are the contracts the
reasoning LLM must satisfy; nodes depend on these, never on a raw model client.
"""

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

# --- runtime governance (§13.5) ------------------------------------------------


class GovernanceSettings(BaseModel):
    """Operator-tunable knobs for the whole reasoning path.

    These are the SINGLE source of truth CRAG reads per run. Field bounds are
    enforced here so an out-of-range value from the API is rejected before it can
    reach a graph. Defaults are supplied by the config layer (``from_settings``),
    never hardcoded twice.
    """

    model_config = {"extra": "forbid"}

    # Retrieval
    top_k: int = Field(ge=1, le=50, description="Chunks fetched per retrieve (stage 1).")
    min_score: float = Field(ge=0.0, le=0.99, description="Similarity floor to qualify.")
    cross_encoder_enabled: bool = Field(description="Enable stage-2 cross-encoder reranking.")
    rerank_candidate_k: int = Field(ge=1, le=100, description="Stage-1 pool size when reranking.")
    rerank_top_n: int = Field(ge=1, le=50, description="Chunks kept after re-scoring.")

    # CRAG graph shape
    max_loops: int = Field(ge=0, le=6, description="Corrective query-rewrite cap.")
    max_regen: int = Field(ge=0, le=5, description="Grounding-retry cap.")
    query_rewrite_enabled: bool = Field(description="Enable the corrective query-rewrite loop.")
    grounding_enabled: bool = Field(description="Enable the post-generation grounding check.")

    # Per-node temperatures
    grade_temperature: float = Field(ge=0.0, le=1.5, description="Relevance grader temp.")
    rewrite_temperature: float = Field(ge=0.0, le=1.5, description="Query rewriter temp.")
    generate_temperature: float = Field(ge=0.0, le=1.5, description="Answer generator temp.")
    grounding_temperature: float = Field(ge=0.0, le=1.5, description="Grounding checker temp.")


@dataclass(frozen=True)
class RetrievedChunk:
    """A passage returned by retrieval, with its source for citation."""

    chunk_id: str
    text: str
    score: float
    document_id: str  # public id (for citations)
    title: str = ""


@dataclass(frozen=True)
class Citation:
    document_id: str
    title: str
    score: float


# --- structured-output schemas (the LLM is forced to emit valid JSON) ---


class Grade(BaseModel):
    """Relevance verdict from the grade node (§6.4)."""

    relevant: bool = Field(description="Do the chunks actually answer the question?")
    reason: str = Field(default="", description="One-sentence justification.")


class Groundedness(BaseModel):
    """Verdict from check_grounded — is every claim supported by a chunk?"""

    grounded: bool = Field(description="Is the answer fully supported by the provided context?")
    reason: str = Field(default="", description="One-sentence justification.")


class DocClass(BaseModel):
    """Classifier output for the ingestion crew (§7.3). The crew's only real
    reasoning step — everything else is a deterministic tool (Principle 3)."""

    doc_type: str = Field(description="One of: contract, deck, chat, article, other.")
    topic: str = Field(default="", description="Short topical label for search/dashboards.")
    suggested_strategy: str = Field(
        default="fixed",
        description="One of: clause, slide, turn, fixed. How this document should be chunked.",
    )


@dataclass
class TextChunk:
    """A produced chunk awaiting embedding + persistence."""

    seq: int
    text: str
    token_count: int
    vector: list[float] = field(default_factory=list)
