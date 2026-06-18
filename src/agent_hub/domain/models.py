"""Pure domain types — no framework or I/O dependencies (hexagonal core).

Structured-output schemas (Grade, Groundedness, DocClass) are the contracts the
reasoning LLM must satisfy; nodes depend on these, never on a raw model client.
"""

from dataclasses import dataclass, field

from pydantic import BaseModel, Field


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
