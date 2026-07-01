"""Ollama adapters — the local, air-gapped reasoning + embedding models.

Implements the reasoning ports with langchain-ollama's ChatOllama. Two things are
injected rather than hardcoded:

  * PROMPTS come from the `PromptRegistry` (Langfuse-managed, §13.5). This adapter
    holds no prompt text of its own — it asks the registry by key.
  * TEMPERATURES come from the live `GovernanceStore` (§4.4 principal's note is now
    an operator-tunable knob, not a constant): grading/rewrite/grounding default
    to deterministic 0, generation runs warmer. A per-temperature ChatOllama is
    built lazily and cached, so changing a temperature in the UI takes effect on
    the next run with no redeploy.

Structured nodes use `.with_structured_output(Schema)`. Because an 8B-Q4 model is
good but not infallible, every structured call FAILS CLOSED: a parse/transport
error becomes the conservative verdict (not relevant / not grounded) rather than
an exception that kills the run.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

from agent_hub.adapters import prompts as P
from agent_hub.adapters.governance_repo import GovernanceStore
from agent_hub.adapters.prompts import PromptRegistry
from agent_hub.config import Settings
from agent_hub.domain.models import (
    DocClass,
    GovernanceSettings,
    Grade,
    Groundedness,
    RetrievedChunk,
)
from agent_hub.observability.logging import get_logger

log = get_logger("ollama")


def _format_context(chunks: list[RetrievedChunk]) -> str:
    # Clear delimiters around untrusted document content — the first line of
    # defence against prompt injection via documents (§14).
    if not chunks:
        return "<no context retrieved>"
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] (doc:{c.document_id} score:{c.score:.2f})\n{c.text}")
    return "\n\n".join(parts)


class OllamaReasoner:
    """CragReasoner + Classifier backed by one local tool-calling model."""

    def __init__(
        self,
        base_url: str,
        model: str,
        prompts: PromptRegistry,
        settings: Settings,
        governance: GovernanceStore | None = None,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._prompts = prompts
        self._settings = settings
        self._governance = governance
        self._clients: dict[float, ChatOllama] = {}  # temperature → cached client

    # -- infra helpers ------------------------------------------------------

    def _gov(self) -> GovernanceSettings | None:
        return self._governance.current() if self._governance is not None else None

    def _chat(self, temperature: float) -> ChatOllama:
        key = round(float(temperature), 3)
        client = self._clients.get(key)
        if client is None:
            client = ChatOllama(model=self._model, base_url=self._base_url, temperature=key)
            self._clients[key] = client
        return client

    # -- CRAG ---------------------------------------------------------------

    async def grade(self, question: str, chunks: list[RetrievedChunk]) -> Grade:
        g = self._gov()
        temp = g.grade_temperature if g else self._settings.crag_grade_temperature
        try:
            grader = self._chat(temp).with_structured_output(Grade)
            result = await grader.ainvoke(
                [
                    SystemMessage(content=self._prompts.get(P.CRAG_GRADE)),
                    HumanMessage(
                        content=f"QUESTION:\n{question}\n\nCONTEXT:\n{_format_context(chunks)}"
                    ),
                ]
            )
            return result if isinstance(result, Grade) else Grade(**result)
        except Exception as exc:  # fail closed → triggers a corrective rewrite
            log.warning("grade.failed", error=str(exc))
            return Grade(relevant=False, reason="grader error — treated as not relevant")

    async def rewrite_query(
        self, question: str, current_query: str, chunks: list[RetrievedChunk]
    ) -> str:
        g = self._gov()
        temp = g.rewrite_temperature if g else self._settings.crag_rewrite_temperature
        messages = [
            SystemMessage(content=self._prompts.get(P.CRAG_REWRITE_QUERY)),
            HumanMessage(
                content=(
                    f"ORIGINAL QUESTION:\n{question}\n\n"
                    f"PREVIOUS QUERY:\n{current_query}\n\n"
                    f"WEAK RESULTS:\n{_format_context(chunks)}"
                )
            ),
        ]
        resp = await self._chat(temp).ainvoke(messages)
        text = (resp.content if isinstance(resp.content, str) else str(resp.content)).strip()
        return text.strip('"').splitlines()[0] if text else question

    def _generate_messages(self, question: str, chunks: list[RetrievedChunk]):
        return [
            SystemMessage(content=self._prompts.get(P.CRAG_GENERATE)),
            HumanMessage(
                content=f"QUESTION:\n{question}\n\nCONTEXT:\n{_format_context(chunks)}"
            ),
        ]

    def _writer(self) -> ChatOllama:
        g = self._gov()
        temp = g.generate_temperature if g else self._settings.crag_generate_temperature
        return self._chat(temp)

    async def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        resp = await self._writer().ainvoke(self._generate_messages(question, chunks))
        return resp.content if isinstance(resp.content, str) else str(resp.content)

    async def stream_generate(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> AsyncIterator[str]:
        async for part in self._writer().astream(self._generate_messages(question, chunks)):
            text = part.content if isinstance(part.content, str) else str(part.content)
            if text:
                yield text

    async def check_grounded(self, answer: str, chunks: list[RetrievedChunk]) -> Groundedness:
        g = self._gov()
        temp = g.grounding_temperature if g else self._settings.crag_grounding_temperature
        try:
            checker = self._chat(temp).with_structured_output(Groundedness)
            result = await checker.ainvoke(
                [
                    SystemMessage(content=self._prompts.get(P.CRAG_CHECK_GROUNDED)),
                    HumanMessage(
                        content=f"ANSWER:\n{answer}\n\nCONTEXT:\n{_format_context(chunks)}"
                    ),
                ]
            )
            return result if isinstance(result, Groundedness) else Groundedness(**result)
        except Exception as exc:  # fail closed → one bounded regeneration
            log.warning("check_grounded.failed", error=str(exc))
            return Groundedness(grounded=False, reason="grounding check error")

    # -- Ingestion (Classifier) --------------------------------------------

    async def classify(self, sample_text: str) -> DocClass:
        try:
            classifier = self._chat(0.0).with_structured_output(DocClass)
            result = await classifier.ainvoke(
                [
                    SystemMessage(content=self._prompts.get(P.INGESTION_CLASSIFY)),
                    HumanMessage(content=f"DOCUMENT SAMPLE:\n{sample_text[:4000]}"),
                ]
            )
            return result if isinstance(result, DocClass) else DocClass(**result)
        except Exception as exc:  # fail closed → safe default strategy
            log.warning("classify.failed", error=str(exc))
            return DocClass(doc_type="other", topic="", suggested_strategy="fixed")


class OllamaEmbedder:
    """Embedder port — nomic-embed-text, same model/box as Phase 1."""

    def __init__(self, base_url: str, model: str = "nomic-embed-text") -> None:
        self._emb = OllamaEmbeddings(model=model, base_url=base_url)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._emb.aembed_documents(texts)
