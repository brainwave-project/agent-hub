"""Extractor — deterministic tool (§7.3). Turns a document into raw text.

know-hub's GetPayload returns either an already-extracted text layer OR the raw
binary for types Phase-1 SKIPPED (PDF/DOCX). When it's binary, THIS is where the
crew earns the Python service: pypdf / python-docx parse formats the TS worker
could not (review Issue 3). Image-only / unparseable docs set has_text=False and
the supervisor short-circuits to SKIPPED — a valid terminal state, not a failure.
"""

from __future__ import annotations

import io

from agent_hub.adapters.ports.knowhub import DocumentGateway
from agent_hub.application.ingestion.state import IngestState
from agent_hub.observability.logging import get_logger

log = get_logger("extractor")


def _parse_binary(data: bytes, mime: str) -> str:
    try:
        if "pdf" in mime:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        if "word" in mime or "officedocument" in mime or mime.endswith("docx"):
            import docx

            doc = docx.Document(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs).strip()
    except Exception as exc:
        log.warning("extractor.parse_failed", mime=mime, error=str(exc))
    return ""


class ExtractorAgent:
    def __init__(self, gateway: DocumentGateway) -> None:
        self._gw = gateway

    async def __call__(self, state: IngestState) -> dict:
        payload = await self._gw.get_payload(state["document_id"], state.get("actor_user_id", ""))
        if not payload.found:
            return {"has_text": False, "raw_text": "", "history": ["extractor:not_found"]}

        text = payload.raw_text
        if not text and payload.binary:
            text = _parse_binary(payload.binary, payload.mime_type)

        has_text = bool(text and text.strip())
        return {
            "raw_text": text,
            "binary_mime": payload.mime_type,
            "has_text": has_text,
            "history": [f"extractor:{'ok' if has_text else 'no_text'}"],
        }
