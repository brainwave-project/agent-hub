"""Minimal know-hub HTTP client for the eval harness.

Deliberately separate from `adapters/know_hub_grpc.py`. That client serves the
CRAG hot path and returns chunk-level results; this one drives ingestion and
document-level retrieval for evaluation, and needs upload/status endpoints the
gRPC contract does not expose.

**It bypasses Kong**, talking to know-hub directly. That is safe here and worth
understanding: `requirePermission` (know-hub/src/middleware/permission.middleware.ts)
reads `X-User-Roles` and nothing else — it never validates a JWT, because it
trusts Kong to have done so upstream. So a direct caller on :3002 supplies the
headers Kong would have injected. Convenient for a local harness; also a live
demonstration of why know-hub must never be exposed outside the compose network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

__all__ = ["SearchHit", "KnowHubEvalClient", "EVAL_USER_ID"]

#: Fixed synthetic actor. Stable across runs so re-seeding is idempotent.
#:
#: Must be a *strictly valid* UUID: know-hub casts it to ::uuid in the ownership
#: predicate and Prisma parses it on the way back out, so anything outside
#: [0-9a-f-] makes every document query fail with P2023 rather than returning
#: empty. Only hex digits are available — "eval" cannot be spelled here.
EVAL_USER_ID = "00000000-0000-4000-8000-00000000e7a1"


@dataclass(frozen=True)
class SearchHit:
    """One document-level search result.

    Note what is NOT here: a chunk id. `GET /api/semantic/search` collapses to
    `DISTINCT ON (document_id)` and returns only the best-matching chunk's text
    as `excerpt`. Chunk-level metrics require the gRPC `SemanticSearch` path.
    """

    document_id: str
    title: str
    excerpt: str
    similarity_score: float
    rrf_score: float
    embedding_status: str = ""

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> SearchHit:
        return cls(
            document_id=raw.get("documentId", ""),
            title=raw.get("title", ""),
            excerpt=raw.get("excerpt", ""),
            similarity_score=float(raw.get("similarityScore", 0.0) or 0.0),
            rrf_score=float(raw.get("rrfScore", 0.0) or 0.0),
            embedding_status=raw.get("embeddingStatus", ""),
        )


class KnowHubEvalClient:
    def __init__(
        self,
        base_url: str = "http://know-hub:3002",
        user_id: str = EVAL_USER_ID,
        roles: tuple[str, ...] = ("EDITOR",),
        timeout: float = 60.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        # EDITOR, deliberately NOT ADMIN. know-hub's search predicate is
        #   (isAdmin OR owner_id = actor)
        # so an ADMIN eval would score against every document in the database —
        # including anything left over from unrelated work — and the same
        # question would score differently on two machines. EDITOR scopes
        # retrieval to documents this harness uploaded, which is what makes a
        # run reproducible. EDITOR carries both permissions needed here:
        # document:create (upload) and document:read (search).
        #
        # Note: reindex() requires permission:manage, i.e. ADMIN. Pass
        # roles=("ADMIN",) explicitly if you need it — at the cost of isolation.
        self._headers = {
            "X-User-Id": user_id,
            "X-User-Roles": json.dumps(list(roles)),
        }
        self._client = httpx.Client(timeout=timeout, headers=self._headers)

    def __enter__(self) -> KnowHubEvalClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # --- health ---------------------------------------------------------

    def ready(self) -> bool:
        try:
            r = self._client.get(f"{self._base}/health/ready", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    # --- ingestion ------------------------------------------------------

    def upload_markdown(self, doc_id: str, body: str) -> str:
        """Upload one corpus document. Returns its public `documentId`.

        `title` is set to the corpus doc id, which is what makes search hits
        joinable back onto golden records without a separate manifest.

        `text/markdown` matters: it matches know-hub's `TEXT_MIME_RE`
        (src/ai/extractor.ts), so the semantic worker extracts text inline
        rather than marking the document SKIPPED.
        """
        response = self._client.post(
            f"{self._base}/api/documents",
            files={"file": (f"{doc_id}.md", body.encode("utf-8"), "text/markdown")},
            data={"title": doc_id, "payloadEngine": "minio", "mimeType": "text/markdown"},
            headers={"Idempotency-Key": f"eval-seed-{doc_id}"},
        )
        response.raise_for_status()
        payload = response.json()
        document_id = payload.get("documentId") or payload.get("id")
        if not document_id:
            raise RuntimeError(f"upload of {doc_id} returned no documentId: {payload}")
        return str(document_id)

    def list_documents(self, limit: int = 100) -> list[dict[str, Any]]:
        response = self._client.get(f"{self._base}/api/documents", params={"limit": limit})
        response.raise_for_status()
        payload = response.json()
        return payload.get("items", payload if isinstance(payload, list) else [])

    def embedding_status(self, document_id: str) -> str:
        response = self._client.get(f"{self._base}/api/semantic/status/{document_id}")
        response.raise_for_status()
        return str(response.json().get("embeddingStatus", "UNKNOWN"))

    def reindex(self, document_id: str) -> None:
        self._client.post(f"{self._base}/api/semantic/reindex/{document_id}").raise_for_status()

    # --- retrieval ------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 10,
        hybrid: bool = True,
        min_score: float = 0.0,
    ) -> list[SearchHit]:
        """Document-level search, best-first.

        `min_score` defaults to 0.0 here, not know-hub's 0.6. A threshold that
        silently drops results would truncate the ranking before the metrics
        ever see it, and recall@k would measure the threshold rather than the
        retriever. Filtering is a decision the eval should make explicitly.
        """
        response = self._client.get(
            f"{self._base}/api/semantic/search",
            params={
                "q": query,
                "limit": limit,
                "hybrid": "true" if hybrid else "false",
                "minScore": min_score,
            },
        )
        response.raise_for_status()
        return [SearchHit.from_json(h) for h in response.json().get("results", [])]
