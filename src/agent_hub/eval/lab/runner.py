"""Search know-hub, score queries and splits, and index a dataset's corpus.

Two retrieval paths, because know-hub exposes two:

* **document** — HTTP `GET /api/semantic/search`. Hybrid (vector + full-text,
  RRF-fused) or vector-only. Collapses to the best chunk per document.
* **chunk** — gRPC `DocumentInternal.SemanticSearch`, the path CRAG uses.
  Vector-only by contract, returns individual chunks. Note: a `min_score` of 0
  on this path means "know-hub's SIMILARITY_THRESHOLD", which the eval stack
  sets to 0.0 but the dev stack sets to 0.60.

Every search runs as the dataset index's synthetic owner, so know-hub's
ownership predicate returns only that index's documents.
"""

from __future__ import annotations

import statistics
import time
import uuid
from collections.abc import Callable

from agent_hub.eval.arabic.normalize import (
    AGGRESSIVE,
    CONSERVATIVE,
    OFF,
    NormalizeConfig,
    normalize,
)
from agent_hub.eval.knowhub_client import KnowHubEvalClient
from agent_hub.eval.lab.jobs import Job
from agent_hub.eval.lab.models import (
    LAB_METRICS,
    GoldenItem,
    IndexedDoc,
    IndexState,
    QueryResult,
    Run,
    VariantSpec,
    now_iso,
)
from agent_hub.eval.lab.scoring import RawHit, compute_metrics, explain, rank_chunks, rank_documents
from agent_hub.eval.lab.store import Bundle, LabStore, owner_id_for
from agent_hub.eval.stats import mean_ci

__all__ = [
    "NORMALIZERS",
    "IndexNotReady",
    "KnowHubLab",
    "search_once",
    "run_split",
    "aggregate",
    "index_corpus",
    "index_is_current",
]

NORMALIZERS: dict[str, NormalizeConfig] = {
    "off": OFF,
    "conservative": CONSERVATIVE,
    "aggressive": AGGRESSIVE,
}

TERMINAL = {"INDEXED", "SKIPPED", "FAILED"}


class IndexNotReady(RuntimeError):
    pass


class KnowHubLab:
    """know-hub access for the lab, scoped per call to an owner id.

    Everything the lab needs from know-hub goes through these methods, which is
    also the seam the tests replace with an in-memory fake.
    """

    def __init__(self, base_url: str, grpc_host: str, grpc_port: int) -> None:
        self.base_url = base_url
        self.grpc_target = f"{grpc_host}:{grpc_port}"
        self._channel = None
        self._stub = None

    def client(self, owner_id: str) -> KnowHubEvalClient:
        return KnowHubEvalClient(base_url=self.base_url, user_id=owner_id, timeout=60.0)

    # --- health -------------------------------------------------------------

    def ready(self) -> bool:
        with self.client(str(uuid.UUID(int=0))) as client:
            return client.ready()

    def _grpc(self):
        if self._stub is None:
            import grpc

            from agent_hub.adapters.grpc import knowhub_pb2_grpc as pb_grpc

            self._channel = grpc.insecure_channel(self.grpc_target)
            self._stub = pb_grpc.DocumentInternalStub(self._channel)
        return self._stub

    def grpc_ready(self) -> bool:
        try:
            import grpc

            self._grpc()
            grpc.channel_ready_future(self._channel).result(timeout=2)
            return True
        except Exception:  # noqa: BLE001
            return False

    # --- ingestion ----------------------------------------------------------

    def list_titles(self, owner_id: str) -> dict[str, str]:
        """title -> documentId for everything this owner has uploaded."""
        out: dict[str, str] = {}
        with self.client(owner_id) as client:
            offset = 0
            while True:
                response = client._client.get(
                    f"{client._base}/api/documents", params={"limit": 100, "offset": offset}
                )
                response.raise_for_status()
                page = response.json().get("items", [])
                for doc in page:
                    if doc.get("title"):
                        out[str(doc["title"])] = str(doc.get("documentId") or doc.get("id"))
                if len(page) < 100:
                    return out
                offset += 100

    def upload(self, owner_id: str, doc_id: str, body: str) -> str:
        with self.client(owner_id) as client:
            return client.upload_markdown(doc_id, body, idempotency_key=f"lab-{owner_id}-{doc_id}")

    def embedding_status(self, owner_id: str, document_id: str) -> str:
        with self.client(owner_id) as client:
            return client.embedding_status(document_id)

    # --- retrieval ----------------------------------------------------------

    def search_documents(
        self, query: str, owner_id: str, limit: int, hybrid: bool, min_score: float
    ) -> list[RawHit]:
        with self.client(owner_id) as client:
            hits = client.search(query, limit=limit, hybrid=hybrid, min_score=min_score)
        return [
            RawHit(
                doc_id=h.title,
                document_id=h.document_id,
                score=h.similarity_score,
                rrf_score=h.rrf_score,
                text=h.excerpt,
            )
            for h in hits
        ]

    def search_chunks(
        self, query: str, owner_id: str, top_k: int, min_score: float
    ) -> list[RawHit]:
        from agent_hub.adapters.grpc import knowhub_pb2 as pb

        response = self._grpc().SemanticSearch(
            pb.SemanticSearchRequest(
                query=query,
                top_k=top_k,
                actor_user_id=owner_id,
                is_admin=False,
                min_score=min_score,
            ),
            timeout=60,
        )
        return [
            RawHit(
                doc_id=r.title,
                document_id=r.document_id,
                chunk_id=r.chunk_id,
                score=r.score,
                text=r.text,
            )
            for r in response.results
        ]


def index_is_current(bundle: Bundle, normalization: str) -> tuple[bool, str]:
    state = bundle.indexes.get(normalization)
    if state is None:
        return False, f"corpus has no '{normalization}' index — index it in Golden Sets → Index"
    if state.corpus_hash != bundle.corpus_hash(normalization):
        return False, f"'{normalization}' index is stale — the corpus changed after indexing"
    not_ready = [
        d for d in bundle.corpus if state.documents.get(d, IndexedDoc()).status != "INDEXED"
    ]
    if not_ready:
        return False, f"'{normalization}' index has {len(not_ready)} document(s) not INDEXED"
    return True, ""


def search_once(
    kh: KnowHubLab,
    bundle: Bundle,
    variant: VariantSpec,
    query: str,
    k: int,
    item: GoldenItem | None = None,
    relevant_override: dict[str, int] | None = None,
) -> tuple[QueryResult, dict]:
    ok, why = index_is_current(bundle, variant.corpus_index)
    if not ok:
        raise IndexNotReady(why)
    owner = bundle.indexes[variant.corpus_index].owner_id

    subject = item or GoldenItem(id="adhoc", query=query)
    if relevant_override is not None:
        subject = subject.model_copy(update={"relevant": relevant_override})
    relevant = dict(subject.relevant)

    query_sent = normalize(query, NORMALIZERS[variant.query_normalization])
    started = time.perf_counter()
    if variant.level == "chunk":
        hits = kh.search_chunks(query_sent, owner, variant.fetch, variant.min_score)
    else:
        hits = kh.search_documents(
            query_sent, owner, variant.fetch, variant.hybrid, variant.min_score
        )
    latency = (time.perf_counter() - started) * 1000

    if variant.level == "chunk":
        ranked, grades = rank_chunks(hits, subject, relevant)
    else:
        ranked, grades = rank_documents(hits, relevant), relevant

    # judged@k: how much of the scored window rests on the "unjudged = not
    # relevant" assumption. Low values mean the metrics are a lower bound.
    judged = subject.judged_docs
    window = ranked[:k]
    judged_at_k = (
        sum(1 for h in window if h.doc_id in judged) / len(window) if judged and window else None
    )

    result = QueryResult(
        item_id=subject.id,
        query=query,
        query_sent=query_sent,
        ranked=ranked,
        relevant_units=grades,
        metrics=compute_metrics([h.unit_id for h in ranked], grades, k) if grades else {},
        judged_at_k=judged_at_k,
        latency_ms=latency,
    )
    return result, explain(ranked, grades, k)


def aggregate(results: list[QueryResult], resamples: int = 2000) -> dict[str, dict[str, float]]:
    scored = [r for r in results if r.error is None and r.metrics]
    out: dict[str, dict[str, float]] = {}
    for metric in LAB_METRICS:
        values = [r.metrics[metric] for r in scored]
        if not values:
            continue
        interval = mean_ci(values, resamples=resamples)
        out[metric] = {
            "mean": interval.mean,
            "low": interval.low,
            "high": interval.high,
            "n": interval.n,
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return out


def run_split(
    kh: KnowHubLab,
    store: LabStore,
    bundle: Bundle,
    variant: VariantSpec,
    split: str,
    k: int,
    job: Job | None = None,
    experiment_id: str | None = None,
    arm: str = "adhoc",
    on_start: Callable[[Run], None] | None = None,
) -> Run:
    """Score every answerable item in `split`. `on_start` fires once the run exists on disk.

    The run is persisted BEFORE any query executes, so an experiment can record
    it immediately. A run that later fails still appears in the experiment's run
    history rather than vanishing.
    """
    ok, why = index_is_current(bundle, variant.corpus_index)
    if not ok:
        raise IndexNotReady(why)

    selected = [i for i in bundle.items if split == "all" or i.split == split]
    answerable = [i for i in selected if i.answerable]
    run = Run(
        id=f"run_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}",
        experiment_id=experiment_id,
        arm=arm,  # type: ignore[arg-type]
        dataset_id=bundle.meta.id,
        dataset_version=bundle.manifest.version,
        dataset_hash=bundle.current_hash,
        dataset_status=bundle.manifest.status,
        corpus_hash=bundle.corpus_hash(variant.corpus_index),
        variant=variant,
        split=split,  # type: ignore[arg-type]
        k=k,
        skipped_unanswerable=[i.id for i in selected if not i.answerable],
    )
    store.save_run(run)
    if on_start:
        on_start(run)
    if job:
        job.progress(0, len(answerable), f"{arm}: {variant.name} on {split}")

    try:
        for n, item in enumerate(answerable, start=1):
            try:
                result, _ = search_once(kh, bundle, variant, item.query, k, item=item)
            except IndexNotReady:
                raise
            except Exception as exc:  # noqa: BLE001
                # Recorded, and EXCLUDED from aggregates. Scoring an
                # infrastructure failure as 0 would report a timeout as a miss.
                result = QueryResult(
                    item_id=item.id,
                    query=item.query,
                    query_sent=item.query,
                    error=f"{type(exc).__name__}: {exc}",
                )
            run.results.append(result)
            if job:
                job.progress(n)
    except Exception as exc:
        run.status = "failed"
        run.error = f"{type(exc).__name__}: {exc}"
        run.finished_at = now_iso()
        store.save_run(run)
        raise

    run.aggregates = aggregate(run.results)
    ok_results = [r for r in run.results if r.error is None]
    judged = [r.judged_at_k for r in ok_results if r.judged_at_k is not None]
    run.diagnostics = {
        "errors": float(len(run.results) - len(ok_results)),
        "empty_results_share": (
            sum(1 for r in ok_results if not r.ranked) / len(ok_results) if ok_results else 0.0
        ),
        "mean_latency_ms": statistics.fmean(r.latency_ms for r in ok_results)
        if ok_results
        else 0.0,
    }
    if judged:
        run.diagnostics["judged_at_k"] = statistics.fmean(judged)
    run.status = "done"
    run.finished_at = now_iso()
    store.save_run(run)
    return run


def index_corpus(
    kh: KnowHubLab,
    store: LabStore,
    bundle: Bundle,
    normalization: str,
    job: Job | None = None,
    timeout_s: float = 1800.0,
    poll_interval_s: float = 3.0,
) -> IndexState:
    """Upload every document under this corpus's owner id and wait for INDEXED.

    Resumable: documents already recorded as INDEXED for the same corpus hash
    are skipped, and documents the owner already uploaded are re-attached by
    title rather than uploaded twice.
    """
    digest = bundle.corpus_hash(normalization)
    owner = owner_id_for(bundle.meta.id, digest)
    previous = bundle.indexes.get(normalization)
    state = (
        previous
        if previous and previous.corpus_hash == digest
        else IndexState(normalization=normalization, corpus_hash=digest, owner_id=owner)  # type: ignore[arg-type]
    )
    state.finished_at = None
    config = NORMALIZERS[normalization]
    total = len(bundle.corpus)

    existing = kh.list_titles(owner)
    for n, (doc_id, doc) in enumerate(sorted(bundle.corpus.items()), start=1):
        recorded = state.documents.get(doc_id)
        if recorded and recorded.status == "INDEXED":
            continue
        if doc_id in existing:
            state.documents[doc_id] = IndexedDoc(document_id=existing[doc_id], status="PENDING")
        else:
            document_id = kh.upload(owner, doc_id, normalize(doc.body, config))
            state.documents[doc_id] = IndexedDoc(document_id=document_id, status="PENDING")
        if job:
            job.progress(n, total * 2, f"uploaded {doc_id}")
    store.save_index(bundle.meta.id, bundle.manifest.version, state)

    deadline = time.monotonic() + timeout_s
    while True:
        for entry in state.documents.values():
            if entry.status in TERMINAL:
                continue
            try:
                entry.status = kh.embedding_status(owner, entry.document_id)
            except Exception:  # noqa: BLE001 — transient while booting
                continue
        done = sum(1 for s in state.documents.values() if s.status in TERMINAL)
        if job:
            job.progress(total + done, total * 2, f"embedded {done}/{total}")
        store.save_index(bundle.meta.id, bundle.manifest.version, state)
        if done == len(state.documents) or time.monotonic() >= deadline:
            break
        time.sleep(poll_interval_s)

    state.finished_at = now_iso()
    store.save_index(bundle.meta.id, bundle.manifest.version, state)
    return state
