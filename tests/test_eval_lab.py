"""Eval Lab — store, splits, chunking, scoring, readiness, decisions, API.

No stack: know-hub is replaced by an in-memory fake that ranks by word overlap.
The fake is deliberately dumb; these tests check the lab's bookkeeping and
guards, not retrieval quality.
"""

import time
import uuid

import pytest
from fastapi.testclient import TestClient

from agent_hub.eval.lab.analysis import decide
from agent_hub.eval.lab.api import create_app
from agent_hub.eval.lab.chunking import chunk_text, locate_evidence
from agent_hub.eval.lab.models import (
    Candidate,
    CorpusDoc,
    DatasetMeta,
    Evidence,
    GoldenItem,
    SplitConfig,
    SuccessCriteria,
)
from agent_hub.eval.lab.readiness import assess
from agent_hub.eval.lab.scoring import RawHit, compute_metrics, rank_chunks
from agent_hub.eval.lab.splits import assign_split
from agent_hub.eval.lab.store import Immutable, LabStore
from agent_hub.eval.retrieval_metrics import score_all

# --- fake know-hub -----------------------------------------------------------------


class FakeKnowHub:
    base_url = "fake://know-hub"

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, tuple[str, str]]] = {}  # owner -> title -> (id, body)

    def ready(self) -> bool:
        return True

    def grpc_ready(self) -> bool:
        return True

    def list_titles(self, owner_id):
        return {t: v[0] for t, v in self.docs.get(owner_id, {}).items()}

    def upload(self, owner_id, doc_id, body):
        document_id = str(uuid.uuid4())
        self.docs.setdefault(owner_id, {})[doc_id] = (document_id, body)
        return document_id

    def embedding_status(self, owner_id, document_id):
        return "INDEXED"

    @staticmethod
    def _overlap(query, text):
        q = set(query.lower().split())
        return len(q & set(text.lower().split())) / max(1, len(q))

    def search_documents(self, query, owner_id, limit, hybrid, min_score):
        hits = [
            RawHit(doc_id=t, document_id=i, score=self._overlap(query, b), text=b[:200])
            for t, (i, b) in self.docs.get(owner_id, {}).items()
        ]
        return sorted(hits, key=lambda h: (-h.score, h.doc_id))[:limit]

    def search_chunks(self, query, owner_id, top_k, min_score):
        hits = []
        for title, (document_id, body) in self.docs.get(owner_id, {}).items():
            for c in chunk_text(body, 24, 4):
                hits.append(
                    RawHit(
                        doc_id=title,
                        document_id=document_id,
                        chunk_id=f"{title}-{c.index}",
                        score=self._overlap(query, c.content),
                        text=c.content,
                    )
                )
        return sorted(hits, key=lambda h: (-h.score, h.chunk_id))[:top_k]


def _seed_store(tmp_path) -> LabStore:
    store = LabStore(tmp_path)
    store.create_dataset(DatasetMeta(id="demo", name="Demo"))
    store.put_doc(
        "demo", 1, CorpusDoc(doc_id="leave", body="Employees get twenty one days of annual leave.")
    )
    store.put_doc(
        "demo",
        1,
        CorpusDoc(doc_id="refund", body="Refunds are accepted within thirty days of purchase."),
    )
    store.put_doc(
        "demo", 1, CorpusDoc(doc_id="lease", body="The lease requires sixty days written notice.")
    )
    return store


# --- store -----------------------------------------------------------------------------


def test_frozen_version_rejects_writes_and_fork_is_editable(tmp_path):
    store = _seed_store(tmp_path)
    store.replace_items(
        "demo", 1, [GoldenItem(id="q1", query="annual leave days", relevant={"leave": 3})]
    )
    manifest = store.freeze("demo", 1)
    assert manifest.content_hash == store.bundle("demo", 1).current_hash

    with pytest.raises(Immutable):
        store.put_doc("demo", 1, CorpusDoc(doc_id="new", body="x y z"))

    fork = store.fork("demo", 1, changelog="add a question")
    assert fork.version == 2 and fork.status == "draft" and fork.parent_version == 1
    store.put_item("demo", 2, GoldenItem(id="q2", query="refund window", relevant={"refund": 3}))
    assert store.bundle("demo", 2).current_hash != manifest.content_hash
    assert store.bundle("demo", 1).current_hash == manifest.content_hash


def test_out_of_band_edit_to_frozen_content_is_a_blocker(tmp_path):
    store = _seed_store(tmp_path)
    store.replace_items(
        "demo", 1, [GoldenItem(id="q1", query="annual leave days", relevant={"leave": 3})]
    )
    store.freeze("demo", 1)
    path = tmp_path / "datasets" / "demo" / "v1" / "corpus" / "leave.json"
    path.write_text(
        path.read_text(encoding="utf-8").replace("twenty one", "thirty"), encoding="utf-8"
    )
    report = assess(store.bundle("demo", 1))
    assert any(c["id"] == "integrity.hash" and c["severity"] == "blocker" for c in report["checks"])


# --- splits ------------------------------------------------------------------------------


def _items():
    out = []
    for doc in ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j"):
        for n in range(3):
            out.append(
                GoldenItem(
                    id=f"{doc}{n}",
                    query=f"question {doc} {n}",
                    relevant={doc: 3},
                    difficulty="easy" if doc < "f" else "hard",
                )
            )
    return out


def test_split_is_deterministic_and_never_straddles_a_document():
    config = SplitConfig(
        holdout_fraction=0.3, seed=7, stratify_by=["difficulty"], group_by="primary_doc"
    )
    first, summary = assign_split(_items(), config)
    second, _ = assign_split(_items(), config)
    assert [i.split for i in first] == [i.split for i in second]
    assert summary.straddling_groups == []
    assert summary.holdout > 0 and summary.dev > 0
    # Stratified: both difficulties have holdout items.
    assert all(sides["holdout"] > 0 for sides in summary.strata.values())


def test_pinned_items_keep_their_split(tmp_path):
    items = _items()
    items[0] = items[0].model_copy(update={"split": "holdout", "pinned": True})
    out, _ = assign_split(items, SplitConfig(group_by="none", seed=1))
    assert out[0].split == "holdout"


# --- chunking & scoring --------------------------------------------------------------------


def test_chunker_matches_core_arithmetic():
    # 512 tokens * 0.75 = 384 words per chunk; 64 * 0.75 = 48 overlap; step 336.
    chunks = chunk_text(" ".join(f"w{i}" for i in range(500)))
    assert [(c.start_word, c.end_word) for c in chunks] == [(0, 384), (336, 500)]


def test_evidence_split_by_a_chunk_boundary_is_detected():
    body = " ".join(f"w{i}" for i in range(40))
    chunks = chunk_text(body, max_tokens=16, overlap=0)  # 12 words per chunk
    located = locate_evidence(body, "w10 w11 w12 w13", chunks)  # crosses 11|12
    assert located["found_in_doc"] and located["straddles_boundary"]


def test_chunk_scoring_credits_novel_evidence_only():
    item = GoldenItem(
        id="q",
        query="q",
        relevant={"doc": 3},
        evidence=[Evidence(doc_id="doc", text="the answer is 42")],
    )
    hits = [
        RawHit(doc_id="doc", chunk_id="c1", text="intro text"),
        RawHit(doc_id="doc", chunk_id="c2", text="so The answer is 42 indeed"),
        RawHit(doc_id="doc", chunk_id="c3", text="repeat: the answer is 42"),
    ]
    ranked, grades = rank_chunks(hits, item, item.relevant)
    assert [h.grade for h in ranked] == [0, 3, 0]
    assert ranked[2].novel is False
    metrics = compute_metrics([h.unit_id for h in ranked], grades, k=3)
    assert metrics["mrr"] == 0.5 and metrics["recall@k"] == 1.0


def test_document_level_matches_offline_harness_metrics():
    ranked = ["x", "leave", "y"]
    lab = compute_metrics(ranked, {"leave": 3}, k=5)
    harness = score_all(ranked, ["leave"], k=5)
    assert all(lab[m] == harness[m] for m in harness)


# --- readiness -----------------------------------------------------------------------------


def test_readiness_flags_label_errors_and_cross_split_leakage(tmp_path):
    store = _seed_store(tmp_path)
    store.replace_items(
        "demo",
        1,
        [
            GoldenItem(
                id="q1", query="how many annual leave days", relevant={"leave": 3}, split="dev"
            ),
            GoldenItem(
                id="q2", query="annual leave days how many", relevant={"leave": 3}, split="holdout"
            ),
            GoldenItem(id="q3", query="refund window", relevant={"missing-doc": 3}, split="dev"),
            GoldenItem(
                id="q4",
                query="lease notice",
                relevant={"lease": 3},
                split="dev",
                evidence=[Evidence(doc_id="lease", text="ninety days notice")],
            ),
        ],
    )
    report = assess(store.bundle("demo", 1))
    blockers = {c["id"] for c in report["checks"] if c["severity"] == "blocker"}
    assert "labels.unresolved" in blockers
    assert "labels.evidence_missing" in blockers
    assert "leakage.duplicate_cross_split" in blockers
    assert report["verdict"] == "not_ready"


# --- decisions -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mean", "low", "high", "verdict"),
    [
        (0.10, 0.04, 0.16, "success"),
        (0.02, 0.01, 0.03, "detectable_but_small"),
        (-0.08, -0.12, -0.03, "harm"),
        (0.00, -0.02, 0.03, "no_meaningful_effect"),
        (0.04, -0.03, 0.11, "inconclusive"),
    ],
)
def test_decision_rule_separates_zero_from_minimum_effect(mean, low, high, verdict):
    criteria = SuccessCriteria(min_effect=0.05, direction="increase")
    assert decide({"mean": mean, "low": low, "high": high}, criteria)["verdict"] == verdict


def test_non_inferiority_uses_margin():
    criteria = SuccessCriteria(min_effect=0.05, direction="non_inferior")
    assert decide({"mean": -0.01, "low": -0.04, "high": 0.02}, criteria)["verdict"] == "success"
    assert (
        decide({"mean": -0.03, "low": -0.09, "high": 0.02}, criteria)["verdict"] == "inconclusive"
    )


# --- API end to end ------------------------------------------------------------------------------


def _wait(client, job):
    for _ in range(200):
        state = client.get(f"/api/eval/jobs/{job['id']}").json()
        if state["status"] != "running":
            assert state["status"] == "done", state
            return state
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_api_full_experiment_lifecycle(tmp_path):
    client = TestClient(create_app(store=LabStore(tmp_path), knowhub=FakeKnowHub()))
    base = "/api/eval/datasets/policies/versions/1"

    assert (
        client.post("/api/eval/datasets", json={"name": "Policies", "id": "policies"}).status_code
        == 200
    )
    docs = {
        "leave": "Employees get twenty one days of annual leave each year.",
        "refund": "Refunds are accepted within thirty days of purchase with a receipt.",
        "lease": "The lease requires sixty days written notice to terminate.",
        "travel": "Travel expenses are reimbursed against original receipts.",
    }
    for doc_id, body in docs.items():
        assert (
            client.put(f"{base}/corpus/{doc_id}", json={"doc_id": doc_id, "body": body}).status_code
            == 200
        )
    lines = [
        f'{{"id": "q{n}", "query": "{q}", "relevant_doc_ids": ["{d}"]}}'
        for n, (q, d) in enumerate(
            [
                ("annual leave days", "leave"),
                ("leave per year", "leave"),
                ("refund within days", "refund"),
                ("purchase refund receipt", "refund"),
                ("lease notice terminate", "lease"),
                ("written notice lease", "lease"),
                ("travel reimbursed receipts", "travel"),
                ("expenses travel", "travel"),
            ]
        )
    ]
    assert client.post(f"{base}/items/import", json={"jsonl": "\n".join(lines)}).status_code == 200
    split = client.post(
        f"{base}/split",
        json={"holdout_fraction": 0.5, "stratify_by": [], "group_by": "primary_doc"},
    ).json()
    assert split["summary"]["straddling_groups"] == []

    _wait(client, client.post(f"{base}/index", json={"normalization": "off"}).json())

    searched = client.post(
        "/api/eval/search",
        json={
            "dataset_id": "policies",
            "version": 1,
            "query": "annual leave days",
            "k": 3,
            "item_id": "q0",
        },
    ).json()
    assert searched["result"]["metrics"]["hit_rate"] == 1.0
    assert searched["explain"]["relevant_count"] == 1

    # Ad-hoc holdout access is refused.
    refused = client.post(
        "/api/eval/runs", json={"dataset_id": "policies", "version": 1, "split": "holdout"}
    )
    assert refused.status_code == 409

    experiment = {
        "name": "Hybrid vs vector",
        "id": "hyb",
        "dataset_id": "policies",
        "dataset_version": 1,
        "hypothesis": "Hybrid improves recall@3",
        "baseline": {"name": "vector", "hybrid": False},
        "treatment": {"name": "hybrid", "hybrid": True},
        "criteria": {"k": 3},
    }
    assert client.post("/api/eval/experiments", json=experiment).status_code == 200
    # Not registrable while the dataset is a draft.
    assert client.post("/api/eval/experiments/hyb/register").status_code == 409
    assert client.post(f"{base}/freeze").status_code == 200
    registered = client.post("/api/eval/experiments/hyb/register").json()
    assert registered["status"] == "registered" and registered["dataset_hash"]
    # Design is locked after registration.
    assert client.put("/api/eval/experiments/hyb", json=experiment).status_code == 409

    _wait(client, client.post("/api/eval/experiments/hyb/runs", json={"split": "holdout"}).json())
    # A second holdout look needs a reason.
    assert (
        client.post("/api/eval/experiments/hyb/runs", json={"split": "holdout"}).status_code == 422
    )

    report = client.get("/api/eval/experiments/hyb/report", params={"split": "holdout"}).json()
    assert report["status"] == "ok"
    assert report["confirmatory"] is True
    assert report["decision"]["verdict"] in {
        "success",
        "detectable_but_small",
        "harm",
        "no_meaningful_effect",
        "inconclusive",
    }
    assert len(report["all_runs"]) == 2

    concluded = client.post("/api/eval/experiments/hyb/conclude", json={"note": "done"}).json()
    assert concluded["status"] == "concluded"


def test_accepted_candidate_with_split_survives_a_bulk_resplit(tmp_path):
    """A curator's split choice at accept-time must be pinned, or the next bulk
    /split silently overwrites it (this was a real bug: accept() set `split`
    but never `pinned`)."""
    store = LabStore(tmp_path)
    client = TestClient(create_app(store=store, knowhub=FakeKnowHub()))
    base = "/api/eval/datasets/policies/versions/1"

    assert (
        client.post("/api/eval/datasets", json={"name": "Policies", "id": "policies"}).status_code
        == 200
    )
    assert (
        client.put(
            f"{base}/corpus/leave",
            json={"doc_id": "leave", "body": "Employees get twenty one days of annual leave each year."},
        ).status_code
        == 200
    )

    store.save_artifact(
        "policies",
        1,
        "candidates",
        [Candidate(id="cand-1", doc_id="leave", query="how many annual leave days").model_dump()],
    )

    accepted = client.post(f"{base}/candidates/cand-1/accept", json={"split": "holdout"}).json()
    accepted_item = next(i for i in accepted["items"] if i["id"] == "syn-001")
    assert accepted_item["split"] == "holdout"
    assert accepted_item["pinned"] is True

    lines = [
        f'{{"id": "q{n}", "query": "{q}", "relevant_doc_ids": ["leave"]}}'
        for n, q in enumerate(["leave days", "annual leave", "leave per year", "days of leave"])
    ]
    assert client.post(f"{base}/items/import", json={"jsonl": "\n".join(lines)}).status_code == 200

    split = client.post(
        f"{base}/split",
        json={"holdout_fraction": 0.1, "seed": 1, "stratify_by": [], "group_by": "none"},
    ).json()
    resplit_item = next(i for i in split["bundle"]["items"] if i["id"] == "syn-001")
    assert resplit_item["split"] == "holdout"
