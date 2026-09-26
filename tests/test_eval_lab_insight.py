"""Eval Lab extensions — label audit, stats simulations, diffs, pooling, run analysis,
synthetic generation and the Learning Path. No stack: know-hub and Ollama are faked.
"""

import json
import math
import uuid

from fastapi.testclient import TestClient

from agent_hub.eval.lab.agreement import agreement_report, cohens_kappa, weighted_kappa
from agent_hub.eval.lab.api import create_app
from agent_hub.eval.lab.chunking import chunk_text
from agent_hub.eval.lab.diff import diff_versions
from agent_hub.eval.lab.generate import generate_candidates, lexical_overlap
from agent_hub.eval.lab.learning import evaluate, update_progress
from agent_hub.eval.lab.models import (
    CorpusDoc,
    DatasetMeta,
    GoldenItem,
    QueryResult,
    RankedHit,
    Run,
    VariantSpec,
)
from agent_hub.eval.lab.runanalysis import failure_analysis, k_sweep, threshold_sweep
from agent_hub.eval.lab.scoring import RawHit
from agent_hub.eval.lab.simulate import coverage_sim, phacking_sim, power_sim
from agent_hub.eval.lab.store import LabStore, content_hash
from agent_hub.eval.stats import cohens_dz, holm_adjust, paired_t_p

# --- stats building blocks ------------------------------------------------------------


def test_holm_adjustment_hand_computed():
    # sorted: a .01 x3 = .03 ; c .03 x2 = .06 ; b .04 x1 = .04 -> monotone max .06
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert math.isclose(adjusted["a"], 0.03)
    assert math.isclose(adjusted["c"], 0.06)
    assert math.isclose(adjusted["b"], 0.06)


def test_paired_t_p_and_dz_edge_cases():
    assert paired_t_p([0.5, 0.5], [0.5, 0.5]) == 1.0
    assert paired_t_p([0, 0, 0], [1, 1, 1]) == 0.0
    assert cohens_dz([0, 0, 0, 0], [1, 0, 1, 0]) > 0


# --- agreement --------------------------------------------------------------------------


def test_cohens_kappa_hand_computed():
    # po = 0.75 ; p(a=1) = .5, p(b=1) = .25 ; pe = .5*.25 + .5*.75 = .5 ; kappa = .25/.5
    assert math.isclose(cohens_kappa([1, 1, 0, 0], [1, 0, 0, 0]), 0.5)


def test_kappa_undefined_without_variation():
    assert cohens_kappa([0, 0, 0], [0, 0, 0]) is None


def test_weighted_kappa_penalises_distance():
    near = weighted_kappa([3, 2, 1, 0], [2, 2, 1, 0])
    far = weighted_kappa([3, 2, 1, 0], [0, 2, 1, 0])
    assert weighted_kappa([3, 2, 1, 0], [3, 2, 1, 0]) == 1.0
    assert near > far


def _bundle(tmp_path, items, docs=("a", "b", "c", "d")):
    store = LabStore(tmp_path)
    store.create_dataset(DatasetMeta(id="ds", name="DS"))
    for d in docs:
        store.put_doc("ds", 1, CorpusDoc(doc_id=d, body=f"document {d} about topic {d} with words"))
    store.replace_items("ds", 1, items)
    return store


def test_agreement_against_gold_counts_assumed_zeros(tmp_path):
    items = [GoldenItem(id="q1", query="topic a", relevant={"a": 3}, judged_nonrelevant=["b"])]
    store = _bundle(tmp_path, items)
    annotations = {"annotators": {"ann": {"q1": {"grades": {"a": 3, "b": 1, "c": 0}}}}}
    report = agreement_report(store.bundle("ds", 1), annotations, "gold", "ann")
    assert report["pairs"] == 3
    assert report["assumed_pairs"] == 1  # doc c was never judged in gold
    assert len(report["disagreements"]) == 1  # b: gold 0 vs annotator 1


# --- hashing stability --------------------------------------------------------------------


def test_new_item_fields_do_not_change_existing_hashes():
    corpus = {"a": CorpusDoc(doc_id="a", body="text")}
    item = GoldenItem(id="q1", query="q", relevant={"a": 3})
    legacy = item.model_dump(mode="json")
    legacy.pop("judged_nonrelevant")
    legacy.pop("origin")
    import hashlib

    def sha(t):
        return hashlib.sha256(t.encode("utf-8")).hexdigest()

    payload = {"corpus": sorted([["a", "en", sha("text")]]), "items": [legacy]}
    assert content_hash(corpus, [item]) == sha(
        json.dumps(payload, sort_keys=True, ensure_ascii=False)
    )
    changed = item.model_copy(update={"judged_nonrelevant": ["b"]})
    assert content_hash(corpus, [changed]) != content_hash(corpus, [item])


# --- simulations ---------------------------------------------------------------------------


def test_bootstrap_coverage_is_near_nominal_for_a_well_behaved_mean():
    population = [1.0] * 500 + [0.0] * 500
    out = coverage_sim(population, n=60, sims=200, resamples=300, confidence=0.95, seed=3)
    assert 0.88 <= out["methods"]["bootstrap"]["coverage"] <= 0.99
    assert out["methods"]["t"]["coverage"] >= out["methods"]["normal"]["coverage"] - 1e-9


def test_small_skewed_samples_produce_degenerate_intervals():
    population = [1.0] * 900 + [0.0] * 100
    out = coverage_sim(population, n=5, sims=150, resamples=200, confidence=0.95, seed=1)
    # Samples of five 1.0s give zero-width intervals that cannot cover 0.9.
    assert out["methods"]["t"]["zero_width_share"] > 0.3
    assert out["methods"]["t"]["coverage"] < 0.9


def test_power_under_the_null_is_the_false_positive_rate():
    null = power_sim(
        n=40, effect=0.0, discordance=0.3, sims=200, resamples=300, confidence=0.95, seed=2
    )
    strong = power_sim(
        n=200, effect=0.2, discordance=0.3, sims=100, resamples=300, confidence=0.95, seed=2
    )
    assert null["empirical_power"] <= 0.15
    assert strong["empirical_power"] > 0.9


def test_phacking_many_metrics_and_peeking_inflate_false_positives():
    out = phacking_sim(
        n=40,
        metrics=7,
        sims=300,
        confidence=0.95,
        discordance=0.4,
        correlation=0.0,
        batch=5,
        seed=4,
    )
    many = out["many_metrics"]
    assert many["raw_fwer"] > many["holm_fwer"]
    assert many["raw_fwer"] > 0.15
    stopping = out["optional_stopping"]
    assert stopping["peeking_fpr"] >= stopping["fixed_n_fpr"]


# --- diff ----------------------------------------------------------------------------------


def test_diff_flags_relabels_and_dev_to_holdout_moves(tmp_path):
    items = [GoldenItem(id="q1", query="topic a", relevant={"a": 3}, split="dev")]
    store = _bundle(tmp_path, items)
    store.freeze("ds", 1)
    store.fork("ds", 1)
    store.put_item(
        "ds", 2, GoldenItem(id="q1", query="topic a", relevant={"a": 2}, split="holdout")
    )
    out = diff_versions(store.bundle("ds", 1), store.bundle("ds", 2))
    assert out["relabeled"] == ["q1"]
    assert out["split_moves"] == [{"id": "q1", "from": "dev", "to": "holdout"}]
    assert not out["comparable"]
    assert any("dev → holdout" in c["title"] for c in out["consequences"])


# --- run analysis ----------------------------------------------------------------------------


def _run(results, level="document"):
    return Run(
        id="run_x",
        dataset_id="ds",
        dataset_version=1,
        dataset_hash="h",
        dataset_status="draft",
        corpus_hash="c",
        variant=VariantSpec(level=level),
        split="dev",
        k=3,
        status="done",
        results=results,
    )


def _result(item_id, docs, relevant, scores):
    ranked = [
        RankedHit(rank=i + 1, unit_id=d, doc_id=d, score=s, grade=relevant.get(d, 0))
        for i, (d, s) in enumerate(zip(docs, scores, strict=True))
    ]
    return QueryResult(
        item_id=item_id,
        query=item_id,
        query_sent=item_id,
        ranked=ranked,
        relevant_units=relevant,
        metrics={"recall@k": 1.0 if set(relevant) & set(docs[:3]) else 0.0},
    )


def test_k_sweep_recall_is_monotone_and_threshold_sweep_starves():
    run = _run([_result("q1", ["x", "y", "a", "z"], {"a": 3}, [0.9, 0.8, 0.7, 0.2])])
    recall = k_sweep(run, 4)["series"]["recall"]
    assert recall == sorted(recall) and recall[1] == 0.0 and recall[2] == 1.0
    points = threshold_sweep(run, [0.0, 0.75, 0.95])["points"]
    assert (
        points[0]["recall"] == 1.0
        and points[1]["recall"] == 0.0
        and points[2]["empty_share"] == 1.0
    )


def test_failure_analysis_suggests_label_gap_and_depth_miss(tmp_path):
    items = [
        GoldenItem(id="q1", query="topic a", relevant={"a": 3, "d": 2}, judged_nonrelevant=["b"])
    ]
    store = _bundle(tmp_path, items)
    run = _run([_result("q1", ["c", "b", "x"], {"a": 3, "d": 2}, [0.9, 0.8, 0.7])])
    out = failure_analysis(run, store.bundle("ds", 1), {}, "recall@k", 1.0)
    suggested = out["failures"][0]["suggested"]
    assert "not_retrieved" in suggested and "label_gap" in suggested


# --- synthetic generation ----------------------------------------------------------------------


class FakeGenerator:
    model = "fake-llm"

    def generate(self, text, lang, style, n, temperature, seed):
        sentence = text.split(".")[0].strip() + "."
        return [
            {
                "query": "How many days of annual leave do employees get",
                "evidence": sentence,
                "answer": "21",
            },
            {
                "query": "paraphrased vacation entitlement?",
                "evidence": "not in the text at all",
                "answer": "x",
            },
        ][:n]


def test_generate_candidates_flags_copied_and_missing_evidence(tmp_path):
    store = LabStore(tmp_path)
    store.create_dataset(DatasetMeta(id="ds", name="DS"))
    store.put_doc(
        "ds",
        1,
        CorpusDoc(
            doc_id="leave", body="How many days of annual leave do employees get. Twenty one."
        ),
    )
    out = generate_candidates(FakeGenerator(), store.bundle("ds", 1), None, 2, "user", 0.5, 1, [])
    first, second = out["created"]
    assert first["evidence_found"] and first["copied"] and first["lexical_overlap"] == 1.0
    assert not second["evidence_found"] and second["lexical_overlap"] < 0.5
    assert lexical_overlap("vacation entitlement", "annual leave") == 0.0


# --- learning path --------------------------------------------------------------------------------


def test_learning_path_auto_missions_cannot_be_self_reported(tmp_path):
    store = LabStore(tmp_path)
    before = evaluate(store)
    assert before["done"] == 0 and before["total"] > 10
    entry = update_progress(store, "dc-scale", True, "x" * 200)
    assert entry["done"] is False
    try:
        update_progress(store, "st-coverage", True, "too short")
        raise AssertionError("short reflection accepted")
    except ValueError:
        pass
    update_progress(store, "st-coverage", True, "The t interval collapses to zero width " * 4)
    assert evaluate(store)["done"] == 1


# --- API: pooling, judging, candidates --------------------------------------------------------------


class FakeKnowHub:
    base_url = "fake://know-hub"

    def __init__(self):
        self.docs = {}

    def ready(self):
        return True

    def grpc_ready(self):
        return True

    def list_titles(self, owner):
        return {t: v[0] for t, v in self.docs.get(owner, {}).items()}

    def upload(self, owner, doc_id, body):
        self.docs.setdefault(owner, {})[doc_id] = (str(uuid.uuid4()), body)
        return self.docs[owner][doc_id][0]

    def embedding_status(self, owner, document_id):
        return "INDEXED"

    def _rank(self, query, owner):
        q = set(query.lower().split())
        return sorted(
            self.docs.get(owner, {}).items(),
            key=lambda kv: (-len(q & set(kv[1][1].lower().split())), kv[0]),
        )

    def search_documents(self, query, owner, limit, hybrid, min_score):
        return [
            RawHit(doc_id=t, document_id=v[0], score=0.5, text=v[1])
            for t, v in self._rank(query, owner)
        ][:limit]

    def search_chunks(self, query, owner, top_k, min_score):
        return [
            RawHit(
                doc_id=t,
                document_id=v[0],
                chunk_id=f"{t}-0",
                score=0.5,
                text=chunk_text(v[1])[0].content,
            )
            for t, v in self._rank(query, owner)
        ][:top_k]


def _wait(client, job):
    import time

    for _ in range(300):
        state = client.get(f"/api/eval/jobs/{job['id']}").json()
        if state["status"] != "running":
            assert state["status"] == "done", state
            return state
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_api_pool_judge_audit_and_candidates(tmp_path):
    store = LabStore(tmp_path)
    client = TestClient(create_app(store=store, knowhub=FakeKnowHub()))
    base = "/api/eval/datasets/ds/versions/1"
    client.post("/api/eval/datasets", json={"name": "DS", "id": "ds"})
    for d, body in {
        "leave": "annual leave twenty one days",
        "refund": "refund within thirty days",
        "lease": "lease notice sixty days",
        "travel": "travel receipts reimbursed",
    }.items():
        client.put(f"{base}/corpus/{d}", json={"doc_id": d, "body": body})
    client.put(
        f"{base}/items/q1",
        json={"id": "q1", "query": "annual leave days", "relevant": {"leave": 3}},
    )
    _wait(client, client.post(f"{base}/index", json={"normalization": "off"}).json())

    _wait(client, client.post(f"{base}/pool", json={"depth": 3}).json())
    pool = client.get(f"{base}/pool").json()
    assert pool["exists"] and pool["summary"]["pooled_pairs"] >= 3
    unjudged = [d["doc_id"] for d in pool["items"][0]["docs"] if d["judgement"] == "unjudged"]
    judged = client.post(
        f"{base}/judge", json={"item_id": "q1", "doc_id": unjudged[0], "grade": 0}
    ).json()
    assert unjudged[0] in judged["items"][0]["judged_nonrelevant"]

    task = client.get(
        f"{base}/annotations/task", params={"annotator": "second", "sample": 1}
    ).json()
    assert "relevant" not in json.dumps(task["items"])  # blind: no gold labels leak
    grades = {d["doc_id"]: (3 if d["doc_id"] == "leave" else 0) for d in task["items"][0]["docs"]}
    client.put(f"{base}/annotations/second/q1", json={"grades": grades})
    report = client.get(f"{base}/agreement", params={"a": "gold", "b": "second"}).json()
    assert report["pairs"] == len(grades) and report["percent_agreement"] is not None

    store.save_artifact(
        "ds",
        1,
        "candidates",
        [
            {
                "id": "cand-1",
                "doc_id": "refund",
                "chunk_index": 0,
                "query": "how long to get a refund",
                "evidence": "refund within thirty days",
                "answer": "30 days",
                "difficulty": "easy",
                "style": "user",
                "model": "fake",
                "prompt_version": "gen-v1",
                "temperature": 0.7,
                "seed": 1,
                "evidence_found": True,
                "lexical_overlap": 0.5,
                "copied": False,
                "status": "pending",
                "reject_reason": "",
                "item_id": None,
                "created_at": "now",
            }
        ],
    )
    accepted = client.post(f"{base}/candidates/cand-1/accept", json={}).json()
    syn = next(i for i in accepted["items"] if i["origin"] == "synthetic")
    assert (
        syn["relevant"] == {"refund": 3}
        and syn["evidence"][0]["text"] == "refund within thirty days"
    )
    assert (
        client.post(f"{base}/candidates/cand-1/reject", json={"reason": "dup"}).status_code == 409
    )

    readiness = client.get(f"{base}/readiness").json()
    assert any(c["id"] == "labels.synthetic_share" for c in readiness["checks"])


def test_simulate_labeling_and_auditing(tmp_path):
    store = LabStore(tmp_path)
    client = TestClient(create_app(store=store, knowhub=FakeKnowHub()))
    base = "/api/eval/datasets/ds/versions/1"
    client.post("/api/eval/datasets", json={"name": "DS", "id": "ds"})
    for d, body in {
        "leave": "annual leave twenty one days",
        "refund": "refund within thirty days",
        "lease": "lease notice sixty days",
    }.items():
        client.put(f"{base}/corpus/{d}", json={"doc_id": d, "body": body})
    client.put(
        f"{base}/items/q1",
        json={"id": "q1", "query": "annual leave days", "relevant": {"leave": 3}},
    )
    _wait(client, client.post(f"{base}/index", json={"normalization": "off"}).json())
    _wait(client, client.post(f"{base}/pool", json={"depth": 3}).json())

    assert client.post(f"{base}/judge/simulate", json={}).status_code == 200
    pool = client.get(f"{base}/pool").json()
    assert pool["summary"]["judged_share"] == 1.0
    assert all(d["judgement"] != "unjudged" for r in pool["items"] for d in r["docs"])
    assert client.post(f"{base}/judge/simulate", json={}).status_code == 409

    sim = client.post(
        f"{base}/annotations/simulate",
        json={"annotator": "sim-bot", "sample": 5, "agreement": 0.8, "seed": 1},
    ).json()
    assert sim["annotated"] >= 1 and sim["simulated"] is True
    report = client.get(f"{base}/agreement", params={"a": "gold", "b": "sim-bot"}).json()
    assert report["pairs"] > 0
    assert any("simulated" in n.lower() or "not judged" in n.lower() for n in report["notes"])
    assert client.post(
        f"{base}/annotations/simulate",
        json={"annotator": "gold", "sample": 1},
    ).status_code == 422
    renamed = client.patch(f"{base}/annotations/sim-bot", json={"name": "sara"}).json()
    assert "sara" in renamed["annotators"] and "sim-bot" not in renamed["annotators"]
    assert client.get(f"{base}/agreement", params={"a": "gold", "b": "sara"}).status_code == 200
