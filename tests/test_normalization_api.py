"""The browser workflow uses the real controlled retriever, with no know-hub or LLM."""

import time

import pytest
from fastapi.testclient import TestClient

from agent_hub.eval.lab.api import create_app
from agent_hub.eval.lab.store import LabStore


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(store=LabStore(tmp_path)))


def fixture_bundle():
    return {
        "id": "diagnostic-fixture",
        "name": "Arabic API fixture",
        "source": "Original test text",
        "scope": "diagnostic",
        "corpus": [
            {"passage_id": "123#0", "title": "", "text": "أحمد يكتب"},
            {"passage_id": "123#1", "title": "", "text": "مريم تقرأ"},
        ],
        "queries": [
            {"qid": "q#1", "query": "احمد", "tags": ["named_entity"]},
            {"qid": "q#2", "query": "مريم", "tags": []},
        ],
        "qrels": [
            {"qid": "q#1", "passage_id": "123#0", "relevance": 1},
            {"qid": "q#1", "passage_id": "123#1", "relevance": 0},
            {"qid": "q#2", "passage_id": "123#1", "relevance": 1},
        ],
    }


def wait_for_job(client, job_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = client.get(f"/api/eval/jobs/{job_id}").json()
        if job["status"] != "running":
            assert job["status"] == "done", job
            return job["result"]["run_id"]
        time.sleep(0.02)
    pytest.fail("small normalization job did not finish")


def test_import_run_inspect_export_and_reload(client, tmp_path):
    base = "/api/eval/normalization"
    opts = client.get(f"{base}/options").json()
    assert "raw" in {preset["id"] for preset in opts["presets"]}
    assert "recall@100" in opts["metrics"]
    response = client.post(f"{base}/datasets/import", json=fixture_bundle())
    assert response.status_code == 201, response.text
    dataset = response.json()
    assert dataset["document_count"] == 2
    assert dataset["provenance"]["scope"] == "diagnostic"
    assert dataset["provenance"]["benchmark_comparable"] is False
    response = client.post(
        f"{base}/runs",
        json={
            "dataset_id": dataset["dataset_id"],
            "config": {"presets": ["raw", "alef"], "bootstrap_samples": 100},
        },
    )
    assert response.status_code == 202, response.text
    run_id = wait_for_job(client, response.json()["id"])
    report = client.get(f"{base}/runs/{run_id}").json()
    assert all("per_query" not in c for c in report["conditions"])
    raw, alef = report["conditions"]
    assert raw["metrics"]["ndcg@10"] == 0.5
    assert alef["metrics"]["ndcg@10"] == 1
    # An unchanged query can benefit from changed corpus text.
    assert alef["changed_queries"] == 0
    assert alef["changed_documents"] == 2
    assert alef["comparison_to_raw"]["ndcg@10"]["wins"] == 1
    page = client.get(
        f"{base}/runs/{run_id}/queries",
        params={
            "condition": 1,
            "qid": "q#1",
            "ranking_limit": 1,
        },
    ).json()
    assert page["total"] == 1
    assert page["items"][0]["ranking"][0]["passage_id"] == "123#0"
    passage = client.get(
        f"{base}/runs/{run_id}/passage",
        params={
            "condition": 1,
            "passage_id": "123#0",
        },
    ).json()
    assert passage["retrieval_input"] == "أحمد يكتب"
    assert passage["normalized_input"] == "احمد يكتب"
    assert passage["preset"] == "alef"
    assert (
        client.get(
            f"{base}/runs/{run_id}/passage",
            params={
                "passage_id": "missing",
            },
        ).status_code
        == 404
    )
    assert client.get(f"{base}/runs/{run_id}/queries?condition=99").status_code == 422
    export = client.get(f"{base}/runs/{run_id}/export")
    assert "attachment" in export.headers["content-disposition"]
    assert len(export.json()["conditions"][0]["per_query"]) == 2
    # Results survive app restarts, unlike the transient job registry.
    reloaded = TestClient(create_app(store=LabStore(tmp_path)))
    runs = reloaded.get(f"{base}/runs", params={"dataset_id": dataset["dataset_id"]}).json()
    assert len(runs) == 1 and runs[0]["run_id"] == run_id
    assert reloaded.get(f"{base}/datasets").json()[0]["dataset_id"] == dataset["dataset_id"]


def test_rejects_bad_qrels_and_duplicate_passages(client):
    bundle = fixture_bundle()
    bundle["qrels"][0]["passage_id"] = "missing"
    assert client.post("/api/eval/normalization/datasets/import", json=bundle).status_code == 422
    bundle = fixture_bundle()
    bundle["corpus"].append(bundle["corpus"][0])
    assert client.post("/api/eval/normalization/datasets/import", json=bundle).status_code == 422
    assert client.get("/api/eval/normalization/datasets").json() == []


def test_rejects_bad_config_missing_pins_and_unknown_ids(client):
    base = "/api/eval/normalization"
    dataset = client.post(f"{base}/datasets/import", json=fixture_bundle()).json()
    for config in (
        {"top_k": 10},
        {"bootstrap_samples": 0},
        {"presets": ["typo"]},
        {"dense_revision": "main"},
        {"retrievers": ["dense"]},
        {"misspelled_config": True},
    ):
        response = client.post(
            f"{base}/runs",
            json={
                "dataset_id": dataset["dataset_id"],
                "config": config,
            },
        )
        assert response.status_code == 422, response.text
    assert client.get(f"{base}/runs/not-a-safe-id").status_code == 422
    assert client.get(f"{base}/runs/{'0' * 32}").status_code == 404
    assert client.post(f"{base}/runs", json={"dataset_id": "../escape"}).status_code == 422
