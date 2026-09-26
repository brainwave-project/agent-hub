"""Generation half of the Eval Lab — metrics, pipeline attribution, Judge Lab, release gate.

No stack: know-hub is the word-overlap fake from test_eval_lab, and the LLM is a
rule-based fake that answers each prompt deterministically. The fake is dumb on
purpose — these tests check bookkeeping, formulas and guards, not model quality.
"""

import json
import re
import time

import pytest
from fastapi.testclient import TestClient
from test_eval_lab import FakeKnowHub

from agent_hub.eval.lab import genmetrics as gm
from agent_hub.eval.lab import judgelab as jl
from agent_hub.eval.lab import routes_gate
from agent_hub.eval.lab.api import create_app
from agent_hub.eval.lab.gate_cli import main as gate_main
from agent_hub.eval.lab.genmodels import (
    GatePolicy,
    GenItemResult,
    GenRun,
    HumanLabel,
    JudgeSample,
    JudgeSet,
    JudgeSpec,
    PipelineSpec,
)
from agent_hub.eval.lab.models import CorpusDoc, DatasetMeta, Evidence, GoldenItem
from agent_hub.eval.lab.pipeline import fault_matrix, run_generation
from agent_hub.eval.lab.release_gate import default_policy, evaluate_gate
from agent_hub.eval.lab.runner import index_corpus
from agent_hub.eval.lab.store import LabStore

WORD = re.compile(r"\w+")


def words(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if len(w) >= 3}


def covered(claim: str, text: str, share: float = 0.8) -> bool:
    c = words(claim)
    return bool(c) and len(c & words(text)) / len(c) >= share


def section(user: str, name: str) -> str:
    """Text after 'NAME:\\n' up to the next ALL-CAPS header."""
    m = re.search(rf"(?:^|\n){name}:\n(.*?)(?:\n\n[A-Z][A-Z ]+:\n|\Z)", user, re.S)
    return m.group(1) if m else ""


class FakeLLM:
    """Deterministic stand-in for Ollama. Each branch mimics one prompt's contract."""

    def __init__(self, pairwise_bias: str = "A", verbose_bonus: bool = False) -> None:
        self.calls = 0
        self.pairwise_bias = pairwise_bias
        self.verbose_bonus = verbose_bonus

    def models(self):
        return ["fake-model"]

    def embed(self, texts, model="x"):
        vocab = sorted({w for t in texts for w in words(t)})
        return [[1.0 if w in words(t) else 0.0 for w in vocab] for t in texts]

    def chat_text(self, system, user, **_):
        self.calls += 1
        context = section(user, "CONTEXT")
        if "general knowledge" in system:  # lenient / generation fault
            return "The limit is 999 euros." if "no context" in context else context.split("\n")[-1]
        if "no context" in context:
            return "I don't have enough information in the available documents to answer that confidently."
        first = [ln for ln in context.split("\n") if ln and not ln.startswith("[")]
        return first[0] if first else "I don't have enough information in the available documents."

    def chat_json(self, system, user, **_):
        self.calls += 1
        if "atomic factual claims" in system:
            answer = section(user, "ANSWER")
            if gm.is_abstention(answer):
                return {"claims": []}
            return {"claims": [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]}
        if "verify whether ONE claim" in system:
            return {
                "supported": covered(section(user, "CLAIM"), section(user, "CONTEXT")),
                "reason": "fake",
            }
        if "addresses the question" in system:
            q, a = section(user, "QUESTION"), section(user, "ANSWER")
            return {"score": 5 if words(q) & words(a) else 2, "reason": "fake"}
        if "questions it would be a direct reply to" in system:
            a = section(user, "ANSWER")
            return {"questions": [a], "noncommittal": gm.is_abstention(a)}
        if "ONE passage of retrieved context" in system:
            return {
                "useful": covered(section(user, "REFERENCE ANSWER"), section(user, "PASSAGE"), 0.5)
            }
        if "explicitly states this statement" in system:
            return {
                "attributed": covered(section(user, "STATEMENT"), section(user, "CONTEXT"), 0.6)
            }
        if "ONE claim from an answer" in system:
            return {
                "supported": covered(section(user, "CLAIM"), section(user, "REFERENCE ANSWER"), 0.5)
            }
        if "conveys this statement" in system:
            return {"covered": covered(section(user, "STATEMENT"), section(user, "ANSWER"), 0.5)}
        if "grade an answer against a reference" in system:
            ref, ans = section(user, "REFERENCE ANSWER"), section(user, "ANSWER")
            nums = set(re.findall(r"\d+", ref))
            ok = (not gm.is_abstention(ans)) and (
                nums <= set(re.findall(r"\d+", ans)) if nums else covered(ref, ans, 0.5)
            )
            return {"correct": ok}
        if "relevance grader" in system:
            return {
                "relevant": bool(
                    words(section(user, "QUESTION")) & words(section(user, "CONTEXT"))
                ),
                "reason": "fake",
            }
        if "fully supported by the provided context" in system:
            return {
                "grounded": covered(section(user, "ANSWER"), section(user, "CONTEXT")),
                "reason": "fake",
            }
        if "Compare two answers" in system:
            return {"winner": self.pairwise_bias, "reasoning": "fake"}
        if "evaluation judge" in system:
            answer = user.split("ANSWER:\n")[-1]
            ref = section(user, "REFERENCE ANSWER")
            nums = set(re.findall(r"\d+", ref))
            score = 5 if nums and nums <= set(re.findall(r"\d+", answer)) else 2
            if self.verbose_bonus and len(answer) > 200:
                score = min(5, score + 2)
            if '"verdict"' in system:
                return {"verdict": "pass" if score >= 4 else "fail", "reasoning": "fake"}
            return {"score": score, "reasoning": "fake"}
        if "Grade how well a passage" in system:
            return {
                "grade": 3
                if words(section(user, "QUESTION")) & words(section(user, "PASSAGE"))
                else 0
            }
        raise AssertionError(f"unexpected prompt: {system[:80]}")


# --- metrics --------------------------------------------------------------------------------


def judge() -> gm.MetricJudge:
    return gm.MetricJudge(FakeLLM(), model="fake-model")


def test_average_precision_hand_computed():
    assert gm.average_precision([False, False, False, True])[0] == pytest.approx(0.25)
    assert gm.average_precision([True, False])[0] == 1.0
    assert gm.average_precision([False, True])[0] == pytest.approx(0.5)
    # (1/1 + 2/3) / 2
    assert gm.average_precision([True, False, True])[0] == pytest.approx((1 + 2 / 3) / 2)
    assert gm.average_precision([False, False])[0] == 0.0


def test_faithfulness_is_blind_to_a_stale_context():
    stale = "Business travel meals are reimbursed up to a daily limit of 45 euros."
    res = gm.faithfulness(
        judge(), "Business travel meals are reimbursed up to 45 euros daily.", [stale]
    )
    assert res.score == 1.0  # faithful to a wrong document
    wrong = gm.answer_correctness(
        judge(),
        "q",
        "Business travel meals are reimbursed up to 45 euros daily.",
        "60 euros per day.",
    )
    assert wrong.score < 0.5


def test_faithfulness_vacuous_on_abstention_and_partial_on_hallucination():
    j = judge()
    res = gm.faithfulness(
        j, "I don't have enough information in the available documents.", ["x y z"]
    )
    assert res.score == 1.0 and "vacuous" in res.flags and "abstention" in res.flags
    ctx = ["Full-time employees receive 25 days of paid annual leave per calendar year."]
    half = gm.faithfulness(
        j,
        "Full-time employees receive 25 days of paid annual leave. Staff also get a birthday day off.",
        ctx,
    )
    assert half.score == pytest.approx(0.5)
    assert [s["supported"] for s in half.steps].count(False) == 1


def test_context_precision_rewards_rank():
    ref = "25 days of paid annual leave per year"
    good = "Full-time employees receive 25 days of paid annual leave per year."
    noise = ["Hotels cost 150 euros per night.", "Book a desk in the app.", "Gifts under 50 euros."]
    late = gm.context_precision(judge(), "holiday?", [*noise, good], ref)
    early = gm.context_precision(judge(), "holiday?", [good, *noise], ref)
    assert late.score == pytest.approx(0.25) and early.score == 1.0


def test_label_based_context_metrics():
    ev = [("leave", "twenty one days"), ("refund", "thirty days")]
    ctx = [
        ("leave", "Employees get twenty one days of annual leave."),
        ("lease", "sixty days notice"),
    ]
    recall = gm.context_recall_labels(ctx, ev, {"leave": 3, "refund": 2})
    assert recall.score == pytest.approx(0.5)
    precision = gm.context_precision_labels(ctx, ev, {"leave": 3, "refund": 2})
    assert precision.score == 1.0  # the only useful chunk is at rank 1
    assert gm.context_recall_labels(ctx, [], {}).score is None  # unanswerable


def test_split_statements_and_abstention_regex():
    assert gm.split_statements("A is 1. B is 2! C is 3?") == ["A is 1.", "B is 2!", "C is 3?"]
    assert gm.is_abstention(
        "The context does not contain the answer. I don't have enough information."
    )
    assert not gm.is_abstention("Yes, up to 20 days.")


def test_judge_fingerprint_moves_with_model():
    a = gm.MetricJudge(FakeLLM(), model="m1")
    b = gm.MetricJudge(FakeLLM(), model="m2")
    assert a.fingerprint != b.fingerprint
    assert a.fingerprint == gm.MetricJudge(FakeLLM(), model="m1").fingerprint


# --- pipeline & attribution ----------------------------------------------------------------------


def seeded(tmp_path):
    store = LabStore(tmp_path)
    store.create_dataset(DatasetMeta(id="demo", name="Demo"))
    docs = {
        "leave": "Employees get twenty one days of annual leave each calendar year.",
        "refund": "Refunds are accepted within thirty days of purchase with a receipt.",
        "lease": "The office lease requires sixty days written notice before moving.",
        "parking": "Parking fines are never reimbursed by the company under any policy.",
    }
    for d, b in docs.items():
        store.put_doc("demo", 1, CorpusDoc(doc_id=d, body=b))
    items = [
        GoldenItem(
            id="q1",
            query="how many days of annual leave do employees get",
            relevant={"leave": 3},
            evidence=[Evidence(doc_id="leave", text="twenty one days of annual leave")],
            ground_truth="twenty one days of annual leave",
            split="dev",
        ),
        GoldenItem(
            id="q2",
            query="within how many days are refunds accepted",
            relevant={"refund": 3},
            evidence=[Evidence(doc_id="refund", text="within thirty days of purchase")],
            ground_truth="within thirty days of purchase",
            split="dev",
        ),
        GoldenItem(
            id="q3",
            query="how much notice does the office lease require",
            relevant={"lease": 3},
            evidence=[Evidence(doc_id="lease", text="sixty days written notice")],
            ground_truth="sixty days written notice",
            split="dev",
        ),
        GoldenItem(
            id="q4",
            query="is there a company pet policy for dogs",
            relevant={},
            split="dev",
            ground_truth="Not covered.",
        ),
    ]
    store.replace_items("demo", 1, items)
    kh = FakeKnowHub()
    index_corpus(kh, store, store.bundle("demo", 1), "off", poll_interval_s=0)
    return store, kh


def gen(store, kh, **spec):
    bundle = store.bundle("demo", 1)
    llm = FakeLLM()
    return run_generation(
        kh,
        llm,
        store,
        bundle,
        PipelineSpec(candidate_k=3, top_n=2, **spec),
        "dev",
        gm.MetricJudge(llm, model="fake-model"),
    )


def test_clean_run_blames_nothing_and_abstains_on_unanswerable(tmp_path):
    store, kh = seeded(tmp_path)
    run = gen(store, kh)
    by_id = {r.item_id: r for r in run.results}
    assert all(by_id[i].outcome == "correct" for i in ("q1", "q2", "q3")), [
        r.outcome for r in run.results
    ]
    assert {by_id[i].blame for i in ("q1", "q2", "q3")} == {"none"}
    assert run.components["funnel"][0]["n"] == 3
    assert run.aggregates["answer_ok"]["n"] == 4
    assert store.get_gen_run(run.id).status == "done"


@pytest.mark.parametrize("fault", ["ingestion", "retrieval", "rerank", "grade", "generation"])
def test_each_injected_fault_is_blamed_on_its_stage(tmp_path, fault):
    store, kh = seeded(tmp_path)
    run = gen(store, kh, fault=fault)
    answerable = [r for r in run.results if r.answerable]
    failed = [r for r in answerable if r.outcome != "correct"]
    assert failed, f"fault {fault} broke nothing"
    assert {r.blame for r in failed} == {fault}, [(r.item_id, r.blame, r.outcome) for r in failed]
    matrix = fault_matrix([run])
    assert matrix["rows"][0]["accuracy"] == 1.0


def test_downstream_stages_are_skipped_not_failed(tmp_path):
    store, kh = seeded(tmp_path)
    run = gen(store, kh, fault="ingestion")
    r = next(x for x in run.results if x.item_id == "q1")
    status = {s.stage: s.status for s in r.stages}
    assert status["ingestion"] == "fail"
    assert status["retrieval"] == "skip" and status["rerank"] == "skip"


# --- judge lab -------------------------------------------------------------------------------------


def judge_set() -> JudgeSet:
    def s(i, q, a, ref, kind, verdict, score):
        return JudgeSample(
            id=i,
            question=q,
            answer=a,
            reference=ref,
            kind=kind,
            labels={"me": HumanLabel(verdict=verdict, score=score)},
        )

    return JudgeSet(
        id="cal",
        name="cal",
        samples=[
            s("a1", "meal limit?", "60 euros per day.", "60 euros per day.", "correct", "pass", 5),
            s("a2", "meal limit?", "45 euros per day.", "60 euros per day.", "stale", "fail", 1),
            s(
                "b1",
                "hotel zurich?",
                "220 euros per night.",
                "220 euros per night.",
                "correct",
                "pass",
                5,
            ),
            s(
                "b2",
                "hotel zurich?",
                "It depends on the city.",
                "220 euros per night.",
                "evasive",
                "fail",
                2,
            ),
            s(
                "c1",
                "leave days?",
                "25 days plus public holidays.",
                "25 days.",
                "correct",
                "pass",
                4,
            ),
            s(
                "c2",
                "leave days?",
                "25 days plus a birthday day 1.",
                "25 days.",
                "hallucination",
                "fail",
                2,
            ),
        ],
    )


def spec(**kw) -> JudgeSpec:
    return JudgeSpec(
        id="j", name="j", rubric="Score factual agreement with the reference.", model="fake", **kw
    )


def test_validation_report_confusion_and_rates():
    js = judge_set()
    run = jl.run_judge(FakeLLM(), spec(), js)
    report = jl.validation_report(js, run, "me")
    # Fake judge passes whenever the reference numbers appear: c2 ("25 ... 1") passes → one false pass.
    assert report["n"] == 6
    assert report["confusion"]["matrix"] == [[3, 0], [1, 2]]
    assert report["fail_recall"] == pytest.approx(2 / 3)
    assert report["pass_rate_gap"] == pytest.approx(4 / 6 - 3 / 6)
    assert [d["sample_id"] for d in report["disagreements"]] == ["c2"]


def test_position_probe_detects_a_judge_that_always_picks_A():
    js = judge_set()
    run = jl.position_probe(FakeLLM(pairwise_bias="A"), spec(), js, rater="me", identical=2)
    assert run.summary["first_position_rate"] == 1.0
    assert run.summary["consistency"] == 0.0  # the pick flips every time the order flips
    assert run.summary["identical_non_tie_rate"] == 1.0
    fair = jl.position_probe(FakeLLM(pairwise_bias="tie"), spec(), js, rater="me", identical=2)
    assert fair.summary["consistency"] == 1.0


def test_verbosity_probe_measures_a_length_bonus():
    js = judge_set()
    biased = jl.verbosity_probe(
        FakeLLM(verbose_bonus=True), spec(), js, padding="filler", rater="me"
    )
    assert biased.summary["mean_delta"] > 0
    assert "a2" in biased.summary["fail_to_pass"]
    neutral = jl.verbosity_probe(FakeLLM(), spec(), js, padding="filler")
    assert neutral.summary["mean_delta"] == 0


def test_drift_report_and_spearman():
    js = judge_set()
    a = jl.run_judge(FakeLLM(), spec(), js)
    b = jl.run_judge(FakeLLM(verbose_bonus=True), spec(pass_threshold=2), js)
    report = jl.drift_report(js, a, b, rater="me")
    assert "pass_threshold" in report["changed_fields"] and not report["same_fingerprint"]
    assert report["pass_rate_b"] >= report["pass_rate_a"]
    assert jl.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert jl.spearman([1, 1, 1], [1, 2, 3]) is None


# --- release gate -----------------------------------------------------------------------------------


def fake_run(run_id, ok, faith, fingerprint="fp", status="frozen", n=24):
    results = []
    for i in range(n):
        results.append(
            GenItemResult(
                item_id=f"q{i}",
                query="q",
                answerable=True,
                metrics={
                    "answer_ok": ok[i % len(ok)],
                    "faithfulness": faith,
                    "false_abstention": 0.0,
                    "answered_unanswerable": 0.0,
                    "answer_relevance": 1.0,
                },
            )
        )
    return GenRun(
        id=run_id,
        dataset_id="d",
        dataset_version=1,
        dataset_hash="h",
        dataset_status=status,
        corpus_hash="c",
        spec=PipelineSpec(),
        split="dev",
        judge_fingerprint=fingerprint,
        status="done",
        results=results,
    )


def test_gate_blocks_on_judge_change_and_passes_equal_runs():
    policy = default_policy()
    base = fake_run("b", [1.0], 0.95)
    same = evaluate_gate(policy, base, fake_run("c", [1.0], 0.95))
    assert same["decision"] == "pass", same["reasons"]
    drifted = evaluate_gate(policy, base, fake_run("c", [1.0], 0.95, fingerprint="other"))
    assert drifted["decision"] == "block"
    assert any("Same judge" in r for r in drifted["reasons"])


def test_gate_floor_and_noise_aware_rules():
    policy = default_policy()
    base = fake_run("b", [1.0], 0.95)
    low_faith = evaluate_gate(policy, base, fake_run("c", [1.0], 0.5))
    assert low_faith["decision"] == "block"
    # One answer in four now wrong: Δ answer_ok = −0.25, clearly beyond the 0.10 margin.
    worse = evaluate_gate(policy, base, fake_run("c", [1.0, 1.0, 1.0, 0.0], 0.95))
    rule = next(r for r in worse["rules"] if r["id"] == "answer-ok-noninferior")
    assert rule["status"] == "fail" and worse["decision"] == "block"


def test_lower_is_better_metrics_are_oriented():
    policy = GatePolicy(
        id="p",
        name="p",
        require_frozen_dataset=False,
        min_items=5,
        rules=[{"id": "fa", "metric": "false_abstention", "kind": "max_drop", "threshold": 0.05}],
    )
    base = fake_run("b", [1.0], 1.0)
    cand = fake_run("c", [1.0], 1.0)
    for r in cand.results[:6]:
        r.metrics["false_abstention"] = 1.0  # MORE abstentions = worse
    result = evaluate_gate(policy, base, cand)
    rule = result["rules"][0]
    assert rule["delta"]["mean"] < 0 and rule["status"] == "fail"


def test_gate_cli_exit_codes(tmp_path):
    policy = default_policy()
    (tmp_path / "policy.json").write_text(policy.model_dump_json(), encoding="utf-8")
    (tmp_path / "base.json").write_text(
        fake_run("b", [1.0], 0.95).model_dump_json(), encoding="utf-8"
    )
    (tmp_path / "good.json").write_text(
        fake_run("c", [1.0], 0.95).model_dump_json(), encoding="utf-8"
    )
    (tmp_path / "bad.json").write_text(
        fake_run("c", [1.0], 0.4).model_dump_json(), encoding="utf-8"
    )
    args = ["--policy", str(tmp_path / "policy.json"), "--baseline", str(tmp_path / "base.json")]
    summary = tmp_path / "summary.md"
    assert (
        gate_main([*args, "--candidate", str(tmp_path / "good.json"), "--summary", str(summary)])
        == 0
    )
    assert "PASS" in summary.read_text(encoding="utf-8")
    assert gate_main([*args, "--candidate", str(tmp_path / "bad.json")]) == 1
    assert gate_main([*args, "--candidate", str(tmp_path / "missing.json")]) == 2


# --- API ---------------------------------------------------------------------------------------------


def wait(client, job):
    for _ in range(200):
        state = client.get(f"/api/eval/jobs/{job['id']}").json()
        if state["status"] != "running":
            assert state["status"] == "done", state
            return state["result"]
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_api_generation_judges_and_gate(tmp_path, monkeypatch):
    store, kh = seeded(tmp_path)
    store.freeze("demo", 1)
    client = TestClient(create_app(store, kh, FakeLLM()))

    cfg = client.get("/api/eval/gen/config").json()
    assert "rerank" in cfg["faults"] and cfg["models"] == ["fake-model"]
    assert client.get("/api/eval/gen/scenarios").json()

    pg = client.post(
        "/api/eval/gen/metrics/playground",
        json={
            "question": "leave days?",
            "contexts": ["Employees get twenty one days of annual leave."],
            "answer": "Employees get twenty one days of annual leave.",
            "reference": "twenty one days of annual leave",
        },
    ).json()
    assert pg["results"]["faithfulness"]["score"] == 1.0 and pg["calls"] > 0

    body = {
        "dataset_id": "demo",
        "version": 1,
        "spec": {"candidate_k": 3, "top_n": 2},
        "split": "dev",
    }
    assert client.post("/api/eval/gen/runs", json={**body, "split": "holdout"}).status_code == 409
    base_id = wait(client, client.post("/api/eval/gen/runs", json=body).json())["run_id"]
    bad = {**body, "spec": {"candidate_k": 3, "top_n": 2, "fault": "generation", "name": "broken"}}
    cand_id = wait(client, client.post("/api/eval/gen/runs", json=bad).json())["run_id"]
    cmp = client.post("/api/eval/gen/compare", json={"run_a": base_id, "run_b": cand_id}).json()
    assert any(m["metric"] == "answer_ok" and m["delta"] < 0 for m in cmp["metrics"])

    # judges: versioning
    j = client.post(
        "/api/eval/judges",
        json={"name": "Correctness", "rubric": "Score agreement with the reference."},
    ).json()
    assert j["version"] == 1
    j2 = client.put(
        f"/api/eval/judges/{j['id']}",
        json={"name": "Correctness", "rubric": "A different rubric text."},
    ).json()
    assert (
        j2["version"] == 2
        and j2["history"][0]["version"] == 1
        and j2["fingerprint"] != j["fingerprint"]
    )

    lines = "\n".join(json.dumps(s.model_dump()) for s in judge_set().samples)
    client.post("/api/eval/judge-sets", json={"name": "cal", "id": "cal", "jsonl": lines})
    task = client.get("/api/eval/judge-sets/cal/task", params={"rater": "you"}).json()
    assert task["remaining"] == 6 and "kind" not in task["next"][0]
    client.put(
        f"/api/eval/judge-sets/cal/labels/you/{task['next'][0]['id']}",
        json={"verdict": "pass", "score": 5},
    )
    jr = wait(
        client,
        client.post("/api/eval/judge-runs", json={"judge_id": j["id"], "set_id": "cal"}).json(),
    )
    val = client.get(
        "/api/eval/judge-sets/cal/validation", params={"run_id": jr["judge_run_id"], "rater": "me"}
    ).json()
    assert val["n"] == 6

    # gate
    policy = client.get("/api/eval/gate/policies").json()[0]
    policy["min_items"] = 2
    client.put(f"/api/eval/gate/policies/{policy['id']}", json=policy)
    res = client.post(
        "/api/eval/gate/evaluate",
        json={"policy_id": policy["id"], "baseline_run": base_id, "candidate_run": cand_id},
    ).json()
    assert res["decision"] == "block"
    assert (
        client.post(
            "/api/eval/gate/decisions",
            json={
                "policy_id": policy["id"],
                "baseline_run": base_id,
                "candidate_run": cand_id,
                "override_reason": "too short",
            },
        ).status_code
        == 422
    )
    client.put(f"/api/eval/gate/baselines/{policy['id']}", json={"run_id": base_id})
    again = client.post(
        "/api/eval/gate/evaluate", json={"policy_id": policy["id"], "candidate_run": base_id}
    ).json()
    # The promoted baseline is used when none is given. (The fake grader lets the dog
    # question through, so the unanswerable ceiling still blocks — correctly.)
    assert again["baseline"]["id"] == base_id
    assert any("unanswerable-ceiling" in r for r in again["reasons"])
    monkeypatch.setattr(routes_gate, "RELEASE_DIR", tmp_path / "release")
    snap = client.post(
        "/api/eval/gate/export",
        json={"policy_id": policy["id"], "baseline_run": base_id, "candidate_run": cand_id},
    ).json()
    assert snap["result"]["exit_code"] == 1 and (tmp_path / "release" / "candidate.json").exists()
