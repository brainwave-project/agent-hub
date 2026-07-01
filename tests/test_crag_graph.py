"""Graph-logic verification for the CRAG agent — runs WITHOUT Ollama or know-hub
by injecting fake ports. Proves topology, the corrective cycle, and both
guardrails (MAX_LOOPS abstain, bounded grounding retry).

Run: PYTHONPATH=src python -m pytest tests/test_crag_graph.py -q
 or: PYTHONPATH=src python tests/test_crag_graph.py   (self-running)
"""

import asyncio

from agent_hub.application.crag.graph import build_crag_graph
from agent_hub.application.crag.nodes import CragNodes
from agent_hub.application.hello_graph import hello_app
from agent_hub.domain.models import Grade, Groundedness, RetrievedChunk

# --- fakes ----------------------------------------------------------------


class FakeRetriever:
    def __init__(self):
        self.calls = []

    async def search(self, query, top_k, actor_user_id, is_admin, min_score=0.0):
        self.calls.append(query)
        # score encodes "weak vs strong" so grade can be scripted off retrieval.
        score = 0.9 if "good" in query else 0.3
        return [RetrievedChunk(chunk_id="c1", text="ctx", score=score, document_id="doc-1", title="T")]


class ScriptedReasoner:
    """grade() returns relevant per a queue; rewrite makes the query 'good';
    grounded() returns per a queue."""

    def __init__(self, grade_seq, grounded_seq):
        self._grade = list(grade_seq)
        self._grounded = list(grounded_seq)
        self.gen_calls = 0

    async def grade(self, question, chunks):
        v = self._grade.pop(0) if self._grade else True
        return Grade(relevant=v, reason="scripted")

    async def rewrite_query(self, question, current_query, chunks):
        return "good query"

    async def generate(self, question, chunks):
        self.gen_calls += 1
        return f"answer #{self.gen_calls} [doc:{chunks[0].document_id}]" if chunks else "answer"

    async def stream_generate(self, question, chunks):
        yield await self.generate(question, chunks)

    async def check_grounded(self, answer, chunks):
        v = self._grounded.pop(0) if self._grounded else True
        return Groundedness(grounded=v, reason="scripted")


class NullRecorder:
    async def record(self, *a, **k):
        return None


class FakeReranker:
    """Reverses the candidate order and truncates — enough to prove wiring."""

    def __init__(self):
        self.calls = []

    async def rerank(self, query, chunks, top_n):
        self.calls.append((query, len(chunks), top_n))
        return list(reversed(chunks))[:top_n]


def _run(
    grade_seq,
    grounded_seq,
    max_loops=2,
    max_regen=1,
    query_rewrite_enabled=True,
    grounding_enabled=True,
):
    r = FakeRetriever()
    reasoner = ScriptedReasoner(grade_seq, grounded_seq)
    nodes = CragNodes(r, reasoner, NullRecorder(), top_k=3)
    app = build_crag_graph(
        nodes,
        max_loops=max_loops,
        max_regen=max_regen,
        query_rewrite_enabled=query_rewrite_enabled,
        grounding_enabled=grounding_enabled,
    )
    state = asyncio.run(
        app.ainvoke({"question": "what is x?", "actor_user_id": "u1", "is_admin": False})
    )
    return state, r, reasoner


def test_hello_graph():
    out = asyncio.run(hello_app.ainvoke({"message": "world", "steps": []}))
    assert out["message"] == "HELLO, WORLD"
    assert out["steps"] == ["greet", "shout"]


def test_happy_path():
    # relevant immediately, grounded immediately
    state, r, reasoner = _run(grade_seq=[True], grounded_seq=[True])
    assert state["answer"].startswith("answer #1")
    assert state.get("abstained") is not True
    assert state["citations"][0].document_id == "doc-1"
    assert len(r.calls) == 1            # retrieved once
    assert state.get("loops", 0) == 0


def test_corrective_loop():
    # weak once → rewrite → strong → generate → grounded
    state, r, reasoner = _run(grade_seq=[False, True], grounded_seq=[True])
    assert state["loops"] == 1          # exactly one corrective rewrite
    assert r.calls == ["what is x?", "good query"]
    assert state.get("abstained") is not True


def test_abstain_after_max_loops():
    # always weak → exhausts MAX_LOOPS → abstain
    state, r, reasoner = _run(grade_seq=[False, False, False, False, False], grounded_seq=[])
    assert state["abstained"] is True
    assert "don't have enough information" in state["answer"]
    assert state["loops"] == 2          # MAX_LOOPS rewrites attempted
    assert reasoner.gen_calls == 0      # never generated a hallucinated answer


def test_grounding_retry_is_bounded():
    # relevant, but never grounded → must regenerate ONCE then stop (Issue 4)
    state, r, reasoner = _run(grade_seq=[True], grounded_seq=[False, False, False])
    assert reasoner.gen_calls == 2      # initial + exactly one regeneration
    assert state["grounded"] is False   # gave up honestly, did not loop forever


# --- governance toggles (§13.5) -------------------------------------------


def test_query_rewrite_disabled_abstains_immediately():
    # weak grade, but the corrective loop is governed off → abstain at once.
    state, r, reasoner = _run(
        grade_seq=[False], grounded_seq=[], query_rewrite_enabled=False
    )
    assert state["abstained"] is True
    assert state.get("loops", 0) == 0   # no rewrite attempted
    assert r.calls == ["what is x?"]    # retrieved exactly once
    assert reasoner.gen_calls == 0


def test_grounding_disabled_skips_check():
    # grounding governed off → generate goes straight to END, no regeneration
    # even though the grounded queue would have said "not grounded".
    state, r, reasoner = _run(
        grade_seq=[True], grounded_seq=[False, False], grounding_enabled=False
    )
    assert reasoner.gen_calls == 1
    assert "grounded" not in state      # check_grounded node never ran
    assert state["answer"].startswith("answer #1")


# --- cross-encoder rerank in the retrieve node ----------------------------


def test_rerank_reorders_and_truncates():
    r = FakeRetriever()
    reasoner = ScriptedReasoner([True], [True])
    reranker = FakeReranker()
    # 3 chunks in; wide candidate pool; keep top 2 after rerank.
    r_multi = FakeRetriever()

    async def multi_search(query, top_k, actor_user_id, is_admin, min_score=0.0):
        return [
            RetrievedChunk(chunk_id=f"c{i}", text=f"t{i}", score=0.5 + i / 10,
                           document_id=f"d{i}", title="T")
            for i in range(3)
        ]

    r_multi.search = multi_search
    nodes = CragNodes(
        r_multi, reasoner, NullRecorder(),
        top_k=3, reranker=reranker, cross_encoder_enabled=True,
        rerank_candidate_k=10, rerank_top_n=2,
    )
    out = asyncio.run(nodes.retrieve({"question": "q", "actor_user_id": "u", "is_admin": False}))
    assert reranker.calls == [("q", 3, 2)]     # got the candidate pool, asked for 2
    assert len(out["chunks"]) == 2             # truncated to rerank_top_n
    assert out["chunks"][0].chunk_id == "c2"   # FakeReranker reversed order


def test_rerank_off_by_default():
    r = FakeRetriever()
    reasoner = ScriptedReasoner([True], [True])
    reranker = FakeReranker()
    nodes = CragNodes(r, reasoner, NullRecorder(), top_k=3, reranker=reranker)
    asyncio.run(nodes.retrieve({"question": "q", "actor_user_id": "u", "is_admin": False}))
    assert reranker.calls == []                # never invoked when toggle is off


if __name__ == "__main__":
    test_hello_graph();            print("hello_graph            OK")
    test_happy_path();             print("happy_path             OK")
    test_corrective_loop();        print("corrective_loop        OK")
    test_abstain_after_max_loops();print("abstain_after_max_loops OK")
    test_grounding_retry_is_bounded(); print("grounding_retry_bounded OK")
    test_query_rewrite_disabled_abstains_immediately(); print("rewrite_disabled       OK")
    test_grounding_disabled_skips_check(); print("grounding_disabled     OK")
    test_rerank_reorders_and_truncates(); print("rerank_reorder         OK")
    test_rerank_off_by_default();  print("rerank_off_default     OK")
    print("\nALL CRAG GRAPH TESTS PASSED")
