"""Crew topology verification — runs the supervisor graph with fake ports.
Proves the happy path, the no-text SKIPPED short-circuit, and the adaptive
'skip classifier when already typed' route.

Run: PYTHONPATH=src python tests/test_ingestion_crew.py
"""

import asyncio

from agent_hub.adapters.ports.knowhub import Payload
from agent_hub.application.ingestion.agents.chunker import ChunkerAgent
from agent_hub.application.ingestion.agents.classifier import ClassifierAgent
from agent_hub.application.ingestion.agents.embedder import EmbedderAgent
from agent_hub.application.ingestion.agents.extractor import ExtractorAgent
from agent_hub.application.ingestion.graph import build_ingestion_graph
from agent_hub.domain.models import DocClass

CONTRACT = "Agreement.\n1. Term. This lasts one year.\n2. Payment. Net 30 days.\n3. Termination. 30 days notice."


class FakeGateway:
    def __init__(self, text=CONTRACT, found=True):
        self._text, self._found = text, found
        self.persisted = None
        self.metadata = None

    async def get_payload(self, document_id, actor_user_id):
        return Payload(raw_text=self._text, binary=b"", mime_type="text/plain", found=self._found)

    async def persist_chunks(self, document_id, chunks, strategy, model):
        self.persisted = (len(chunks), strategy)
        return len(chunks)

    async def update_metadata(self, document_id, doc_type="", topic="", chunk_strategy="", status=""):
        self.metadata = {"doc_type": doc_type, "status": status, "strategy": chunk_strategy}
        return True


class FakeClassifier:
    def __init__(self):
        self.called = False

    async def classify(self, sample_text):
        self.called = True
        return DocClass(doc_type="contract", topic="services", suggested_strategy="clause")


class FakeEmbedder:
    async def embed(self, texts):
        return [[0.0] * 768 for _ in texts]


def _crew(gateway, classifier, embedder):
    return build_ingestion_graph(
        extractor=ExtractorAgent(gateway),
        classifier=ClassifierAgent(classifier),
        chunker=ChunkerAgent(),
        embedder=EmbedderAgent(embedder, gateway),
    )


def test_happy_path():
    gw, clf, emb = FakeGateway(), FakeClassifier(), FakeEmbedder()
    app = _crew(gw, clf, emb)
    out = asyncio.run(app.ainvoke({"document_id": "d1", "actor_user_id": "u1"}))
    assert out["status"] == "INDEXED"
    assert out["doc_type"] == "contract"
    assert out["strategy"] == "clause"
    assert out["indexed"] > 0
    assert clf.called is True
    assert gw.persisted[1] == "clause"          # persisted with the chosen strategy
    assert gw.metadata["status"] == "INDEXED"
    # full crew path visited, in order
    nodes = [h.split(":")[0] for h in out["history"]]
    assert nodes == ["extractor", "classifier", "chunker", "embedder"]


def test_no_text_skips():
    gw = FakeGateway(text="", found=True)
    clf, emb = FakeClassifier(), FakeEmbedder()
    out = asyncio.run(_crew(gw, clf, emb).ainvoke({"document_id": "img1", "actor_user_id": "u1"}))
    assert out["status"] == "SKIPPED"
    assert clf.called is False                   # never classified an image
    assert gw.persisted is None                  # nothing persisted


def test_already_typed_skips_classifier():
    gw, clf, emb = FakeGateway(), FakeClassifier(), FakeEmbedder()
    # Pre-seed doc_type → supervisor's adaptive skip avoids the LLM classifier.
    out = asyncio.run(
        _crew(gw, clf, emb).ainvoke(
            {"document_id": "d2", "actor_user_id": "u1", "doc_type": "article", "strategy": "fixed"}
        )
    )
    assert out["status"] == "INDEXED"
    assert clf.called is False                   # classification skipped
    assert "classifier" not in [h.split(":")[0] for h in out["history"]]


if __name__ == "__main__":
    test_happy_path();                  print("happy_path                  OK")
    test_no_text_skips();               print("no_text_skips               OK")
    test_already_typed_skips_classifier(); print("already_typed_skips_classifier OK")
    print("\nALL INGESTION CREW TESTS PASSED")
