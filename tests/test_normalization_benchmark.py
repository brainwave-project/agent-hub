"""Controlled retrieval invariants. No stack, network, GPU, or real model required."""

import json
import math
import sys
import types

import pytest

from agent_hub.eval.normalization import BenchmarkStore, RunConfig
from agent_hub.eval.normalization.benchmark import METRICS, paired_comparison
from agent_hub.eval.normalization.prepare import (
    CORPUS_REVISION,
    TOPICS_REVISION,
    prepare_miracl,
    select_queries,
)
from agent_hub.eval.normalization.retrievers import BM25, build_bm25
from agent_hub.eval.normalization.text import normalize, tokenize


@pytest.fixture
def store(tmp_path):
    return BenchmarkStore(tmp_path / "normalization")


def import_pair(store):
    return store.import_records(
        "Arabic pair",
        [
            {"passage_id": "wiki#0", "title": "", "text": "أحمد"},
            {"passage_id": "wiki#1", "title": "", "text": "خالد"},
        ],
        [
            {"qid": "query#0", "query": "احمد", "tags": ["unchanged_query"]},
            {"qid": "query#1", "query": "خالد", "tags": ["control"]},
        ],
        [
            {"qid": "query#0", "passage_id": "wiki#0", "relevance": 1},
            {"qid": "query#0", "passage_id": "wiki#1", "relevance": 0},
            {"qid": "query#1", "passage_id": "wiki#1", "relevance": 1},
        ],
    )


def test_raw_tokenizer_preserves_marks_and_every_transform_is_explicit():
    sample = "أَحـمد، إِبراهيم آمال هدى مدرسة ٢٠٢٦ A_B"
    assert normalize(sample, "raw") == sample
    assert tokenize("عِلْم علـم علم") == ["عِلْم", "علـم", "علم"]
    assert tokenize("أحمد،123،A_B!") == ["أحمد", "123", "A_B"]
    assert normalize("أَحـمد", "alef") == "اَحـمد"
    assert normalize("أَحـمد", "diacritics") == "أحـمد"
    assert normalize("أَحـمد", "tatweel") == "أَحمد"
    assert normalize("أَحـمد", "conservative") == "احمد"
    assert normalize("أَحـمد هدى مدرسة", "aggressive") == "احمد هدي مدرسه"
    # Hamza above U+0654 is meaningful, outside our explicitly limited vowel-mark rule.
    assert normalize("ا\u0654 ماء ٢٠٢٦", "conservative") == "ا\u0654 ماء ٢٠٢٦"


def test_bm25_matches_hand_calculation_and_never_pads_unmatched_results(store):
    manifest = store.import_records(
        "Tiny BM25",
        [
            {"passage_id": "p0", "text": "أحمد أحمد"},
            {"passage_id": "p1", "text": "أحمد خالد"},
            {"passage_id": "p2", "text": "خالد"},
        ],
        [{"qid": "q0", "query": "أحمد"}],
        [{"qid": "q0", "passage_id": "p0", "relevance": 1}],
    )
    corpus = store.root / "datasets" / manifest["dataset_id"] / "corpus.sqlite"
    path = store.root / "cache" / "test.sqlite"
    build_bm25(corpus, path, "raw")
    retriever = BM25(path)
    try:
        rows = retriever.search("أحمد أحمد", 100)
        assert [pid for pid, score in rows] == ["p0", "p1"]
        idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
        expected = idf * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 2 / (5 / 3)))
        assert rows[0][1] == pytest.approx(expected)
        assert retriever.search("غيرموجود", 100) == []
        assert retriever.search("", 100) == []
    finally:
        retriever.close()


def test_corpus_normalization_can_help_an_unchanged_query_and_exports_are_real(store):
    manifest = import_pair(store)
    config = RunConfig(presets=["raw", "alef"], bootstrap_samples=100)
    report = store.run(manifest["dataset_id"], config)
    raw, alef = report["conditions"]
    assert raw["metrics"]["ndcg@10"] == 0.5
    assert alef["metrics"]["ndcg@10"] == 1
    assert alef["changed_queries"] == 0
    assert alef["changed_documents"] == 1
    assert alef["slices"]["query_unchanged"]["metrics"]["ndcg@10"] == 1
    assert alef["comparison_to_raw"]["ndcg@10"]["delta"] == 0.5
    assert alef["comparison_to_raw"]["ndcg@10"]["wins"] == 1
    assert raw["per_query"][0]["ranking"] == []
    assert alef["per_query"][0]["ranking"][0]["passage_id"] == "wiki#0"
    assert set(raw["metrics"]) == set(METRICS)
    second = store.run(manifest["dataset_id"], config)
    assert second["conditions"] == report["conditions"]
    assert store.get_run(report["run_id"]) == report
    assert "per_query" not in store.get_summary(report["run_id"])["conditions"][0]
    assert len(store.list_runs()) == 2
    trec = store.root / "runs" / f"{report['run_id']}-trec" / "bm25-alef.trec"
    assert "query#0 Q0 wiki#0 1 " in trec.read_text()


def test_query_only_keeps_raw_documents_and_complete_relevance_denominator(store):
    manifest = store.import_records(
        "Two relevant passages",
        [
            {"passage_id": "p0", "text": "أحمد"},
            {"passage_id": "p1", "text": "خالد"},
        ],
        [{"qid": "q0", "query": "أحمد"}],
        [
            {"qid": "q0", "passage_id": "p0", "relevance": 1},
            {"qid": "q0", "passage_id": "p1", "relevance": 1},
        ],
    )
    report = store.run(
        manifest["dataset_id"],
        RunConfig(presets=["raw", "alef"], normalization_side="query", bootstrap_samples=100),
    )
    raw, alef = report["conditions"]
    assert raw["metrics"]["recall@100"] == 0.5
    assert alef["metrics"]["recall@100"] == 0
    assert alef["changed_documents"] == 0
    assert alef["changed_queries"] == 1


@pytest.mark.parametrize("problem", ["missing", "duplicate", "no_positive"])
def test_invalid_qrels_leave_no_dataset(store, problem):
    judgments = [{"qid": "q0", "passage_id": "p0", "relevance": 1}]
    if problem == "missing":
        judgments[0]["passage_id"] = "missing#99"
    elif problem == "duplicate":
        judgments.append(dict(judgments[0]))
    else:
        judgments[0]["relevance"] = 0
    with pytest.raises(ValueError):
        store.import_records(
            "Bad", [{"passage_id": "p0", "text": "نص"}], [{"qid": "q0", "query": "نص"}], judgments
        )
    assert store.list_datasets() == []


@pytest.mark.parametrize(
    "filename", ["corpus.jsonl", "queries.jsonl", "qrels.tsv", "corpus.sqlite"]
)
def test_dataset_tampering_is_rejected_before_index_cache_reuse(store, filename):
    manifest = import_pair(store)
    config = RunConfig(presets=["raw"], bootstrap_samples=100)
    store.run(manifest["dataset_id"], config)
    path = store.root / "datasets" / manifest["dataset_id"] / filename
    with path.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="Immutable dataset content changed"):
        store.run(manifest["dataset_id"], config)
    assert len(store.list_runs()) == 1
    assert not (store.root / "run.lock").exists()


def test_failure_does_not_record_success_and_releases_lock(store, monkeypatch):
    manifest = import_pair(store)

    def fail(*_args):
        raise RuntimeError("injected retrieval failure")

    monkeypatch.setattr(BM25, "search", fail)
    with pytest.raises(RuntimeError, match="injected"):
        store.run(manifest["dataset_id"], RunConfig(presets=["raw"], bootstrap_samples=100))
    assert not store.list_runs()
    assert not list((store.root / "runs").glob("*.json"))
    assert not (store.root / "run.lock").exists()


def test_config_requires_raw_and_pinned_model_and_ids_cannot_escape_store(store):
    for config in [
        RunConfig(presets=["alef"]),
        RunConfig(retrievers=["hybrid"]),
        RunConfig(retrievers=["dense"], dense_model="x", dense_revision="main"),
        RunConfig(bootstrap_samples=0),
        RunConfig(top_k=10),
        RunConfig(retrievers=[{}]),
        RunConfig(presets=["raw", {}]),
    ]:
        with pytest.raises(ValueError):
            config.validate()
    for method in (store.get_dataset, store.get_run, store.get_summary):
        with pytest.raises(ValueError):
            method("../../secrets")


def test_paired_bootstrap_aligns_queries_and_is_reproducible():
    def row(qid, score):
        return {"qid": qid, "metrics": dict.fromkeys(METRICS, score)}

    before = [row("a", 0), row("b", 1), row("c", 0)]
    after = [row("a", 1), row("b", 0), row("c", 1)]
    comparison = paired_comparison(after, before, 100, 42)
    assert comparison == paired_comparison(after, before, 100, 42)
    assert comparison["ndcg@10"]["delta"] == pytest.approx(1 / 3)
    assert comparison["ndcg@10"]["wins"] == 2
    assert comparison["ndcg@10"]["losses"] == 1
    assert comparison["ndcg@10"] == comparison["mrr@10"]
    with pytest.raises(ValueError, match="identical ordered"):
        paired_comparison(after[::-1], before, 100, 42)


def test_dense_uses_one_fixed_encoder_and_reuses_document_cache(store, monkeypatch):
    np = pytest.importorskip("numpy")
    from agent_hub.eval.normalization import retrievers

    constructions, encodings = [], []

    class FakeEncoder:
        device = "cpu"

        def __init__(self, model, revision, trust_remote_code):
            constructions.append((model, revision, trust_remote_code))

        def __str__(self):
            return "FakeEncoder(mean pooling)"

        def encode(self, texts, **kwargs):
            assert kwargs["normalize_embeddings"] is True
            encodings.append(list(texts))
            result = np.asarray(
                [[text.count("أ"), text.count("ا"), 1] for text in texts], dtype="float32"
            )
            return result / np.linalg.norm(result, axis=1, keepdims=True)

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeEncoder)
    )
    monkeypatch.setattr(retrievers.importlib.metadata, "version", lambda _: "fake-version")
    manifest = import_pair(store)
    config = RunConfig(
        retrievers=["dense"],
        presets=["raw", "alef"],
        bootstrap_samples=100,
        dense_model="fixture/model",
        dense_revision="a" * 40,
        dense_batch_size=2,
    )
    report = store.run(manifest["dataset_id"], config)
    assert constructions == [("fixture/model", "a" * 40, False)]
    assert len(encodings) == 6  # Two corpus batches and two queries per condition.
    assert len(list((store.root / "cache").rglob("*.npy"))) == 2
    encodings.clear()
    repeated = store.run(manifest["dataset_id"], config)
    assert len(encodings) == 4  # Queries only; both corpus memmaps are reused.
    assert repeated["conditions"] == report["conditions"]
    assert report["provenance"]["dense"]["architecture"] == "FakeEncoder(mean pooling)"
    config.document_prefix = "passage: "
    encodings.clear()
    store.run(manifest["dataset_id"], config)
    assert len(encodings) == 6
    assert any(batch[0].startswith("passage: ") for batch in encodings)
    assert len(list((store.root / "cache").rglob("*.npy"))) == 4


@pytest.mark.parametrize("mode", ["pilot", "full"])
@pytest.mark.parametrize("split", ["train", "dev"])
def test_official_miracl_preparation_with_mocked_streams(tmp_path, monkeypatch, mode, split):
    from agent_hub.eval.normalization import prepare

    calls = []
    passages = [{"docid": f"wiki#{i}", "title": "", "text": f"نص {i}"} for i in range(8)]

    def lines(url, compressed=False):
        calls.append(url)
        if "/topics/" in url:
            yield "q#0\tأحمد\n"
            yield "q#1\tخالد\n"
        elif "/qrels/" in url:
            yield "q#0\t0\twiki#0\t1\n"
            yield "q#0\t0\twiki#1\t0\n"
            yield "q#1\t0\twiki#2\t1\n"
            yield "q#1\t0\twiki#3\t0\n"
        elif "docs-0.jsonl.gz" in url:
            assert compressed
            yield from (json.dumps(row, ensure_ascii=False) + "\n" for row in passages)

    monkeypatch.setattr(prepare, "_lines", lines)
    output = tmp_path / f"prepared-{mode}"
    provenance = prepare_miracl(
        output, mode=mode, query_limit=1 if mode == "pilot" else None, distractors=2, split=split
    )
    assert all(TOPICS_REVISION in url for url in calls[:2])
    assert all(f"-{split}.tsv" in url for url in calls[:2])
    assert all(CORPUS_REVISION in url for url in calls[2:])
    assert len(calls) == 7
    assert not (output / "INCOMPLETE").exists()
    imported = BenchmarkStore(tmp_path / "store").import_files(
        mode, output / "corpus.jsonl", output / "queries.jsonl", output / "qrels.tsv", provenance
    )
    assert imported["document_count"] == (4 if mode == "pilot" else 8)
    assert imported["query_count"] == (1 if mode == "pilot" else 2)
    assert imported["qrel_count"] == (2 if mode == "pilot" else 4)
    assert provenance["scope"] == (
        "miracl-pilot" if mode == "pilot" else f"full-corpus-all-{split}-queries"
    )
    if mode == "pilot":
        assert provenance["corpus_selection"]["label_informed"] is True
        assert imported["warnings"]
    with pytest.raises(ValueError, match="already exists"):
        prepare_miracl(output)


def test_query_subset_selection_uses_ids_and_seed_not_input_order_or_results():
    queries = [{"qid": f"q#{i}", "query": "نص"} for i in range(20)]
    assert select_queries(queries, 5, 42) == select_queries(queries[::-1], 5, 42)
    assert select_queries(queries, 5, 42) != select_queries(queries, 5, 43)
    with pytest.raises(ValueError):
        select_queries(queries, 0, 42)


def test_trec_export_preserves_evaluated_tie_order(store):
    report = {
        "conditions": [
            {
                "retriever": "bm25",
                "preset": "raw",
                "per_query": [
                    {
                        "qid": "q1",
                        "ranking": [
                            {"passage_id": "a", "score": 0.5},
                            {"passage_id": "z", "score": 0.5},
                            {"passage_id": "m", "score": 0.49999999999999},
                        ],
                    }
                ],
            }
        ]
    }
    directory = store.export_trec("a" * 32, report=report)
    rows = [line.split() for line in (directory / "bm25-raw.trec").read_text().splitlines()]
    assert [row[2] for row in sorted(rows, key=lambda row: -float(row[4]))] == ["a", "z", "m"]
    assert len({row[4] for row in rows}) == 3
