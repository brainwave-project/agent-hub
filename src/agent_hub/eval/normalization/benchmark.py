"""Immutable local datasets and reproducible, paired normalization runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import sqlite3
import statistics
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from agent_hub.eval.retrieval_metrics import graded_ndcg_at_k, mrr, recall_at_k

from .retrievers import BM25, Dense, build_bm25
from .text import PRESETS, TOKENIZER, normalize

METRICS = ["ndcg@10", "recall@5", "recall@10", "recall@100", "mrr@10"]
Progress = Callable[[dict], None]


@dataclass
class RunConfig:
    retrievers: list[str] = field(default_factory=lambda: ["bm25"])
    presets: list[str] = field(default_factory=lambda: [p["id"] for p in PRESETS])
    normalization_side: str = "both"
    bootstrap_samples: int = 1000
    seed: int = 42
    dense_model: str | None = None
    dense_revision: str | None = None
    dense_batch_size: int = 32
    dense_max_seq_length: int = 256
    query_prefix: str = ""
    document_prefix: str = ""
    top_k: int = 100

    def validate(self):
        if (
            not isinstance(self.retrievers, list)
            or not self.retrievers
            or any(not isinstance(value, str) for value in self.retrievers)
            or len(set(self.retrievers)) != len(self.retrievers)
            or set(self.retrievers) - {"bm25", "dense"}
        ):
            raise ValueError("retrievers must be a unique nonempty list of bm25 and/or dense")
        if (
            not isinstance(self.presets, list)
            or "raw" not in self.presets
            or any(not isinstance(value, str) for value in self.presets)
            or len(set(self.presets)) != len(self.presets)
            or set(self.presets) - {p["id"] for p in PRESETS}
        ):
            raise ValueError(
                "presets must be unique, valid, and include raw for paired comparisons"
            )
        if self.normalization_side not in ("both", "query"):
            raise ValueError("normalization_side must be both or query")
        for key, low, high in (
            ("bootstrap_samples", 100, 20000),
            ("top_k", 100, 1000),
            ("dense_batch_size", 1, 256),
            ("dense_max_seq_length", 8, 8192),
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{key} must be an integer between {low} and {high}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(self.query_prefix, str) or not isinstance(self.document_prefix, str):
            raise ValueError("query_prefix and document_prefix must be strings")
        if "dense" in self.retrievers:
            if not isinstance(self.dense_model, str) or not self.dense_model.strip():
                raise ValueError("dense_model is required for dense runs")
            if (
                not re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", self.dense_model
                )
                or Path(self.dense_model).exists()
            ):
                raise ValueError(
                    "dense_model must be a Hugging Face owner/model ID, not a local path"
                )
            if not isinstance(self.dense_revision, str) or not re.fullmatch(
                r"[0-9a-fA-F]{40}", self.dense_revision
            ):
                raise ValueError("dense_revision must be a pinned 40-character model commit SHA")


def options() -> dict:
    return {
        "presets": PRESETS,
        "metrics": METRICS,
        "defaults": asdict(RunConfig()),
        "tokenizer": TOKENIZER,
    }


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value):
    temp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _json_line(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("expected a JSON object")
                    yield value
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path.name} line {line_number}: {exc}") from exc


def read_qrels(path: Path):
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"qrels line {line_number}: expected qid 0 passage_id relevance")
            try:
                relevance = int(parts[3])
            except ValueError as exc:
                raise ValueError(f"qrels line {line_number}: relevance must be integer") from exc
            yield {"qid": parts[0], "passage_id": parts[2], "relevance": relevance}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _external_id(value, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise ValueError(f"{field_name} must be a nonempty string without whitespace")
    return value


def _mean(rows: list[dict]) -> dict:
    return {
        metric: statistics.fmean(row["metrics"][metric] for row in rows) if rows else 0.0
        for metric in METRICS
    }


def _percentile(values: list[float], p: float) -> float:
    position = (len(values) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def paired_comparison(rows: list[dict], baseline: list[dict], samples: int, seed: int) -> dict:
    """Percentile CI of mean paired query differences, sharing resamples across metrics.

    A fresh RNG with the same seed per metric/condition creates identical index
    draws over the same ordered qids. Resample queries, never individual ranks.
    No multiple-comparison correction is implied by these exploratory intervals.
    """
    if [r["qid"] for r in rows] != [r["qid"] for r in baseline]:
        raise ValueError("paired comparison requires identical ordered query IDs")
    result = {}
    for metric in METRICS:
        differences = [
            row["metrics"][metric] - base["metrics"][metric]
            for row, base in zip(rows, baseline, strict=True)
        ]
        n = len(differences)
        if not n:
            raise ValueError("paired comparison requires at least one query")
        if min(differences) == max(differences):
            # Identical deltas give a point-mass bootstrap distribution, especially raw/raw.
            means = [differences[0]]
        else:
            rng = random.Random(seed)
            means = sorted(
                statistics.fmean(differences[rng.randrange(n)] for _ in range(n))
                for _ in range(samples)
            )
        result[metric] = {
            "delta": statistics.fmean(differences),
            "ci_low": _percentile(means, 0.025),
            "ci_high": _percentile(means, 0.975),
            "wins": sum(d > 1e-12 for d in differences),
            "losses": sum(d < -1e-12 for d in differences),
            "ties": sum(abs(d) <= 1e-12 for d in differences),
        }
    return result


class BenchmarkStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        for directory in ("datasets", "runs", "cache"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)

    def _id(self, value: str) -> str:
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value):
            raise ValueError("dataset/run ID must be a generated 32-character hexadecimal ID")
        return value

    def get_dataset(self, dataset_id: str) -> dict:
        return _json(self.root / "datasets" / self._id(dataset_id) / "manifest.json")

    def list_datasets(self) -> list[dict]:
        return sorted(
            (_json(p) for p in (self.root / "datasets").glob("*/manifest.json")),
            key=lambda row: row["created_at"],
            reverse=True,
        )

    def import_files(
        self,
        name: str,
        corpus_path: Path,
        queries_path: Path,
        qrels_path: Path,
        provenance: dict | None = None,
    ) -> dict:
        if (Path(corpus_path).parent / "INCOMPLETE").exists():
            raise ValueError("MIRACL preparation is incomplete; do not import partial data")
        return self.import_records(
            name,
            read_jsonl(Path(corpus_path)),
            read_jsonl(Path(queries_path)),
            read_qrels(Path(qrels_path)),
            provenance,
        )

    def import_records(
        self,
        name: str,
        corpus: Iterable[dict],
        queries: Iterable[dict],
        qrels: Iterable[dict],
        provenance: dict | None = None,
    ) -> dict:
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            raise ValueError("dataset name must have 1–200 characters")
        if provenance is not None and not isinstance(provenance, dict):
            raise ValueError("provenance must be a JSON object")
        dataset_id = uuid.uuid4().hex
        directory = self.root / "datasets" / dataset_id
        directory.mkdir()
        conn = sqlite3.connect(directory / "corpus.sqlite")
        try:
            conn.execute(
                "CREATE TABLE corpus (ordinal INTEGER PRIMARY KEY, "
                "passage_id TEXT UNIQUE, title TEXT, text TEXT)"
            )
            document_count = 0
            with (directory / "corpus.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
                for ordinal, row in enumerate(corpus):
                    pid = _external_id(row.get("passage_id"), "passage_id")
                    title, text = row.get("title", ""), row.get("text")
                    if not isinstance(title, str) or not isinstance(text, str) or not text.strip():
                        raise ValueError(f"passage {pid}: title and nonempty text must be strings")
                    try:
                        conn.execute(
                            "INSERT INTO corpus VALUES (?, ?, ?, ?)", (ordinal, pid, title, text)
                        )
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(f"duplicate passage_id: {pid}") from exc
                    stream.write(_json_line({"passage_id": pid, "title": title, "text": text}))
                    document_count += 1
            if not document_count:
                raise ValueError("corpus is empty")
            query_rows = {}
            for row in queries:
                qid = _external_id(row.get("qid"), "qid")
                if qid in query_rows:
                    raise ValueError(f"duplicate qid: {qid}")
                if not isinstance(row.get("query"), str) or not row["query"].strip():
                    raise ValueError(f"query {qid}: query must be a nonempty string")
                tags = row.get("tags", [])
                if not isinstance(tags, list) or any(not isinstance(t, str) for t in tags):
                    raise ValueError(f"query {qid}: tags must be a list of strings")
                query_rows[qid] = {"qid": qid, "query": row["query"], "tags": sorted(set(tags))}
            if not query_rows:
                raise ValueError("queries are empty")
            judgments, positive_qids = {}, set()
            for row in qrels:
                qid = _external_id(row.get("qid"), "qrel qid")
                pid = _external_id(row.get("passage_id"), "qrel passage_id")
                grade = row.get("relevance")
                if isinstance(grade, bool) or not isinstance(grade, int) or not 0 <= grade <= 31:
                    raise ValueError("qrel relevance must be an integer in [0, 31]")
                if qid not in query_rows:
                    raise ValueError(f"qrel references missing query: {qid}")
                if not conn.execute("SELECT 1 FROM corpus WHERE passage_id=?", (pid,)).fetchone():
                    raise ValueError(f"qrel references missing passage: {pid}")
                if (qid, pid) in judgments:
                    raise ValueError(f"duplicate qrel: {qid} {pid}")
                judgments[qid, pid] = grade
                if grade > 0:
                    positive_qids.add(qid)
            missing_positive = set(query_rows) - positive_qids
            if missing_positive:
                raise ValueError(
                    f"queries need at least one positive qrel: {min(missing_positive)}"
                )
            with (directory / "queries.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
                for qid in sorted(query_rows):
                    stream.write(_json_line(query_rows[qid]))
            with (directory / "qrels.tsv").open("w", encoding="utf-8", newline="\n") as stream:
                for (qid, pid), grade in sorted(judgments.items()):
                    stream.write(f"{qid}\t0\t{pid}\t{grade}\n")
            conn.commit()
            conn.close()
            manifest = {
                "dataset_id": dataset_id,
                "name": name.strip(),
                "created_at": _now(),
                "document_count": document_count,
                "query_count": len(query_rows),
                "qrel_count": len(judgments),
                "hashes": {
                    key: sha_file(directory / filename)
                    for key, filename in (
                        ("corpus", "corpus.jsonl"),
                        ("queries", "queries.jsonl"),
                        ("qrels", "qrels.tsv"),
                        ("corpus_db", "corpus.sqlite"),
                    )
                },
                "provenance": provenance or {"source": "local", "scope": "custom"},
                "warnings": [],
            }
            scope = manifest["provenance"].get("scope", "custom")
            if scope != "full-corpus-all-dev-queries":
                manifest["warnings"].append(
                    "Custom, pilot, or query-subset data: not comparable to full MIRACL dev scores."
                )
            _write_json(directory / "manifest.json", manifest)
            return manifest
        except BaseException:
            conn.close()
            # This directory is an uncommitted import with a freshly generated UUID.
            shutil.rmtree(directory)
            raise

    def get_run(self, run_id: str) -> dict:
        return _json(self.root / "runs" / f"{self._id(run_id)}.json")

    def get_summary(self, run_id: str) -> dict:
        return _json(self.root / "runs" / f"{self._id(run_id)}.summary.json")

    def list_runs(self) -> list[dict]:
        return sorted(
            (_json(p) for p in (self.root / "runs").glob("*.summary.json")),
            key=lambda row: row["created_at"],
            reverse=True,
        )

    def run(
        self,
        dataset_id: str,
        config: RunConfig | dict | None = None,
        progress: Progress | None = None,
    ) -> dict:
        if config is None:
            config = RunConfig()
        elif isinstance(config, dict):
            try:
                config = RunConfig(**config)
            except TypeError as exc:
                raise ValueError(f"invalid run configuration: {exc}") from exc
        config.validate()
        manifest = self.get_dataset(dataset_id)
        lock = self.root / "run.lock"
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise ValueError(
                "A benchmark run owns run.lock. Wait for it; after a crashed process, "
                "verify its PID is gone before removing that lock."
            ) from exc
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump({"pid": os.getpid(), "created_at": _now()}, stream)
            return self._run(manifest, config, progress)
        finally:
            lock.unlink(missing_ok=True)

    def _run(self, manifest: dict, config: RunConfig, progress: Progress | None) -> dict:
        directory = self.root / "datasets" / manifest["dataset_id"]
        for key, filename in (
            ("corpus", "corpus.jsonl"),
            ("queries", "queries.jsonl"),
            ("qrels", "qrels.tsv"),
            ("corpus_db", "corpus.sqlite"),
        ):
            if sha_file(directory / filename) != manifest["hashes"][key]:
                raise ValueError(f"Immutable dataset content changed: {filename}; import anew")
        queries = list(read_jsonl(directory / "queries.jsonl"))
        qrels = {row["qid"]: {} for row in queries}
        for row in read_qrels(directory / "qrels.tsv"):
            qrels[row["qid"]][row["passage_id"]] = row["relevance"]
        source_hash = hashlib.sha256(
            "".join(sha_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))).encode()
        ).hexdigest()
        cache = self.root / "cache" / manifest["hashes"]["corpus"] / source_hash
        cache.mkdir(parents=True, exist_ok=True)
        conditions, dense = [], None
        run_id = uuid.uuid4().hex
        warnings = list(manifest["warnings"]) + [
            "Unjudged passages count as nonrelevant; qrels are incomplete relevance judgments.",
            "95% paired percentile intervals are unadjusted; conservative vs raw nDCG@10 "
            "is the proposed primary comparison. Other contrasts are exploratory.",
            "Controlled category-tokenizer BM25 is not the official ArabicAnalyzer baseline.",
        ]
        try:
            for retriever in config.retrievers:
                if retriever == "dense":
                    dense = Dense(config)
                    warnings.append(
                        "Dense uses the recorded sentence-transformers architecture "
                        "and cosine protocol; this does not reproduce MIRACL baselines."
                    )
                presets = ["raw"] + [p for p in config.presets if p != "raw"]
                baseline = None
                for preset in presets:
                    if progress:
                        progress(
                            {
                                "stage": "condition",
                                "detail": f"{retriever} / {preset}",
                                "completed": len(conditions),
                                "total": len(config.retrievers) * len(presets),
                            }
                        )
                    document_preset = preset if config.normalization_side == "both" else "raw"
                    if retriever == "bm25":
                        index = cache / f"bm25-{document_preset}.sqlite"
                        metadata = build_bm25(
                            directory / "corpus.sqlite", index, document_preset, progress
                        )
                        searcher = BM25(index)
                    else:
                        dense.close()
                        metadata = dense.prepare(
                            directory / "corpus.sqlite",
                            cache,
                            manifest["hashes"]["corpus"],
                            document_preset,
                            progress,
                        )
                        searcher = dense
                    rows = []
                    try:
                        for i, query in enumerate(queries):
                            transformed = normalize(query["query"], preset)
                            ranked = searcher.search(transformed, config.top_k)
                            ids = [pid for pid, _score in ranked]
                            grades = qrels[query["qid"]]
                            positives = {pid for pid, grade in grades.items() if grade > 0}
                            metrics = {
                                "ndcg@10": graded_ndcg_at_k(ids, grades, 10),
                                "recall@5": recall_at_k(ids, positives, 5),
                                "recall@10": recall_at_k(ids, positives, 10),
                                "recall@100": recall_at_k(ids, positives, 100),
                                "mrr@10": mrr(ids, positives, 10),
                            }
                            rows.append(
                                {
                                    **query,
                                    "normalized_query": transformed,
                                    "query_changed": transformed != query["query"],
                                    "metrics": metrics,
                                    "ranking": [
                                        {
                                            "passage_id": pid,
                                            "score": score,
                                            "relevance": grades.get(pid),
                                        }
                                        for pid, score in ranked
                                    ],
                                }
                            )
                            if progress and (i + 1) % 25 == 0:
                                progress(
                                    {
                                        "stage": "retrieval",
                                        "detail": f"{retriever}/{preset}",
                                        "completed": i + 1,
                                        "total": len(queries),
                                    }
                                )
                    finally:
                        searcher.close()
                    if baseline is None:
                        baseline = rows
                    tags = sorted({tag for row in rows for tag in row["tags"]})
                    slices = {
                        "all": rows,
                        "query_changed": [r for r in rows if r["query_changed"]],
                        "query_unchanged": [r for r in rows if not r["query_changed"]],
                    }
                    slices.update(
                        {f"tag:{tag}": [r for r in rows if tag in r["tags"]] for tag in tags}
                    )
                    if progress:
                        progress({"stage": "statistics", "detail": f"{retriever}/{preset}"})
                    condition = {
                        "retriever": retriever,
                        "preset": preset,
                        "metrics": _mean(rows),
                        "changed_queries": sum(row["query_changed"] for row in rows),
                        "changed_documents": metadata["changed_documents"],
                        "per_query": rows,
                        "slices": {
                            key: {"query_count": len(group), "metrics": _mean(group)}
                            for key, group in slices.items()
                        },
                        "comparison_to_raw": paired_comparison(
                            rows, baseline, config.bootstrap_samples, config.seed
                        ),
                    }
                    conditions.append(condition)
            report = {
                "run_id": run_id,
                "dataset_id": manifest["dataset_id"],
                "created_at": _now(),
                "config": asdict(config),
                "dataset": manifest,
                "conditions": conditions,
                "warnings": warnings,
                "provenance": {
                    "engine_source_sha256": source_hash,
                    "python": platform.python_version(),
                    "sqlite": sqlite3.sqlite_version,
                    "platform": platform.platform(),
                    "tokenizer": TOKENIZER,
                    "document_fields": "title + newline + text",
                    "tie_break": "corpus ordinal ascending",
                    "trec_export_scores": "rank-based decreasing integers; raw scores in JSON",
                    "bm25": {
                        "k1": 1.2,
                        "b": 0.75,
                        "idf": "ln(1+(N-df+0.5)/(df+0.5))",
                        "query_term_frequency": "unique terms",
                    },
                    "dense": dense.runtime if dense else None,
                    "bootstrap": {
                        "unit": "query",
                        "paired": True,
                        "method": "percentile",
                        "confidence": 0.95,
                    },
                },
            }
            # Publish only completed reports; a failed condition leaves no successful run.
            self.export_trec(run_id, report=report)
            _write_json(self.root / "runs" / f"{run_id}.json", report)
            summary = {key: value for key, value in report.items() if key != "conditions"}
            summary["conditions"] = [
                {key: value for key, value in c.items() if key != "per_query"} for c in conditions
            ]
            _write_json(self.root / "runs" / f"{run_id}.summary.json", summary)
            return report
        finally:
            if dense:
                dense.close()

    def export_trec(self, run_id: str, output: Path | None = None, report: dict | None = None):
        run_id = self._id(run_id)
        report = report or self.get_run(run_id)
        output = Path(output) if output is not None else self.root / "runs" / f"{run_id}-trec"
        output.mkdir(parents=True, exist_ok=True)
        for condition in report["conditions"]:
            tag = f"{condition['retriever']}-{condition['preset']}"
            with (output / f"{tag}.trec").open("w", encoding="utf-8", newline="\n") as stream:
                for row in condition["per_query"]:
                    for rank, passage in enumerate(row["ranking"], 1):
                        # trec_eval ignores the rank column and breaks score ties by
                        # docno. Ordinal scores preserve the evaluated order exactly;
                        # original retrieval scores remain available in the JSON report.
                        export_score = len(row["ranking"]) - rank + 1
                        stream.write(
                            f"{row['qid']} Q0 {passage['passage_id']} {rank} {export_score} {tag}\n"
                        )
        return output
