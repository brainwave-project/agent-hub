"""Stream official MIRACL Arabic files at resolved immutable revisions.

No datasets loading script, remote code, or embedding model is executed here.
Even the reduced pilot scans the whole gzip corpus to avoid an order-selected
distractor pool. The qrel-informed pilot is intentionally NOT benchmark-comparable.
"""

from __future__ import annotations

import gzip
import hashlib
import heapq
import io
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

from .benchmark import _json_line, _write_json, sha_file

TOPICS_REVISION = "5be20db9509754dadad47689368639fcec739c00"
CORPUS_REVISION = "d921ec7e349ce0d28daf30b2da9da5ee698bef0d"


def _url(repo: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"


def resolve_revision(repo: str, revision: str) -> str:
    if re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        return revision.lower()
    url = (
        f"https://huggingface.co/api/datasets/{repo}/revision/"
        f"{urllib.parse.quote(revision, safe='')}"
    )
    with urllib.request.urlopen(url, timeout=120) as response:
        sha = json.load(response)["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Hugging Face did not return an immutable dataset revision")
    return sha


def _lines(url: str, compressed: bool = False):
    request = urllib.request.Request(url, headers={"User-Agent": "brainwave-normalization/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        binary = gzip.GzipFile(fileobj=response) if compressed else response
        with io.TextIOWrapper(binary, encoding="utf-8-sig") as stream:
            yield from stream


def select_queries(queries: list[dict], limit: int | None, seed: int) -> list[dict]:
    if limit is not None and (isinstance(limit, bool) or limit <= 0):
        raise ValueError("query-limit must be positive; omit it for all queries in full mode")
    ordered = sorted(
        queries, key=lambda q: (hashlib.sha256(f"{seed}:{q['qid']}".encode()).hexdigest(), q["qid"])
    )
    selected = ordered if limit is None else ordered[:limit]
    return sorted(selected, key=lambda q: q["qid"])


def _tags(query: str) -> list[str]:
    tags = []
    if any("\u064b" <= c <= "\u0652" for c in query):
        tags.append("contains_diacritics")
    if "ـ" in query:
        tags.append("contains_tatweel")
    if any(c in query for c in "أإآ"):
        tags.append("contains_alef_variants")
    if "ى" in query:
        tags.append("contains_alef_maqsura")
    if "ة" in query:
        tags.append("contains_taa_marbuta")
    return tags


def prepare_miracl(
    output: Path,
    mode: str = "pilot",
    query_limit: int | None = None,
    distractors: int = 1000,
    seed: int = 42,
    dataset_revision: str = TOPICS_REVISION,
    corpus_revision: str = CORPUS_REVISION,
    progress=None,
    split: str = "dev",
) -> dict:
    if mode not in ("pilot", "full"):
        raise ValueError("mode must be pilot or full")
    if split not in ("train", "dev"):
        raise ValueError("split must be train or dev")
    if distractors < 0:
        raise ValueError("distractors must be nonnegative")
    if mode == "pilot" and query_limit is None:
        query_limit = 100
    output = Path(output)
    if output.exists():
        raise ValueError("output directory already exists; choose a new preparation directory")
    topics_sha = resolve_revision("miracl/miracl", dataset_revision)
    corpus_sha = resolve_revision("miracl/miracl-corpus", corpus_revision)
    topics_url = _url(
        "miracl/miracl", topics_sha, f"miracl-v1.0-ar/topics/topics.miracl-v1.0-ar-{split}.tsv"
    )
    qrels_url = _url(
        "miracl/miracl", topics_sha, f"miracl-v1.0-ar/qrels/qrels.miracl-v1.0-ar-{split}.tsv"
    )
    all_queries = []
    for line in _lines(topics_url):
        if line.strip():
            qid, query = line.rstrip("\n\r").split("\t", 1)
            all_queries.append({"qid": qid, "query": query, "tags": _tags(query)})
    selected = select_queries(all_queries, query_limit, seed)
    selected_ids = {q["qid"] for q in selected}
    if not selected:
        raise ValueError(f"No Arabic {split} queries found")
    judgments, required = [], set()
    for line in _lines(qrels_url):
        if not line.strip():
            continue
        qid, _iteration, pid, grade = line.split()
        if qid in selected_ids:
            judgments.append((qid, pid, int(grade)))
            required.add(pid)
    if not judgments:
        raise ValueError(f"No qrels for the selected {split} queries")
    output.mkdir(parents=True)
    # This sentinel survives interrupted network transfers, so partial preparations are obvious.
    incomplete = output / "INCOMPLETE"
    incomplete.write_text(
        "Preparation has not completed. Do not import these files.\n", encoding="utf-8"
    )
    with (output / "queries.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for query in selected:
            stream.write(_json_line(query))
    with (output / "qrels.tsv").open("w", encoding="utf-8", newline="\n") as stream:
        for qid, pid, grade in sorted(judgments):
            stream.write(f"{qid}\t0\t{pid}\t{grade}\n")
    found, selected_passages, background = set(), {}, []
    scanned = written = 0
    urls = [
        _url("miracl/miracl-corpus", corpus_sha, f"miracl-corpus-v1.0-ar/docs-{shard}.jsonl.gz")
        for shard in range(5)
    ]
    with (output / "corpus.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for shard, url in enumerate(urls):
            if progress:
                progress({"stage": "download", "detail": f"Arabic corpus shard {shard + 1}/5"})
            for line in _lines(url, compressed=True):
                if not line.strip():
                    continue
                original = json.loads(line)
                pid = original["docid"]
                row = {"passage_id": pid, "title": original["title"], "text": original["text"]}
                scanned += 1
                if pid in required:
                    found.add(pid)
                if mode == "full":
                    stream.write(_json_line(row))
                    written += 1
                elif pid in required:
                    selected_passages[pid] = row
                elif distractors:
                    priority = int(hashlib.sha256(f"{seed}:{pid}".encode()).hexdigest(), 16)
                    entry = (-priority, pid, row)
                    if len(background) < distractors:
                        heapq.heappush(background, entry)
                    elif priority < -background[0][0]:
                        heapq.heapreplace(background, entry)
                if progress and scanned % 50000 == 0:
                    progress({"stage": "prepare", "detail": f"Scanned {scanned} passages"})
        if required - found:
            raise ValueError(
                f"Corpus is missing {len(required - found)} judged passages; "
                "preparation is incomplete"
            )
        if mode == "pilot":
            selected_passages.update({pid: row for _priority, pid, row in background})
            for pid in sorted(selected_passages):
                stream.write(_json_line(selected_passages[pid]))
            written = len(selected_passages)
    scope = (
        "miracl-pilot"
        if mode == "pilot"
        else f"full-corpus-all-{split}-queries"
        if len(selected) == len(all_queries)
        else "full-corpus-query-subset"
    )
    provenance = {
        "source": "MIRACL",
        "language": "ar",
        "split": split,
        "scope": scope,
        "corpus_scope": "qrel-complete-pilot" if mode == "pilot" else "full",
        "dataset_revision": topics_sha,
        "corpus_revision": corpus_sha,
        "query_selection": {
            "method": "lowest sha256(seed:qid) before retrieval",
            "seed": seed,
            "requested_limit": query_limit,
            "available_queries": len(all_queries),
            "selected_queries": len(selected),
            "selected_qids": sorted(selected_ids),
        },
        "corpus_selection": {
            "method": "all passages"
            if mode == "full"
            else "all judged passages for selected queries + lowest sha256(seed:passage_id)",
            "requested_distractors": distractors if mode == "pilot" else None,
            "scanned_passages": scanned,
            "included_passages": written,
            "label_informed": mode == "pilot",
        },
        "sources": {"topics": topics_url, "qrels": qrels_url, "corpus": urls},
        "license_notice": "Retain the MIRACL dataset citation and original source licenses; "
        "Wikipedia-derived passage content requires its upstream attribution.",
        "prepared_hashes": {
            name: sha_file(output / name) for name in ("corpus.jsonl", "queries.jsonl", "qrels.tsv")
        },
    }
    _write_json(output / "provenance.json", provenance)
    incomplete.unlink()
    return provenance
