"""Disk-backed controlled BM25 and optional exact dense retrieval.

BM25's index and SQL aggregation live on disk. Dense embeddings are float32
memmaps and scoring reads bounded blocks; neither loads the corpus into RAM.
The initial full MIRACL run is still expensive (six indexes/embedding passes).
"""

from __future__ import annotations

import hashlib
import heapq
import importlib.metadata
import json
import math
import os
import sqlite3
import uuid
from collections import Counter
from pathlib import Path

from .text import normalize, tokenize


def corpus_rows(path: Path):
    with sqlite3.connect(path) as conn:
        yield from conn.execute(
            "SELECT ordinal, passage_id, title, text FROM corpus ORDER BY ordinal"
        )


def document_text(title: str, text: str) -> str:
    return f"{title}\n{text}" if title else text


def build_bm25(corpus: Path, path: Path, preset: str, progress=None) -> dict:
    if path.exists():
        with sqlite3.connect(path) as conn:
            return json.loads(conn.execute("SELECT value FROM metadata").fetchone()[0])
    temp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    conn = sqlite3.connect(temp)
    try:
        conn.executescript("""
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=FILE;
            CREATE TABLE docs (ordinal INTEGER PRIMARY KEY, passage_id TEXT, length INTEGER);
            CREATE TABLE postings (term TEXT, ordinal INTEGER, frequency INTEGER);
            CREATE TABLE metadata (value TEXT);
        """)
        count = total_length = changed = 0
        postings = []
        for ordinal, passage_id, title, text in corpus_rows(corpus):
            original = document_text(title, text)
            transformed = normalize(original, preset)
            tokens = tokenize(transformed)
            count += 1
            changed += transformed != original
            total_length += len(tokens)
            conn.execute("INSERT INTO docs VALUES (?, ?, ?)", (ordinal, passage_id, len(tokens)))
            postings.extend((term, ordinal, n) for term, n in Counter(tokens).items())
            if len(postings) >= 10000:
                conn.executemany("INSERT INTO postings VALUES (?, ?, ?)", postings)
                postings.clear()
            if progress and count % 10000 == 0:
                progress({"stage": "index", "detail": f"BM25 {preset}: {count} passages"})
        if postings:
            conn.executemany("INSERT INTO postings VALUES (?, ?, ?)", postings)
        conn.executescript("""
            CREATE INDEX term_postings ON postings(term);
            CREATE TABLE terms AS SELECT term, COUNT(*) AS df FROM postings GROUP BY term;
            CREATE UNIQUE INDEX term_lookup ON terms(term);
        """)
        metadata = {
            "documents": count,
            "avg_length": total_length / count,
            "changed_documents": changed,
        }
        conn.execute("INSERT INTO metadata VALUES (?)", (json.dumps(metadata),))
        conn.commit()
        conn.close()
        os.replace(temp, path)
        return metadata
    except BaseException:
        conn.close()
        temp.unlink(missing_ok=True)
        raise


class BM25:
    def __init__(self, path: Path, k1: float = 1.2, b: float = 0.75):
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA temp_store=FILE")
        self.meta = json.loads(self.conn.execute("SELECT value FROM metadata").fetchone()[0])
        self.k1, self.b = k1, b
        self.conn.execute("CREATE TEMP TABLE query_terms (term TEXT PRIMARY KEY, idf REAL)")

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        self.conn.execute("DELETE FROM query_terms")
        for term in sorted(set(tokenize(query))):
            row = self.conn.execute("SELECT df FROM terms WHERE term=?", (term,)).fetchone()
            if row:
                df = row[0]
                idf = math.log(1 + (self.meta["documents"] - df + 0.5) / (df + 0.5))
                self.conn.execute("INSERT INTO query_terms VALUES (?, ?)", (term, idf))
        # Only matching documents are returned; never pad with arbitrary zero-score rows.
        return self.conn.execute(
            """
            SELECT d.passage_id,
                SUM(q.idf * p.frequency * (? + 1) /
                    (p.frequency + ? * (1 - ? + ? * d.length / ?))) AS score
            FROM query_terms q CROSS JOIN postings p INDEXED BY term_postings CROSS JOIN docs d
            WHERE p.term=q.term AND d.ordinal=p.ordinal
            GROUP BY d.ordinal ORDER BY score DESC, d.ordinal ASC LIMIT ?
        """,
            (self.k1, self.k1, self.b, self.b, self.meta["avg_length"] or 1, k),
        ).fetchall()

    def close(self):
        self.conn.close()


class Dense:
    """One fixed sentence-transformers encoder; exact cosine search, no ANN noise."""

    def __init__(self, config):
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ValueError("Dense retrieval requires: uv sync --extra rerank") from exc
        self.np = np
        self.config = config
        self.model = SentenceTransformer(
            config.dense_model, revision=config.dense_revision, trust_remote_code=False
        )
        self.model.max_seq_length = config.dense_max_seq_length
        # Use explicit strings below, never an implicit model-specific prompt.
        self.model.default_prompt_name = None
        self.runtime = {
            "sentence_transformers": importlib.metadata.version("sentence-transformers"),
            "numpy": np.__version__,
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "device": str(self.model.device),
            "architecture": str(self.model),
            "similarity": "dot product of L2-normalized float32 embeddings (cosine)",
        }

    def prepare(self, corpus: Path, cache_dir: Path, corpus_hash: str, preset: str, progress=None):
        key = hashlib.sha256(
            json.dumps(
                {
                    "corpus": corpus_hash,
                    "preset": preset,
                    "model": self.config.dense_model,
                    "revision": self.config.dense_revision,
                    "max_seq_length": self.config.dense_max_seq_length,
                    "document_prefix": self.config.document_prefix,
                    "runtime": self.runtime,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self.path = cache_dir / f"dense-{key}.npy"
        meta_path = cache_dir / f"dense-{key}.json"
        self.corpus = corpus
        if self.path.exists() and meta_path.exists():
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.embeddings = self.np.load(self.path, mmap_mode="r")
            return self.meta
        with sqlite3.connect(corpus) as conn:
            count = conn.execute("SELECT COUNT(*) FROM corpus").fetchone()[0]
        temp = self.path.with_suffix(f".{uuid.uuid4().hex}.tmp.npy")
        batch, offset, changed, target = [], 0, 0, None
        try:
            for _ordinal, _pid, title, text in corpus_rows(corpus):
                original = document_text(title, text)
                transformed = normalize(original, preset)
                changed += transformed != original
                batch.append(self.config.document_prefix + transformed)
                if len(batch) == self.config.dense_batch_size:
                    target, offset = self._write_batch(batch, temp, target, count, offset)
                    batch.clear()
                    if progress:
                        progress(
                            {
                                "stage": "embedding",
                                "detail": f"Dense {preset}: {offset}",
                                "completed": offset,
                                "total": count,
                            }
                        )
            if batch:
                target, offset = self._write_batch(batch, temp, target, count, offset)
            target.flush()
            del target
            target = None
            os.replace(temp, self.path)
            self.meta = {
                "documents": count,
                "changed_documents": changed,
                "runtime": self.runtime,
                "cache_key": key,
            }
            meta_path.write_text(json.dumps(self.meta), encoding="utf-8")
            self.embeddings = self.np.load(self.path, mmap_mode="r")
            return self.meta
        except BaseException:
            del target
            temp.unlink(missing_ok=True)
            raise

    def _write_batch(self, batch, path, target, count, offset):
        vectors = self.model.encode(
            batch,
            batch_size=self.config.dense_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        if not self.np.isfinite(vectors).all():
            raise ValueError("Dense encoder returned non-finite passage embeddings")
        if target is None:
            target = self.np.lib.format.open_memmap(
                path, mode="w+", dtype="float32", shape=(count, vectors.shape[1])
            )
        target[offset : offset + len(vectors)] = vectors
        return target, offset + len(vectors)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        vector = self.model.encode(
            [self.config.query_prefix + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0].astype("float32")
        if not self.np.isfinite(vector).all():
            raise ValueError("Dense encoder returned a non-finite query embedding")
        best = []
        block_size = 16384
        for start in range(0, len(self.embeddings), block_size):
            scores = self.embeddings[start : start + block_size] @ vector
            # Stable tie order by corpus ordinal. Partition-only selection can drop tied IDs.
            order = self.np.lexsort((self.np.arange(len(scores)), -scores))[:k]
            for local in order:
                ordinal = start + int(local)
                candidate = (float(scores[local]), -ordinal)
                if len(best) < k:
                    heapq.heappush(best, candidate)
                elif candidate > best[0]:
                    heapq.heapreplace(best, candidate)
        result = []
        with sqlite3.connect(self.corpus) as conn:
            for score, negative_ordinal in sorted(best, reverse=True):
                pid = conn.execute(
                    "SELECT passage_id FROM corpus WHERE ordinal=?", (-negative_ordinal,)
                ).fetchone()[0]
                result.append((pid, score))
        return result

    def close(self):
        if hasattr(self, "embeddings"):
            del self.embeddings
