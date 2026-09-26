"""Synthetic question generation — a fast way to grow a set, and to fool yourself.

A local LLM (the stack's Ollama, so no document text leaves the network) reads a
chunk and proposes questions with a verbatim evidence sentence. Candidates land
in a REVIEW QUEUE; nothing is scored until a human accepts it.

What the queue measures for each candidate, because these are the known ways
synthetic data misleads:

    evidence_found    the quoted sentence really is in the document (LLMs
                      paraphrase when asked to quote)
    lexical_overlap   share of the question's content words that appear in the
                      document. Generated questions echo their source text, so
                      they are easier for lexical/hybrid search than real user
                      questions — a synthetic-heavy set flatters BM25-style
                      retrieval.
    copied            the question appears verbatim in the document

Structural limits no filter fixes: every synthetic question is answerable from
ONE chunk of ONE document (no multi-hop, no unanswerable), and its phrasing
distribution is the model's, not your users'. Keep the holdout human-written.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re

import httpx

from agent_hub.eval.lab.chunking import chunk_text, locate_evidence, norm_text
from agent_hub.eval.lab.jobs import Job
from agent_hub.eval.lab.models import Candidate
from agent_hub.eval.lab.store import Bundle

__all__ = ["PROMPT_VERSION", "STYLES", "OllamaGenerator", "generate_candidates", "lexical_overlap"]

PROMPT_VERSION = "gen-v1"

STYLES: dict[str, str] = {
    "user": (
        "Write questions the way a real employee would type them into a search box. "
        "Do NOT reuse distinctive phrases from the text; use everyday wording."
    ),
    "keyword": "Write short keyword-style search queries of 2 to 5 words, no question mark.",
    "hard": (
        "Write questions that paraphrase heavily: use synonyms and different sentence structure "
        "so that few words overlap with the text, while the text still clearly answers them."
    ),
}

_WORD = re.compile(r"\w+", re.UNICODE)
_LANG_NAMES = {"en": "English", "ar": "Arabic"}


def lexical_overlap(query: str, body: str) -> float:
    """Share of the query's content words (4+ characters) that occur in the document."""
    doc_words = set(_WORD.findall(norm_text(body)))
    words = [w for w in _WORD.findall(norm_text(query)) if len(w) >= 4]
    if not words:
        return 0.0
    return sum(1 for w in words if w in doc_words) / len(words)


class OllamaGenerator:
    def __init__(self, base_url: str, model: str, timeout: float = 240.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(
        self, text: str, lang: str, style: str, n: int, temperature: float, seed: int
    ) -> list[dict]:
        system = (
            "You write evaluation questions for a document retrieval system. "
            "Reply with a single JSON object and nothing else."
        )
        user = (
            f"{STYLES[style]}\n"
            f"Write exactly {n} question(s) in {_LANG_NAMES.get(lang, lang)} that the text below answers.\n"
            "For each, copy ONE sentence from the text EXACTLY, character for character, as evidence.\n"
            'Return: {"questions": [{"query": "...", "evidence": "...", "answer": "...", '
            '"difficulty": "easy|medium|hard"}]}\n\n'
            f"TEXT:\n<<<\n{text}\n>>>"
        )
        response = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "format": "json",
                "stream": False,
                "options": {"temperature": temperature, "seed": seed, "num_predict": 900},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "{}")
        parsed = json.loads(content)
        questions = parsed.get("questions", []) if isinstance(parsed, dict) else []
        return [q for q in questions if isinstance(q, dict) and str(q.get("query", "")).strip()]


def generate_candidates(
    generator: OllamaGenerator,
    bundle: Bundle,
    doc_ids: list[str] | None,
    per_doc: int,
    style: str,
    temperature: float,
    seed: int,
    existing: list[dict],
    job: Job | None = None,
) -> dict:
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}")
    targets = [d for d in sorted(bundle.corpus) if not doc_ids or d in doc_ids]
    known_ids = {c["id"] for c in existing}
    created: list[Candidate] = []
    failures: list[str] = []
    if job:
        job.progress(0, len(targets), "generating")

    for n, doc_id in enumerate(targets, start=1):
        doc = bundle.corpus[doc_id]
        chunks = chunk_text(doc.body)
        if not chunks:
            continue
        rng = random.Random(f"{seed}:{doc_id}")
        picked = rng.sample(chunks, k=min(per_doc, len(chunks)))
        per_chunk = math.ceil(per_doc / len(picked))
        produced = 0
        for chunk in picked:
            try:
                questions = generator.generate(
                    chunk.content, doc.lang, style, per_chunk, temperature, seed
                )
            except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
                failures.append(f"{doc_id}#{chunk.index}: {type(exc).__name__}: {exc}")
                continue
            for q in questions:
                if produced >= per_doc:
                    break
                query = str(q.get("query", "")).strip()
                evidence = str(q.get("evidence", "")).strip()
                cid = "cand-" + hashlib.sha256(f"{doc_id}:{query}".encode()).hexdigest()[:12]
                if cid in known_ids:
                    continue
                located = locate_evidence(doc.body, evidence, chunks) if len(evidence) >= 3 else {}
                q_norm = norm_text(query).rstrip("?.!؟")
                created.append(
                    Candidate(
                        id=cid,
                        doc_id=doc_id,
                        chunk_index=chunk.index,
                        query=query,
                        evidence=evidence,
                        answer=str(q.get("answer", "")).strip(),
                        difficulty=str(q.get("difficulty", "medium")).lower()
                        if str(q.get("difficulty", "medium")).lower() in ("easy", "medium", "hard")
                        else "medium",
                        style=style,
                        model=generator.model,
                        prompt_version=PROMPT_VERSION,
                        temperature=temperature,
                        seed=seed,
                        evidence_found=bool(located.get("found_in_doc")),
                        lexical_overlap=lexical_overlap(query, doc.body),
                        copied=len(q_norm.split()) >= 4 and q_norm in norm_text(doc.body),
                    )
                )
                known_ids.add(cid)
                produced += 1
        if job:
            job.progress(n, message=f"{doc_id}: {produced} candidate(s)")

    return {"created": [c.model_dump() for c in created], "failures": failures}
