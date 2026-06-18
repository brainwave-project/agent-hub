"""Adaptive chunking strategies (§7.6) — the deterministic tool the Chunker runs.

The Classifier *reasons* about which strategy fits; this module *executes* it
(Principle 3: reasoning chooses, code does). Each strategy returns TextChunks
with an approximate word-boundary token count, matching Phase-1's estimator so
embeddings stay comparable across the corpus.

  clause — split on numbered clauses/sections (contracts): a whole obligation per chunk
  slide  — one chunk per slide/page-break (decks)
  turn   — group by speaker turns (chat/transcripts)
  fixed  — 512 tok / 64 overlap (Phase-1 default; the general baseline)
"""

from __future__ import annotations

import re

from agent_hub.domain.models import TextChunk

FIXED_MAX_TOKENS = 512
FIXED_OVERLAP = 64


def _tokens(text: str) -> int:
    return len(text.split())


def _emit(parts: list[str]) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    seq = 0
    for p in parts:
        p = p.strip()
        if not p:
            continue
        chunks.append(TextChunk(seq=seq, text=p, token_count=_tokens(p)))
        seq += 1
    return chunks


def _fixed(text: str, max_tokens: int = FIXED_MAX_TOKENS, overlap: int = FIXED_OVERLAP) -> list[TextChunk]:
    words = text.split()
    if not words:
        return []
    step = max(1, max_tokens - overlap)
    parts = [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), step)]
    return _emit(parts)


def _clause(text: str) -> list[TextChunk]:
    # Split before numbered clauses ("1.", "1.2", "Section 3", "Article IV").
    pattern = re.compile(r"(?=^\s*(?:\d+(?:\.\d+)*\.?|Section\s+\d+|Article\s+[IVXLC\d]+)\b)", re.M)
    parts = [p for p in pattern.split(text) if p.strip()]
    out: list[TextChunk] = []
    # A single clause can still be huge — fixed-split any over the cap.
    for p in parts:
        out.extend(_fixed(p) if _tokens(p) > FIXED_MAX_TOKENS else _emit([p]))
    for i, c in enumerate(out):
        c.seq = i
    return out or _fixed(text)


def _slide(text: str) -> list[TextChunk]:
    # Form-feed is the canonical page/slide break; fall back to blank-line runs.
    parts = text.split("\f") if "\f" in text else re.split(r"\n\s*\n\s*\n+", text)
    return _emit(parts) or _fixed(text)


def _turn(text: str) -> list[TextChunk]:
    # Group consecutive lines until a new "Speaker:" prefix starts a turn.
    speaker = re.compile(r"^\s*([A-Z][\w .'-]{0,40}):\s")
    turns: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if speaker.match(line) and current:
            turns.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        turns.append("\n".join(current))
    return _emit(turns) or _fixed(text)


_STRATEGIES = {"clause": _clause, "slide": _slide, "turn": _turn, "fixed": _fixed}


def chunk_text(text: str, strategy: str) -> list[TextChunk]:
    return _STRATEGIES.get(strategy, _fixed)(text)
