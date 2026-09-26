"""Preview know-hub's chunker, and find where evidence lands in it.

`chunk_text` is a line-for-line port of `chunkText` in core/src/ai/chunker.ts,
which know-hub's semantic worker calls with maxTokens=512, overlap=64. Keep the
two in step: a preview that chunks differently from the indexer is worse than
no preview.

The question this module answers for a dataset designer: *can the evidence be
retrieved as a chunk at all?* An answer sentence that straddles a chunk
boundary exists in no single chunk. Chunk-level recall for that question is
capped below 1.0 by construction, whatever the retriever does, and the cap
moves every time chunk size changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "KNOWHUB_MAX_TOKENS",
    "KNOWHUB_OVERLAP",
    "WORDS_PER_TOKEN",
    "Chunk",
    "chunk_text",
    "norm_text",
    "locate_evidence",
]

#: Mirrors know-hub/src/ai/worker.ts — the chunking the index actually uses.
KNOWHUB_MAX_TOKENS = 512
KNOWHUB_OVERLAP = 64
#: Mirrors core/src/ai/chunker.ts.
WORDS_PER_TOKEN = 0.75


@dataclass(frozen=True)
class Chunk:
    index: int
    content: str
    token_count: int
    start_word: int
    end_word: int  # exclusive


def _approx_tokens(word_count: int) -> int:
    return math.ceil(word_count / WORDS_PER_TOKEN)


def chunk_text(
    text: str, max_tokens: int = KNOWHUB_MAX_TOKENS, overlap: int = KNOWHUB_OVERLAP
) -> list[Chunk]:
    words = text.split()
    if not words:
        return []
    max_words = max(1, math.floor(max_tokens * WORDS_PER_TOKEN))
    overlap_words = max(0, min(max_words - 1, math.floor(overlap * WORDS_PER_TOKEN)))
    step = max(1, max_words - overlap_words)

    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        piece = words[start : start + max_words]
        chunks.append(
            Chunk(
                index=len(chunks),
                content=" ".join(piece),
                token_count=_approx_tokens(len(piece)),
                start_word=start,
                end_word=start + len(piece),
            )
        )
        if start + max_words >= len(words):
            break
        start += step
    return chunks


def norm_text(text: str) -> str:
    """Whitespace-collapsed, case-folded. The only normalisation evidence matching uses.

    Deliberately NOT Arabic orthographic normalisation: evidence is meant to be
    copied verbatim from the document, and a span that only matches after
    normalisation is a labelling error worth surfacing, not hiding.
    """
    return " ".join(text.split()).casefold()


def locate_evidence(body: str, evidence: str, chunks: list[Chunk]) -> dict:
    span = norm_text(evidence)
    found = span in norm_text(body)
    containing = [c.index for c in chunks if span in norm_text(c.content)]
    return {
        "found_in_doc": found,
        "chunks_containing": containing,
        # Present in the document but in no single chunk: split by a boundary.
        "straddles_boundary": found and not containing,
    }
