"""Loading the eval corpus and golden question sets.

Dataset discipline enforced here:

* **Versioned filenames.** Golden sets are `golden_v1.jsonl`, `golden_v2.jsonl`.
  A version is never edited in place — changing questions under a fixed name
  makes two runs silently incomparable, which is the most common way an eval
  suite starts lying.
* **Held-out records.** Records flagged `holdout: true` are excluded by default.
  Tuning against your whole set means your score measures how well you fit the
  set, not how well the system works.
* **Stratification metadata.** Every record carries `difficulty` and `trap` so
  aggregate scores can be broken down. A flat mean over mixed difficulty hides
  exactly the failures worth finding.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Iterator
from dataclasses import dataclass, field

__all__ = [
    "GoldenRecord",
    "EVAL_DIR",
    "CORPUS_DIR",
    "DATASETS_DIR",
    "LANGUAGES",
    "load_golden",
    "load_corpus",
    "corpus_doc_ids",
]

EVAL_DIR = pathlib.Path(__file__).resolve().parent
CORPUS_DIR = EVAL_DIR / "corpus"
DATASETS_DIR = EVAL_DIR / "datasets"
LANGUAGES = ("en", "ar")


@dataclass(frozen=True)
class GoldenRecord:
    """One evaluated question.

    `relevant_doc_ids` are corpus document ids (the `.md` stem), which is also
    the `title` know-hub stores at upload time — that is the join key the
    retrieval runner uses to turn search hits back into ids.
    """

    id: str
    lang: str
    query: str
    relevant_doc_ids: list[str]
    difficulty: str = "medium"
    trap: str = "none"
    ground_truth: str = ""
    holdout: bool = False
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, raw: dict) -> GoldenRecord:
        known = {
            "id",
            "lang",
            "query",
            "relevant_doc_ids",
            "difficulty",
            "trap",
            "ground_truth",
            "holdout",
        }
        return cls(
            id=raw["id"],
            lang=raw["lang"],
            query=raw["query"],
            relevant_doc_ids=list(raw["relevant_doc_ids"]),
            difficulty=raw.get("difficulty", "medium"),
            trap=raw.get("trap", "none"),
            ground_truth=raw.get("ground_truth", ""),
            holdout=bool(raw.get("holdout", False)),
            extra={k: v for k, v in raw.items() if k not in known},
        )


def _golden_path(lang: str, version: str) -> pathlib.Path:
    return DATASETS_DIR / lang / f"golden_{version}.jsonl"


def load_golden(
    lang: str,
    version: str = "v1",
    *,
    include_holdout: bool = False,
    only_holdout: bool = False,
) -> list[GoldenRecord]:
    """Read one language's golden set.

    `include_holdout=False` (the default) is the tuning view. Pass
    `only_holdout=True` for the final, honest number — the one you are allowed
    to quote precisely because you never tuned against it.
    """
    path = _golden_path(lang, version)
    if not path.exists():
        raise FileNotFoundError(f"no golden set at {path}")

    records: list[GoldenRecord] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            records.append(GoldenRecord.from_json(json.loads(line)))
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"{path}:{line_no} — malformed golden record: {exc}") from exc

    if only_holdout:
        return [r for r in records if r.holdout]
    if not include_holdout:
        return [r for r in records if not r.holdout]
    return records


def load_corpus(lang: str) -> dict[str, str]:
    """`{doc_id: markdown}` for one language. Doc id is the filename stem."""
    directory = CORPUS_DIR / lang
    if not directory.exists():
        raise FileNotFoundError(f"no corpus at {directory}")
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(directory.glob("*.md"))}


def corpus_doc_ids(lang: str) -> set[str]:
    return {p.stem for p in (CORPUS_DIR / lang).glob("*.md")}


def iter_all_golden(
    version: str = "v1", *, include_holdout: bool = False
) -> Iterator[GoldenRecord]:
    for lang in LANGUAGES:
        yield from load_golden(lang, version, include_holdout=include_holdout)
