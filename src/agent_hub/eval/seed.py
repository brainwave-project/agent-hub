"""Seed the eval corpus into know-hub and wait for it to become searchable.

Idempotent: documents already present (matched by title) are not re-uploaded,
so `make evals` is cheap to re-run. Pass `--force` to re-upload anyway.

Two corpus variants are seeded for Arabic:

* **raw** — the corpus exactly as written, with its orthographic inconsistencies
  intact. This is what a real ingested corpus looks like.
* **normalized** (`~norm` suffix) — the same documents passed through
  `arabic.normalize` at ingestion time.

Both exist because query-side normalisation alone cannot fix every trap. When
the *document* carries the marks — a fully-vowelled text, say — normalising the
query does nothing, since the query had no diacritics to strip. Only
re-embedding the corpus closes that gap. Seeding both variants turns that into
a measured difference instead of an assertion, and makes the cost of the fix
(a full reindex) concrete.

The wait step matters. Indexing is asynchronous — know-hub returns 201 from the
upload immediately and embeds in a background worker driven by the
DOCUMENT_UPLOADED Kafka event. Scoring retrieval before that finishes produces a
zero that looks like a retrieval failure but is really a race, which is exactly
the kind of result that sends you debugging the wrong component.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field

from agent_hub.eval import configure_stdout
from agent_hub.eval.arabic.normalize import CONSERVATIVE, NormalizeConfig, normalize
from agent_hub.eval.datasets import LANGUAGES, load_corpus
from agent_hub.eval.knowhub_client import KnowHubEvalClient

#: Statuses the semantic worker will not move away from.
TERMINAL = {"INDEXED", "SKIPPED", "FAILED"}
#: The only terminal status that means "retrievable".
SEARCHABLE = "INDEXED"
#: Title suffix marking the ingestion-normalised copy of a document.
NORM_SUFFIX = "~norm"


def base_doc_id(title: str) -> str:
    """Map a know-hub title back to its corpus doc id, dropping any variant tag."""
    return title[: -len(NORM_SUFFIX)] if title.endswith(NORM_SUFFIX) else title


def is_normalized(title: str) -> bool:
    return title.endswith(NORM_SUFFIX)


def build_documents(
    languages: tuple[str, ...],
    *,
    normalized_langs: tuple[str, ...] = ("ar",),
    config: NormalizeConfig = CONSERVATIVE,
) -> dict[str, str]:
    """`{title: body}` for every document to seed, both variants included."""
    documents: dict[str, str] = {}
    for lang in languages:
        corpus = load_corpus(lang)
        documents.update(corpus)
        if lang in normalized_langs:
            for doc_id, body in corpus.items():
                documents[f"{doc_id}{NORM_SUFFIX}"] = normalize(body, config)
    return documents


@dataclass
class SeedResult:
    uploaded: dict[str, str] = field(default_factory=dict)   # title -> documentId
    reused: dict[str, str] = field(default_factory=dict)
    statuses: dict[str, str] = field(default_factory=dict)   # title -> terminal status

    @property
    def all_ids(self) -> dict[str, str]:
        return {**self.reused, **self.uploaded}

    @property
    def indexed(self) -> list[str]:
        return [t for t, s in self.statuses.items() if s == SEARCHABLE]

    @property
    def not_indexed(self) -> dict[str, str]:
        return {t: s for t, s in self.statuses.items() if s != SEARCHABLE}


def _existing_by_title(client: KnowHubEvalClient) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in client.list_documents(limit=100):
        title = item.get("title")
        doc_id = item.get("documentId") or item.get("id")
        if title and doc_id:
            out[str(title)] = str(doc_id)
    return out


def seed(
    client: KnowHubEvalClient,
    languages: tuple[str, ...] = LANGUAGES,
    *,
    force: bool = False,
    timeout_s: float = 900.0,
    poll_interval_s: float = 3.0,
    verbose: bool = True,
) -> SeedResult:
    def say(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    documents = build_documents(languages)
    existing = {} if force else _existing_by_title(client)
    result = SeedResult()

    for title, body in documents.items():
        if title in existing:
            result.reused[title] = existing[title]
        else:
            result.uploaded[title] = client.upload_markdown(title, body)

    say(f"  {len(documents)} documents — uploaded {len(result.uploaded)}, "
        f"reused {len(result.reused)}")

    # --- wait for the async embedding worker ----------------------------
    pending = dict(result.all_ids)
    deadline = time.monotonic() + timeout_s
    last_report = 0.0

    while pending and time.monotonic() < deadline:
        for title, document_id in list(pending.items()):
            try:
                status = client.embedding_status(document_id)
            except Exception:            # noqa: BLE001 — transient while booting
                continue
            if status in TERMINAL:
                result.statuses[title] = status
                pending.pop(title, None)

        if pending:
            now = time.monotonic()
            if verbose and now - last_report > 15:
                say(f"  embedding... {len(result.statuses)}/{len(documents)} done")
                last_report = now
            time.sleep(poll_interval_s)

    for title, document_id in pending.items():
        result.statuses[title] = f"TIMEOUT (id={document_id})"

    return result


def main() -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description="Seed the eval corpus into know-hub")
    parser.add_argument("--know-hub-url", default="http://know-hub:3002")
    parser.add_argument("--lang", action="append", choices=list(LANGUAGES))
    parser.add_argument("--force", action="store_true", help="re-upload even if title exists")
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    languages = tuple(args.lang) if args.lang else LANGUAGES

    with KnowHubEvalClient(base_url=args.know_hub_url) as client:
        if not client.ready():
            print(f"know-hub not ready at {args.know_hub_url}", file=sys.stderr)
            return 2
        result = seed(client, languages, force=args.force, timeout_s=args.timeout)

    print(f"  indexed {len(result.indexed)}/{len(result.statuses)}")
    if result.not_indexed:
        print("  NOT searchable:", file=sys.stderr)
        for title, status in sorted(result.not_indexed.items()):
            print(f"    {title}: {status}", file=sys.stderr)
        # For a markdown corpus, SKIPPED means the mime type did not match
        # know-hub's TEXT_MIME_RE, or SEMANTIC_WORKER_ENABLED is false.
        # Either way it is a setup bug, not an acceptable outcome.
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
