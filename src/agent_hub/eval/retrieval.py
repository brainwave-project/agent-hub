"""Retrieval evaluation — run a golden set through know-hub and score the ranking.

A *variant* is one complete retrieval configuration: which language, whether the
query is normalised, which corpus copy is searched, and whether the lexical half
of hybrid search is on. Every variant runs the same questions, so any two are
directly comparable with a paired test.

The comparisons this makes possible are the point:

    raw          vs  query-norm   how much does normalising the query alone buy?
    query-norm   vs  both-norm    how much more requires a full reindex?
    hybrid       vs  vector-only  what is the 'english' tsvector contributing?
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from agent_hub.eval.arabic.normalize import CONSERVATIVE, OFF, NormalizeConfig, normalize
from agent_hub.eval.datasets import GoldenRecord
from agent_hub.eval.knowhub_client import KnowHubEvalClient
from agent_hub.eval.retrieval_metrics import METRIC_NAMES, score_all
from agent_hub.eval.seed import NORM_SUFFIX, base_doc_id, is_normalized

__all__ = ["Variant", "RunResult", "run_variant", "standard_variants"]


@dataclass(frozen=True)
class Variant:
    """One retrieval configuration."""

    name: str
    lang: str
    query_config: NormalizeConfig = OFF
    corpus: str = "raw"          # "raw" | "normalized"
    hybrid: bool = True

    @property
    def description(self) -> str:
        query = "normalised" if self.query_config != OFF else "as-typed"
        return (
            f"{self.lang} | query {query} | corpus {self.corpus} | "
            f"{'hybrid' if self.hybrid else 'vector-only'}"
        )


@dataclass
class RunResult:
    variant: Variant
    k: int
    per_record: dict[str, dict[str, float]] = field(default_factory=dict)
    ranked: dict[str, list[str]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def record_ids(self) -> list[str]:
        """Stable ordering — paired statistics depend on it."""
        return sorted(self.per_record)

    def metric_scores(self, metric: str) -> list[float]:
        return [self.per_record[r][metric] for r in self.record_ids]

    def mean(self, metric: str) -> float:
        scores = self.metric_scores(metric)
        return sum(scores) / len(scores) if scores else 0.0

    def means(self) -> dict[str, float]:
        return {m: self.mean(m) for m in METRIC_NAMES}

    def by_stratum(
        self, records: Sequence[GoldenRecord], attr: str, metric: str
    ) -> dict[str, float]:
        """Mean of `metric` grouped by `attr` ('trap' or 'difficulty')."""
        buckets: dict[str, list[float]] = {}
        index = {r.id: r for r in records}
        for record_id in self.record_ids:
            record = index.get(record_id)
            if record is None:
                continue
            buckets.setdefault(getattr(record, attr), []).append(self.per_record[record_id][metric])
        return {k: sum(v) / len(v) for k, v in sorted(buckets.items()) if v}


def _accept(title: str, want_normalized: bool) -> bool:
    """Keep only hits from the corpus copy this variant is measuring.

    Both copies live in the same index, so without this filter a 'raw' run would
    score against normalised documents it was never meant to see — and the
    comparison between variants would be meaningless.
    """
    return is_normalized(title) if want_normalized else not is_normalized(title)


#: know-hub caps `limit` at 50 (SearchQuerySchema). Fetch the maximum.
MAX_FETCH = 50


def run_variant(
    client: KnowHubEvalClient,
    records: Sequence[GoldenRecord],
    variant: Variant,
    k: int = 5,
    fetch: int = MAX_FETCH,
) -> RunResult:
    """Run every record through one configuration and score the ranking.

    `fetch` is the API maximum, not a tuned value, for two reasons. The
    rank-aware metrics (MRR, NDCG) look deeper than `k`. More importantly, the
    raw and `~norm` corpus copies share one index, so a variant sees roughly
    half the hits it fetches — a narrow window would starve `ar:both-norm` of
    its own documents and depress its score for a reason that has nothing to do
    with retrieval quality.
    """
    result = RunResult(variant=variant, k=k)
    want_normalized = variant.corpus == "normalized"

    for record in records:
        query = normalize(record.query, variant.query_config)
        try:
            hits = client.search(query, limit=fetch, hybrid=variant.hybrid, min_score=0.0)
        except Exception as exc:                       # noqa: BLE001
            result.errors[record.id] = f"{type(exc).__name__}: {exc}"
            result.per_record[record.id] = dict.fromkeys(METRIC_NAMES, 0.0)
            result.ranked[record.id] = []
            continue

        ranked: list[str] = []
        for hit in hits:
            if not _accept(hit.title, want_normalized):
                continue
            doc_id = base_doc_id(hit.title)
            if doc_id not in ranked:                   # dedupe, keep best rank
                ranked.append(doc_id)

        result.ranked[record.id] = ranked
        result.per_record[record.id] = score_all(ranked, record.relevant_doc_ids, k=k)

    return result


def standard_variants(lang: str) -> list[Variant]:
    """The variants `make evals` reports for one language.

    English gets the baseline pair only — there is nothing to normalise, and
    that is precisely the contrast that makes the Arabic numbers legible.
    """
    baseline = [
        Variant(f"{lang}:baseline", lang, OFF, "raw", hybrid=True),
        Variant(f"{lang}:vector-only", lang, OFF, "raw", hybrid=False),
    ]
    if lang != "ar":
        return baseline
    return baseline + [
        # Cheap fix: normalise the query, leave the index alone.
        Variant("ar:query-norm", lang, CONSERVATIVE, "raw", hybrid=True),
        # Expensive fix: normalise both sides — requires re-embedding the corpus.
        Variant("ar:both-norm", lang, CONSERVATIVE, "normalized", hybrid=True),
        # Isolates the lexical half's contribution once both sides are normalised.
        Variant("ar:both-norm-vector-only", lang, CONSERVATIVE, "normalized", hybrid=False),
    ]


#: Pairs the report compares, as (baseline, variant, what it answers).
STANDARD_COMPARISONS: list[tuple[str, str, str]] = [
    ("ar:baseline", "ar:query-norm",
     "Query-side normalisation alone (no reindex)"),
    ("ar:query-norm", "ar:both-norm",
     "Additional gain from normalising the corpus too (requires reindex)"),
    ("ar:baseline", "ar:both-norm",
     "Total gain from end-to-end normalisation"),
    ("ar:vector-only", "ar:baseline",
     "What the 'english' tsvector contributes on Arabic"),
    ("en:vector-only", "en:baseline",
     "What the 'english' tsvector contributes on English"),
]

_ = NORM_SUFFIX  # re-exported for callers that build titles directly
