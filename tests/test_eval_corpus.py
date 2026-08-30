"""Corpus + golden-set integrity. No stack, no network.

The load-bearing test here is `test_every_trap_responds_to_normalisation`.
The Arabic corpus exists to make a specific effect measurable — orthographic
variants suppress retrieval, and normalisation recovers it. If a trap stops
responding, the corpus can no longer detect the thing the harness was built to
detect, and every Arabic number it produces becomes uninterpretable. That is a
silent failure, so it gets an assertion.
"""

import re
import uuid

import pytest

from agent_hub.eval.arabic.normalize import AGGRESSIVE, OFF, normalize
from agent_hub.eval.datasets import (
    LANGUAGES,
    corpus_doc_ids,
    load_corpus,
    load_golden,
)
from agent_hub.eval.knowhub_client import EVAL_USER_ID

# Split on anything that is neither a word char nor in the Arabic block, so
# trailing '؟' does not weld itself to the last token. Postgres FTS and the
# embedding tokeniser both do their own segmentation; this mirrors that.
_TOKEN_SPLIT = re.compile(r"[^\w؀-ۿ]+")


def _tokens(text: str, config) -> set[str]:
    return {t for t in _TOKEN_SPLIT.split(normalize(text, config)) if t}


def test_eval_user_id_is_a_valid_uuid():
    """know-hub casts this to ::uuid and Prisma parses it back.

    A near-miss (any character outside [0-9a-f-]) does not degrade gracefully —
    every document query fails with Prisma P2023 and the whole seed stage dies
    with a 500 that looks like a know-hub bug. Worth one assertion.
    """
    assert uuid.UUID(EVAL_USER_ID)
    assert str(uuid.UUID(EVAL_USER_ID)) == EVAL_USER_ID


@pytest.mark.parametrize("lang", LANGUAGES)
def test_corpus_is_non_empty(lang):
    corpus = load_corpus(lang)
    assert corpus, f"{lang} corpus is empty"
    for doc_id, body in corpus.items():
        assert body.strip(), f"{doc_id} is blank"


@pytest.mark.parametrize("lang", LANGUAGES)
def test_golden_ids_are_unique(lang):
    records = load_golden(lang, include_holdout=True)
    ids = [r.id for r in records]
    assert len(ids) == len(set(ids)), f"duplicate ids in {lang}"


@pytest.mark.parametrize("lang", LANGUAGES)
def test_every_relevant_doc_exists(lang):
    """A golden record pointing at a missing document scores 0 forever."""
    available = corpus_doc_ids(lang)
    for record in load_golden(lang, include_holdout=True):
        missing = set(record.relevant_doc_ids) - available
        assert not missing, f"{record.id} references missing docs: {sorted(missing)}"


@pytest.mark.parametrize("lang", LANGUAGES)
def test_every_record_has_a_query_and_relevant_set(lang):
    for record in load_golden(lang, include_holdout=True):
        assert record.query.strip(), f"{record.id} has an empty query"
        assert record.relevant_doc_ids, f"{record.id} has no relevant docs"
        assert record.lang == lang, f"{record.id} declares lang={record.lang} in {lang} file"


@pytest.mark.parametrize("lang", LANGUAGES)
def test_holdout_is_excluded_by_default(lang):
    tuning = load_golden(lang)
    everything = load_golden(lang, include_holdout=True)
    holdout = load_golden(lang, only_holdout=True)

    assert holdout, f"{lang} has no held-out records — nothing to report honestly against"
    assert len(tuning) + len(holdout) == len(everything)
    assert not any(r.holdout for r in tuning)


def test_every_arabic_trap_responds_to_normalisation():
    """Each trapped record must gain query/document token overlap once normalised.

    Measured as raw token-set intersection rather than through the retrieval
    stack on purpose: this must stay a fast, offline, deterministic check that
    fails loudly when someone edits the corpus, with no Ollama or Postgres in
    the loop.
    """
    corpus = load_corpus("ar")
    dead: list[str] = []

    for record in load_golden("ar", include_holdout=True):
        if record.trap == "none":
            continue
        document = " ".join(corpus[d] for d in record.relevant_doc_ids)
        before = len(_tokens(record.query, OFF) & _tokens(document, OFF))
        after = len(_tokens(record.query, AGGRESSIVE) & _tokens(document, AGGRESSIVE))
        if after <= before:
            dead.append(f"{record.id} ({record.trap}): overlap {before} -> {after}")

    assert not dead, (
        "These traps no longer respond to normalisation, so the Arabic eval "
        "cannot detect the effect it exists to measure:\n  " + "\n  ".join(dead)
    )


def test_arabic_control_records_need_no_normalisation():
    """Controls must already match at baseline.

    Without these, a rise in the Arabic score could be normalisation doing
    something unrelated and general rather than fixing the specific
    orthographic mismatch. The controls are what make the attribution valid.
    """
    corpus = load_corpus("ar")
    controls = [r for r in load_golden("ar", include_holdout=True) if r.trap == "none"]
    assert controls, "no control records — the Arabic delta would be unattributable"

    for record in controls:
        document = " ".join(corpus[d] for d in record.relevant_doc_ids)
        overlap = len(_tokens(record.query, OFF) & _tokens(document, OFF))
        assert overlap > 0, f"control {record.id} has no baseline overlap"


def test_trap_coverage_spans_the_documented_failure_modes():
    traps = {r.trap for r in load_golden("ar", include_holdout=True)}
    expected = {"alef-hamza", "taa-marbuta", "tatweel", "diacritics", "alef-maqsura", "digits"}
    missing = expected - traps
    assert not missing, f"Arabic set does not exercise: {sorted(missing)}"


def test_english_set_has_no_traps():
    """English is the control language — it establishes the working baseline."""
    for record in load_golden("en", include_holdout=True):
        assert record.trap == "none", f"{record.id} unexpectedly carries trap={record.trap}"


def test_at_least_one_multi_document_question():
    """Multi-doc questions are the ones that expose MRR's blindness to recall."""
    records = list(load_golden("en", include_holdout=True)) + list(
        load_golden("ar", include_holdout=True)
    )
    assert any(len(r.relevant_doc_ids) > 1 for r in records)
