# Evaluation Harness

Measurement for the brainwave RAG pipeline: retrieval metrics, Arabic
orthographic normalisation, RAGAS generation scoring, and a regression gate.

```bash
cd infra
make evals            # up → seed → score → report
make evals-run        # retrieval only (fast, no LLM judge)
make evals-ragas      # generation only (slow, local 8B judge)
make evals-gate       # compare newest report to baseline; non-zero on regression
```

Offline pieces need no stack at all:

```bash
cd agent-hub
uv run pytest tests/test_retrieval_metrics.py tests/test_arabic_normalize.py tests/test_eval_corpus.py
```

---

## What the stack looks like

`make evals` runs postgres, mongodb, minio, kafka, ollama, know-hub, and a
one-shot `eval-runner`. It does **not** run Kong, Keycloak, auth-hub, user-hub,
audit-hub, or the observability stack.

Two non-obvious decisions:

**Kong and Keycloak are skipped** because know-hub's `requirePermission`
middleware reads `X-User-Roles` and nothing else — it never validates a JWT,
trusting Kong to have done that upstream. The eval client supplies the headers
Kong would have injected. Convenient here; also a standing reminder that
know-hub must never be reachable from outside the compose network.

**agent-hub's service is not started.** Its Kafka consumer subscribes to
`DOCUMENT_UPLOADED` unconditionally, so running it beside know-hub's semantic
worker would index every document twice and corrupt the vector store. The
harness runs as `eval-runner` and builds the CRAG graph in-process — it needs
agent-hub's *code*, not its HTTP server.

The eval overlay also forces `SEMANTIC_WORKER_ENABLED=true` on know-hub. The dev
default is `false`, and without the override nothing is ever embedded and every
score reads as a confusing zero.

---

## Layout

| Module | Needs the stack? | Purpose |
|---|---|---|
| `retrieval_metrics.py` | no | Hit rate, Recall@k, Precision@k, MRR, NDCG@k, MAP |
| `stats.py` | no | Bootstrap CIs, paired deltas, sample-size arithmetic |
| `arabic/normalize.py` | no | Per-transform Arabic normalisation |
| `datasets.py` | no | Corpus + versioned golden-set loading |
| `seed.py` | yes | Ingest corpus, wait for `INDEXED` |
| `retrieval.py` | yes | Run golden sets, score rankings |
| `ragas_eval.py` | yes | Drive CRAG, judge answers with local Ollama |
| `report.py` | no | Console rendering + JSON report |
| `gate.py` | no | Regression gate against a promoted baseline |

---

## Domain 1 — measurement

### The metrics are hand-written on purpose

`retrieval_metrics.py` implements every metric directly rather than importing
one. Knowing that NDCG exists is not the skill; knowing that its log discount
makes a nine-place improvement at rank 20 nearly invisible is. Each function's
docstring names the specific situation in which it lies to you:

- **Hit rate** saturates. Once most questions find something it pins near 1.0
  and stops moving while ranking quality changes underneath.
- **MRR** stops at the first hit, so it is structurally blind to recall.
  Retrieving 1 of 5 relevant documents at rank 1 scores identically to
  retrieving all 5.
- **Precision@k** caps below 1.0 whenever fewer than k documents are relevant.
  With one relevant document and k=5 a perfect system reads as 0.20.
- **NDCG** discounts logarithmically, so deep-ranking improvements barely
  register.
- **Recall@k** is the one that matters most for RAG — the generator cannot cite
  what retrieval never returned — but it trivially approaches 1.0 as k
  approaches corpus size.

### Never report a bare mean

Every aggregate carries a 95% bootstrap CI; every comparison is a **paired**
delta. Paired because both variants run the same questions, which cancels
per-question difficulty — by far the largest variance source in a small set. An
unpaired test on 20 questions will almost never reach significance even for a
real improvement.

The only question to ask of an A/B result is whether the interval excludes zero.
`stats.min_sample_size()` will tell you, uncomfortably, that detecting a 5-point
change at 80% power needs a few hundred questions. This corpus has far fewer.
That is a real limit on what it can prove, and the report says so rather than
hiding it.

### Dataset discipline

- Golden sets are **versioned** (`golden_v1.jsonl`) and never edited in place.
  Changing questions under a fixed filename makes two runs silently
  incomparable.
- Records flagged `holdout: true` are **excluded by default**. Run
  `--holdout` for the honest number — the one you may quote precisely because
  you never tuned against it.
- Every record carries `difficulty` and `trap` so scores can be **stratified**.
  A flat mean over mixed difficulty hides exactly the failures worth finding.

### The gate

`make evals-gate` blocks when a gated metric falls more than `--tolerance`
(default 0.05) below the promoted baseline. Deliberately loose: a gate that
cries wolf gets disabled, and a disabled gate protects nothing. It also refuses
to compare across dataset versions.

---

## Domain 2 — Arabic retrieval

### The problem being measured

Arabic writers are inconsistent about hamza seating, taa marbuta, and
diacritics. The same word appears as several distinct byte sequences across a
corpus. Lexical matching sees unrelated tokens; embedding models see merely
similar vectors.

The Arabic corpus is seeded with these traps deliberately — golden queries spell
a term one way, the relevant document spells it another:

| Trap | Query | Document |
|---|---|---|
| `alef-hamza` | إجراءات | اجراءات |
| `taa-marbuta` | اللائحة | اللائحه |
| `tatweel` | التعميم | تعمـــيم |
| `diacritics` | المشتريات | الْمُشْتَرَيَات |
| `alef-maqsura` | على | علي |
| `digits` | 30 | ٣٠ |

Control records (`trap: none`) spell everything consistently. Without them a
rise in the Arabic score could be normalisation doing something general rather
than fixing the specific mismatch — the controls are what make the attribution
valid.

`tests/test_eval_corpus.py` asserts every trap still responds to normalisation.
If someone edits the corpus and a trap goes inert, that test fails. An eval that
cannot detect the effect it was built to detect is worse than none, because it
reports a confident number instead of an error.

### Query-side vs both-side normalisation

The suite runs both, and the gap between them is the lesson:

- **`ar:query-norm`** normalises the query only. Cheap, deployable as a config
  change, no reindex.
- **`ar:both-norm`** searches a separately-ingested normalised copy of the
  corpus.

Query-side normalisation **cannot** fix the diacritics traps. When the document
carries the marks and the query does not, stripping diacritics from the query
does nothing — there were none. Only re-embedding the corpus closes that gap.
That is the reindexing problem made concrete, with its cost attached.

### The dead lexical half

know-hub's hybrid search fuses vector and full-text results with RRF, and it is
on by default. But the tsvector is:

```sql
-- infra/postgres/init/05-semantic.sql
content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
```

Hardcoded to `'english'`, and queried with `plainto_tsquery('english', …)`.
Postgres ships no Arabic configuration, so Arabic text is run through an English
stemmer and stopword list. The `ar:vector-only` vs `ar:baseline` comparison
measures what that lexical half actually contributes — set against the same
comparison on English, where it works.

This harness only **measures** the problem. Fixing it means changing a
`GENERATED ALWAYS ... STORED` column, which is a migration plus a full
reindex — a separate piece of work that should be justified by these numbers
first.

---

## Finding: `CRAG_MIN_SCORE=0.0` makes CRAG abstain on answerable questions

The first thing this harness found, and a good illustration of why
component-level evaluation is worth the trouble.

The shipped default is `CRAG_MIN_SCORE=0.0`
(`infra/env/services/agent-hub.dev.env`). At that setting the retriever returns
`top_k` chunks no matter how weak they are, so a typical question reaches the
grader as one good chunk plus five off-topic ones. The grade prompt is
deliberately strict:

> "Be conservative: if the context is off-topic or only tangentially related,
> mark it NOT relevant."

It duly rejects the whole context. The corrective loop rewrites the query,
retrieves, is rejected again, exhausts `CRAG_MAX_LOOPS`, and abstains.

Swept over the first five English golden questions:

| `CRAG_MIN_SCORE` | chunks per question | abstained | failure mode |
|---|---|---|---|
| `0.0` (shipped) | 6, 6, 6, 6, 6 | **5/5** | grader rejects noisy context |
| `0.5` | 1, 6, 6, … | 2/3 | same, partially |
| **`0.65`** | 1, 2, 1, 1, 1 | **0/5** | — |
| `0.75` | 1, 1, 1, 1, **0** | 1/5 | retrieval starved |

A U-shaped curve with a *different* failure at each end. Too permissive and the
grader rejects a context that contains the answer; too strict and nothing is
retrieved to grade.

The clearest single case: for "How many annual leave days do full-time employees
get?", the top retrieved chunk literally reads *"A full-time employee accrues
twenty-one (21) working days"* — and CRAG abstained, because five unrelated
chunks came with it.

Nothing here is broken in isolation. Retrieval ranks the right document first.
The grader returns `relevant=True` reliably when handed that chunk alone. The
failure lives only in the *interaction* between a permissive retrieval threshold
and a grader that judges the whole context blob rather than each chunk — which
is precisely the class of failure an end-to-end score reports as "bad answer"
while telling you nothing about where to look.

The eval overlay therefore sets `CRAG_MIN_SCORE=0.65`, chosen from that sweep
rather than guessed, so the generation stage measures answer quality instead of
measuring this one misconfiguration repeatedly. The deviation is commented in
`infra/docker/compose.evals.yml`.

**This is reported, not fixed.** Changing the shipped default is a governance
decision (`PUT /api/agent/governance`, or the Retrieval Lab UI) that should be
justified by numbers across a larger question set than this one. What the
harness does guarantee is that the failure can no longer be silent: the report
prints the abstention rate next to every generation score, and the diagnosis
names `CRAG_MIN_SCORE` whenever abstentions exceed 30%.

---

## Generation scoring — two judges

`ragas_eval.py` invokes the compiled CRAG graph with `.ainvoke()` and reads
`answer` and `chunks` straight off the final state. It does not parse
`CragService.stream()`'s SSE, and it does not need Postgres — `checkpointer=None`
plus a `NullRecorder` keep the durability layer out of the eval path.

Scoring is available two ways, and they run in **different virtualenvs**:

```bash
make evals-ragas                    # built-in rubric judge  (default, fast)
make evals-ragas-real               # RAGAS, isolated venv
make evals-ragas-real lang=ar n=4
```

### Why two venvs

RAGAS imports `langchain_community.chat_models.vertexai` at module load. That
module was removed in langchain-community 0.4, and agent-hub runs the
langchain-core 1.x stack (langchain-ollama 1.1, langchain-community 0.4.x).
Pinning langchain-community below 0.4 to satisfy RAGAS would drag langchain-core
below 0.4 and break agent-hub. **They cannot share a virtualenv.**

So RAGAS runs in `ragas-runner`, its own container with its own venv
(verified: ragas 0.2.15 + langchain-community 0.3.31 + langchain-core 0.3.86),
executing exactly one file — `scripts/ragas_score.py`, which imports no
`agent_hub` code. Stage 1 dumps plain JSON; stage 2 reads it. That JSON file is
the entire interface.

### The built-in judge (`judge.py`)

Faithfulness and answer relevancy implemented as readable rubrics against the
same local Ollama. The rubrics are editable text at the top of the module, the
bias mitigations are commented, and the arithmetic is visible — so "why is
faithfulness 0.83?" has an answer you can point at.

Bias handling, briefly: faithfulness scores **per claim** (so padding lowers the
score rather than raising it), each claim is verified in an independent call (no
list position to exploit), `temperature=0.0` with a sorted claim set (same input
→ same calls → same score), and the rubrics state the strict reading explicitly.

One deliberate difference from RAGAS: relevancy is a direct rubric rather than
RAGAS's reverse-question-generation-plus-embedding-similarity. That method's
accuracy depends on the embedding model's handling of the language — which for
Arabic is the exact variable under test elsewhere in this harness, so keeping
the two measurements independent matters more than matching RAGAS's method.

### Neither is ground truth

Both are an 8B model with an opinion. Run both and compare; when they disagree,
hand-label 30–50 answers and see which one tracks you. **That comparison is the
exercise** — an unvalidated judge is an opinion with a decimal point.

Abstentions are excluded from both. Abstaining is correct on an unanswerable
question and has no claims to verify, so scoring it would penalise the system
for behaving well. The report prints the abstention count next to every score
because a high abstention rate makes the scores unrepresentative.

---

## Extending it

**Add questions.** Append to `datasets/<lang>/golden_v1.jsonl`, or start
`golden_v2.jsonl` if you are changing existing ones. Add the matching document
to `corpus/<lang>/`. `test_eval_corpus.py` enforces that every
`relevant_doc_ids` entry resolves.

**Add a metric.** Write it in `retrieval_metrics.py` with a docstring naming its
failure mode, add it to `METRIC_NAMES`, and add a hand-computed fixture to
`tests/test_retrieval_metrics.py`.

**Add a retrieval variant.** Add a `Variant` in `retrieval.standard_variants()`
and a pair in `STANDARD_COMPARISONS`.

**Change a normalisation transform.** Add the field to `NormalizeConfig`, the
logic to `normalize()`, and an entry to `TRANSFORMS` describing what it costs —
a test enforces that every field is documented.
