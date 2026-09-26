"""Hand-built cases that make each generation metric fail in a known way.

Every scenario fixes the four texts a metric can read — question, context,
answer, reference — so exactly one thing is wrong, and says which metric
should notice and which should not. The texts are verbatim sentences from the
Kestrel corpus (docs/experiment-1/data/corpus), so the playground and the
pipeline runs speak about the same documents.

The `expect` notes were checked against the stack's llama3.1:8b judge. Where
the judge itself gets it wrong (scenarios 2 and 11), the lesson says so: a
metric computed by an LLM contains that LLM's errors.
"""

from __future__ import annotations

__all__ = ["SCENARIOS"]

MEALS_CURRENT = (
    "When you travel for business, meals are reimbursed up to a daily limit of 60 euros. Meals "
    "with clients are reimbursed at actual cost, provided the names of the attendees and the "
    "business purpose are recorded on the claim."
)
MEALS_ARCHIVED = (
    "Business travel meals are reimbursed up to a daily limit of 45 euros. Client meals are "
    "reimbursed at actual cost with the attendees recorded."
)
PER_DIEM = (
    "For each full day spent travelling on business, you may claim a per diem of 45 euros to "
    "cover incidental costs such as local transport, tips and small purchases. The per diem does "
    "not cover meals, which are claimed under the Expense Reimbursement Policy. Half days of "
    "travel receive half the per diem."
)
LEAVE = (
    "Full-time employees receive 25 days of paid annual leave per calendar year, in addition to "
    "public holidays. Part-time employees receive a pro-rata entitlement based on their contracted "
    "hours. After five years of continuous service, the entitlement rises to 28 days."
)
HOTELS = (
    "Choose hotels within the nightly limits published on TravelDesk. The limit is 150 euros per "
    "night in most cities and 220 euros per night in high-cost cities such as London, Zurich and "
    "New York. Hotel stays above the limit need approval from your line manager before booking."
)
ANCHOR = (
    "Eligible employees work from the office on at least two anchor days per week, Tuesday and "
    "Thursday, and may work remotely on the other days."
)
DESKS = (
    "Most desks are shared. Book a desk or a meeting room in the workplace app up to four weeks in "
    "advance. The office is busiest on team anchor days, so book early."
)
SICK = (
    "If you are too ill to work, tell your line manager by phone or message before 9:30 on the "
    "first day of absence. An email to a shared inbox is not enough."
)
GIFTS = (
    "You may accept small gifts from suppliers or customers, such as branded pens or a box of "
    "chocolates, if their value is 50 euros or less. Cash, vouchers and gift cards must never be "
    "accepted, whatever the amount."
)

LEAVE_Q = "How many days of paid holiday do full-time staff get each year?"
LEAVE_REF = "25 days per calendar year plus public holidays. 28 days after five years of service."
MEAL_Q = "What is the daily meal limit when I'm travelling for work?"
MEAL_REF = "60 euros per day under the current policy (version 4)."
TRIP_Q = "What's the daily allowance on a business trip and does it include food?"
TRIP_ANSWER = "You get a per diem of 45 euros per full travel day for incidentals. It does not cover meals."
ANCHOR_Q = "Which weekdays do we have to be in the office?"

SCENARIOS: list[dict] = [
    {
        "id": "baseline-good",
        "title": "1 · Control: a good answer",
        "lesson": "Everything agrees, so every metric should be high. Run it first to see what "
        "'normal' looks like before you break things.",
        "question": LEAVE_Q,
        "contexts": [LEAVE],
        "answer": "25 days of paid annual leave per year, plus public holidays; 28 days after five years.",
        "reference": LEAVE_REF,
        "expect": {
            "faithfulness": "1.0",
            "answer_relevance": "high",
            "context_precision": "1.0",
            "context_recall": "1.0",
            "answer_correctness": "1.0",
        },
    },
    {
        "id": "phrasing",
        "title": "2 · Same facts, different words",
        "lesson": "The same facts as the control, in fuller sentences. Answer correctness can fall to "
        "about 0.5. Open its steps: the judge marks a true claim as FP because it 'only mentions part' "
        "of the reference — although the prompt says partial claims are fine. That is judge error "
        "inside a metric. The binary answer verdict stays 'correct'. Measuring this kind of error is "
        "what Judge Lab is for.",
        "question": LEAVE_Q,
        "contexts": [LEAVE],
        "answer": "Full-time staff get 25 days of paid annual leave per year, plus public holidays. "
        "After five years of service this rises to 28 days.",
        "reference": LEAVE_REF,
        "expect": {"answer_correctness": "~0.5 (judge error)", "answer_verdict": "correct"},
    },
    {
        "id": "stale-context",
        "title": "3 · Faithful but wrong: the archived policy",
        "lesson": "The retriever returned the 2021 archived policy and the answer repeats it. "
        "Faithfulness is 1.0: it only checks the answer against the context, and the context is wrong. "
        "Answer relevance is 1.0 too. Only the metrics that read the reference see the problem. Look "
        "at answer correctness: it can be about 0.5, not 0, because the wrong number is only ONE of the "
        "claims — claim-level F1 gives partial credit to a wrong answer. The binary verdict says "
        "'incorrect'.",
        "question": MEAL_Q,
        "contexts": [MEALS_ARCHIVED],
        "answer": "Meals on business travel are reimbursed up to 45 euros per day.",
        "reference": MEAL_REF,
        "expect": {
            "faithfulness": "1.0 (misleading)",
            "answer_relevance": "1.0 (misleading)",
            "context_precision": "0",
            "context_recall": "0",
            "answer_correctness": "0–0.5",
            "answer_verdict": "incorrect",
        },
    },
    {
        "id": "hallucinated-detail",
        "title": "4 · Hallucination: one invented detail",
        "lesson": "Most of the answer is right, but it adds a birthday day off that no document "
        "mentions. Faithfulness drops to about 0.67: one of three claims is unsupported. The claims "
        "list shows you exactly which one.",
        "question": LEAVE_Q,
        "contexts": [LEAVE],
        "answer": "Full-time staff get 25 days of paid annual leave plus public holidays, and one extra "
        "day off on your birthday.",
        "reference": LEAVE_REF,
        "expect": {"faithfulness": "~0.67", "answer_relevance": "high"},
    },
    {
        "id": "relevant-but-wrong",
        "title": "5 · Relevant but wrong",
        "lesson": "The answer addresses the question directly and confidently — with the wrong number. "
        "Answer relevance never reads the context or the reference, so it scores 1.0. Faithfulness (0) "
        "and correctness (0) catch it.",
        "question": "What's the hotel budget per night in Zurich?",
        "contexts": [HOTELS],
        "answer": "The hotel budget in Zurich is 150 euros per night.",
        "reference": "220 euros per night (high-cost city).",
        "expect": {
            "faithfulness": "0",
            "answer_relevance": "1.0 (misleading)",
            "answer_correctness": "0",
        },
    },
    {
        "id": "abstention",
        "title": "6 · Vacuous faithfulness: 'I don't know'",
        "lesson": "An answer with no claims cannot contain an unsupported claim, so faithfulness is 1.0 "
        "by convention (flagged vacuous). Relevance and correctness show it is useless. The context held "
        "the answer (context recall 1.0), so this is a generation failure. Never read faithfulness "
        "alone.",
        "question": "By what time must I tell my manager I'm too unwell to come in?",
        "contexts": [SICK],
        "answer": "I don't have enough information in the available documents to answer that "
        "confidently.",
        "reference": "Before 9:30 on the first day of absence, by phone or message.",
        "expect": {
            "faithfulness": "1.0 (vacuous)",
            "answer_relevance": "low",
            "answer_correctness": "0",
        },
    },
    {
        "id": "rank-matters",
        "title": "7 · Context precision: the useful chunk ranked last",
        "lesson": "The context holds the answer (recall 1.0) but in 4th place, behind three off-topic "
        "chunks. Context precision is rank-weighted: one useful chunk at rank 4 gives precision@4 = "
        "0.25. Move the leave chunk to rank 1 with the arrows and run again: 1.0 with the same four "
        "chunks.",
        "question": LEAVE_Q,
        "contexts": [HOTELS, DESKS, GIFTS, LEAVE],
        "answer": "Full-time employees receive 25 days of paid annual leave per year, plus public holidays.",
        "reference": "25 days per calendar year plus public holidays.",
        "expect": {"context_precision": "0.25", "context_recall": "1.0"},
    },
    {
        "id": "half-the-context",
        "title": "8 · Context recall: half of a multi-document answer",
        "lesson": "The full answer needs the travel policy (per diem) AND the expense policy (meal "
        "limit). Only the travel policy was retrieved, so context recall is 0.5. No generator can "
        "complete this answer. It is a retrieval failure that an end-to-end score would blame on "
        "generation.",
        "question": TRIP_Q,
        "contexts": [PER_DIEM],
        "answer": TRIP_ANSWER,
        "reference": "45 euros per full day for incidentals, excluding meals. Meals are claimed "
        "separately up to 60 euros per day.",
        "expect": {"context_recall": "0.5", "faithfulness": "1.0"},
    },
    {
        "id": "vague-reference",
        "title": "9 · A lazy reference inflates recall",
        "lesson": "Scenario 8 with a vague reference. With nothing specific to find, context recall "
        "reaches 1.0. A reference-based metric is only as good as the reference you wrote.",
        "question": TRIP_Q,
        "contexts": [PER_DIEM],
        "answer": TRIP_ANSWER,
        "reference": "There is a daily allowance for business trips.",
        "expect": {"context_recall": "1.0 (misleading)"},
    },
    {
        "id": "padded",
        "title": "10 · Verbosity: correct answer plus fluff",
        "lesson": "The first sentence is correct; the rest is generic advice. Per-claim faithfulness "
        "goes DOWN (the fluff claims are not in the context) and the relevance rubric penalises burying "
        "the answer. Claim-level metrics resist padding. Compare a single-score judge on the same kind "
        "of padding in Judge Lab → Bias probes.",
        "question": "A supplier offered me a 40 euro shopping voucher. Can I keep it?",
        "contexts": [GIFTS],
        "answer": "No — vouchers must never be accepted, whatever the amount. It is always wise to be "
        "careful with suppliers, and most companies have similar rules. Speaking to your manager is a "
        "good idea, and many employees find the compliance training helpful for situations like this "
        "one.",
        "reference": "No. Vouchers are never accepted, even under the 50 euro gift limit.",
        "expect": {
            "faithfulness": "low (~0.4)",
            "answer_relevance": "low",
            "answer_verdict": "correct",
        },
    },
    {
        "id": "distractor-context",
        "title": "11 · On-topic distractor — and a judge that guesses",
        "lesson": "The office guide mentions anchor days but not which days. Context precision should "
        "be 0. Context recall may still say 1.0: read the step's reason — the judge 'infers' Tuesday and "
        "Thursday from the words 'anchor days'. An LLM-computed context metric can hallucinate too. The "
        "label-based twin in a pipeline run (evidence spans, no LLM) cannot.",
        "question": ANCHOR_Q,
        "contexts": [DESKS],
        "answer": "The office is busiest on team anchor days, but the context does not say which "
        "weekdays they are.",
        "reference": "Tuesday and Thursday are the anchor days.",
        "expect": {
            "context_precision": "0",
            "context_recall": "should be 0 — may be 1.0 (judge error)",
            "answer_verdict": "incorrect",
        },
    },
    {
        "id": "current-and-archived",
        "title": "12 · A context that contradicts itself",
        "lesson": "The context holds the archived (45) and the current (60) policy. The answer picks "
        "45. Is that claim 'supported by the context'? Chunk 1 says it; chunk 2 contradicts it. The "
        "judge may say either. The metric's definition is ambiguous here — a judge design choice you "
        "must make on purpose and write into the rubric. Context precision is 0.5: the harmful chunk is "
        "at rank 1, the useful one at rank 2.",
        "question": MEAL_Q,
        "contexts": [MEALS_ARCHIVED, MEALS_CURRENT],
        "answer": "The daily meal limit is 45 euros.",
        "reference": MEAL_REF,
        "expect": {
            "faithfulness": "ambiguous (0 or 1)",
            "context_precision": "0.5",
            "answer_verdict": "incorrect",
        },
    },
    {
        "id": "anchor-good",
        "title": "13 · Control: the remote-work policy",
        "lesson": "The right chunk and the right answer — compare with scenario 11.",
        "question": ANCHOR_Q,
        "contexts": [ANCHOR],
        "answer": "Tuesday and Thursday are the office anchor days.",
        "reference": "Tuesday and Thursday are the anchor days.",
        "expect": {"faithfulness": "1.0", "context_precision": "1.0", "answer_verdict": "correct"},
    },
]
