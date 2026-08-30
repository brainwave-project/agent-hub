"""Stage 2 of the RAGAS eval — scores CRAG outcomes in an ISOLATED environment.

    python scripts/ragas_score.py --in outcomes.json --out scores.json

Why this file exists, and why it lives in scripts/ rather than in the
agent_hub package:

RAGAS imports `langchain_community.chat_models.vertexai` at module load. That
module was removed in langchain-community 0.4, and agent-hub runs the
langchain-core 1.x stack (langchain-ollama 1.1, langchain-community 0.4.x).
Pinning langchain-community below 0.4 to satisfy RAGAS would drag langchain-core
below 0.4 and break agent-hub itself. The two cannot coexist in one virtualenv.

So RAGAS runs in its own venv, in its own container (`ragas-runner`), and this
script is the only thing that executes there.

**It must not import `agent_hub`.** Doing so would pull agent-hub's langchain
pins into the isolated environment and reintroduce the exact conflict this
separation exists to avoid. Stage 1 (`agent_hub.eval.ragas_eval
--dump-outcomes`) writes plain JSON; this script reads it. That JSON file is the
entire interface between the two environments — stdlib plus RAGAS, nothing else.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys


def load_outcomes(path: pathlib.Path) -> tuple[str, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("lang", "?"), payload.get("outcomes", [])


def build_judge(base_url: str, model: str, embed_model: str):
    """RAGAS judge + embeddings pinned to local Ollama — nothing leaves the network."""
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    llm = LangchainLLMWrapper(
        # temperature=0.0: a judge that is not reproducible cannot detect a
        # regression, because its own noise is indistinguishable from the change.
        ChatOllama(model=model, base_url=base_url, temperature=0.0)
    )
    embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=embed_model, base_url=base_url)
    )
    return llm, embeddings


def score(outcomes: list[dict], base_url: str, model: str, embed_model: str) -> dict[str, float]:
    # Abstentions and errored turns are excluded, matching the built-in judge:
    # an abstention is the right answer to an unanswerable question and has no
    # claims to verify, so scoring it would punish correct behaviour.
    scorable = [
        o
        for o in outcomes
        if (o.get("answer") or "").strip() and not o.get("abstained") and not o.get("error")
    ]
    result_scores: dict[str, float] = {
        "_scored": 0.0,
        "_abstained": float(sum(1 for o in outcomes if o.get("abstained"))),
    }
    if not scorable:
        return result_scores

    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness

    llm, embeddings = build_judge(base_url, model, embed_model)

    rows = [
        {
            "user_input": o["question"],
            "response": o["answer"],
            "retrieved_contexts": o.get("contexts") or [""],
            "reference": o.get("ground_truth", ""),
        }
        for o in scorable
    ]

    try:
        from ragas.dataset_schema import EvaluationDataset

        dataset = EvaluationDataset.from_list(rows)
    except Exception:                                   # noqa: BLE001 — RAGAS 0.1 shape
        from datasets import Dataset

        dataset = Dataset.from_dict(
            {
                "question": [r["user_input"] for r in rows],
                "answer": [r["response"] for r in rows],
                "contexts": [r["retrieved_contexts"] for r in rows],
                "ground_truth": [r["reference"] for r in rows],
            }
        )

    evaluation = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )

    for key, value in dict(evaluation).items():
        try:
            result_scores[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    result_scores["_scored"] = float(len(scorable))
    return result_scores


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass

    parser = argparse.ArgumentParser(description="Score CRAG outcomes with RAGAS (isolated venv)")
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--out", dest="outfile", default=None)
    parser.add_argument("--base-url", default=os.environ.get("OLLAMA_BASE_URL",
                                                             "http://ollama:11434"))
    parser.add_argument("--model", default=os.environ.get("AGENT_REASONING_MODEL",
                                                          "llama3.1:8b-instruct-q4_K_M"))
    parser.add_argument("--embed-model", default="nomic-embed-text")
    args = parser.parse_args()

    in_path = pathlib.Path(args.infile)
    if not in_path.exists():
        print(f"no outcomes file at {in_path} — run stage 1 first", file=sys.stderr)
        return 2

    lang, outcomes = load_outcomes(in_path)
    if not outcomes:
        print(f"{in_path} contains no outcomes", file=sys.stderr)
        return 2

    print(f"  scoring {len(outcomes)} {lang} outcome(s) with RAGAS...", flush=True)
    try:
        scores = score(outcomes, args.base_url, args.model, args.embed_model)
    except Exception as exc:                            # noqa: BLE001
        print(f"  RAGAS failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    for key, value in sorted(scores.items()):
        print(f"    {key:<20} {value:.3f}")

    if args.outfile:
        out = pathlib.Path(args.outfile)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"lang": lang, "scores": scores}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  scores -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
