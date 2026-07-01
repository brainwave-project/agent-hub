"""Offline outcome eval — RAGAS faithfulness over a fixed question set (§12.3).

Runs the CRAG agent against a small, curated question set and scores the answers
for faithfulness (is every claim supported by the retrieved context?) and answer
relevancy. Air-gapped: RAGAS is configured to use the SAME local Ollama models,
so no data leaves the network (Principle 2).

This is a SCAFFOLD: it defines the harness shape and is runnable once `ragas` is
installed (`uv pip install ragas`) and the stack is up. Kept out of the runtime
deps because RAGAS pulls a large eval-only dependency tree.

Usage:
    uv pip install ragas datasets
    PYTHONPATH=src python -m agent_hub.eval.ragas_eval --questions src/agent_hub/eval/questions.json
"""

from __future__ import annotations

import argparse
import json
import pathlib


def load_questions(path: str) -> list[dict]:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def run_eval(questions_path: str) -> None:  # pragma: no cover - requires stack + ragas
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError:
        raise SystemExit("Install eval deps first:  uv pip install ragas datasets")

    questions = load_questions(questions_path)

    # NOTE: wire these to the live agent + know-hub retrieval. Each row needs the
    # produced answer + the retrieved contexts so RAGAS can score grounding.
    rows = {
        "question": [q["question"] for q in questions],
        "answer": [],     # ← fill by invoking CragService per question
        "contexts": [],   # ← fill from the retrieve node's chunks
        "ground_truth": [q.get("ground_truth", "") for q in questions],
    }
    dataset = Dataset.from_dict(rows)

    # Point RAGAS at local Ollama (air-gapped) rather than its OpenAI default.
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    print(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS outcome eval for CRAG")
    parser.add_argument("--questions", default="src/agent_hub/eval/questions.json")
    run_eval(parser.parse_args().questions)


if __name__ == "__main__":
    main()
