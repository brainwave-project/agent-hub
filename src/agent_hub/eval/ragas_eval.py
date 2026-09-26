"""Generation-quality eval — RAGAS over the live CRAG graph, judged locally.

Replaces the earlier scaffold, which built an empty dataset (`answer` and
`contexts` were never populated) and called `evaluate()` with no `llm=`, so it
would have reached for RAGAS's OpenAI default. Both are fixed here: the graph is
actually invoked, and the judge is pinned to local Ollama so no document text
leaves the network.

How the graph is driven: `CragService.stream()` is SSE-only, so reconstructing an
answer from it means parsing token events. The compiled LangGraph underneath is
a plain `CompiledStateGraph`, so `.ainvoke()` returns the entire final
`CragState` — answer, chunks, citations and the abstain flag — in one call.
That is the seam this module uses, the same one `tests/test_crag_graph.py` uses.

A caution that belongs next to the numbers: an 8B local judge is not a reliable
oracle. It is positionally biased, rewards verbosity, and drifts between model
versions. Treat these scores as a regression signal on a fixed judge, not as
absolute quality, and validate the judge against human labels before quoting it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
from collections.abc import Sequence
from dataclasses import dataclass

from agent_hub.adapters.governance_repo import defaults_from_settings
from agent_hub.adapters.know_hub_grpc import KnowHubGrpcClient
from agent_hub.adapters.ollama import OllamaReasoner
from agent_hub.adapters.prompts import PromptRegistry
from agent_hub.application.crag.graph import build_crag_graph
from agent_hub.application.crag.nodes import CragNodes
from agent_hub.config import Settings, get_settings
from agent_hub.eval import configure_stdout
from agent_hub.eval.datasets import GoldenRecord, load_golden
from agent_hub.eval.knowhub_client import EVAL_USER_ID

__all__ = ["CragOutcome", "collect_outcomes", "score_with_ragas", "run"]


class NullRecorder:
    """StepRecorder that persists nothing — keeps Postgres out of the eval path."""

    async def record(self, *_args: object, **_kwargs: object) -> None:
        return None


@dataclass
class CragOutcome:
    """What one question produced, flattened for scoring."""

    record_id: str
    question: str
    answer: str
    contexts: list[str]
    context_doc_ids: list[str]
    ground_truth: str
    abstained: bool
    error: str = ""


def _build_graph(settings: Settings, know_hub_host: str):
    """Assemble the same graph `CragService._build` assembles, minus durability.

    `checkpointer=None` is explicitly supported by `build_crag_graph`, and a
    NullRecorder replaces the steps repository — so no Postgres is required to
    evaluate, only know-hub and Ollama.
    """
    governance = defaults_from_settings(settings)
    reasoner = OllamaReasoner(
        base_url=settings.ollama_base_url,
        model=settings.agent_reasoning_model,
        prompts=PromptRegistry(settings),
        settings=settings,
        governance=None,          # shipped defaults; no DB read on the eval path
    )
    nodes = CragNodes(
        retriever=KnowHubGrpcClient(host=know_hub_host, port=settings.know_hub_grpc_port),
        reasoner=reasoner,
        recorder=NullRecorder(),
        top_k=governance.top_k,
        min_score=governance.min_score,
        reranker=None,
        cross_encoder_enabled=False,
        rerank_candidate_k=governance.rerank_candidate_k,
        rerank_top_n=governance.rerank_top_n,
    )
    graph = build_crag_graph(
        nodes,
        max_loops=governance.max_loops,
        max_regen=governance.max_regen,
        query_rewrite_enabled=governance.query_rewrite_enabled,
        grounding_enabled=governance.grounding_enabled,
        checkpointer=None,
    )
    return graph, governance


async def collect_outcomes(
    records: Sequence[GoldenRecord],
    settings: Settings | None = None,
    know_hub_host: str = "know-hub",
    verbose: bool = True,
) -> tuple[list[CragOutcome], dict]:
    """Run every question through CRAG and capture answer + retrieved contexts."""
    settings = settings or get_settings()
    graph, governance = _build_graph(settings, know_hub_host)

    outcomes: list[CragOutcome] = []
    for index, record in enumerate(records, start=1):
        if verbose:
            print(f"    [{index}/{len(records)}] {record.id}", flush=True)
        try:
            state = await graph.ainvoke(
                {
                    # is_admin=False on purpose: it must match the retrieval
                    # suite's EDITOR scoping, or generation would be judged over
                    # a different (larger) corpus than the one retrieval was
                    # scored on, and the two halves of the report would not be
                    # describing the same system.
                    "question": record.query,
                    "actor_user_id": EVAL_USER_ID,
                    "is_admin": False,
                }
            )
            chunks = state.get("chunks", []) or []
            outcomes.append(
                CragOutcome(
                    record_id=record.id,
                    question=record.query,
                    answer=state.get("answer", "") or "",
                    contexts=[c.text for c in chunks],
                    context_doc_ids=[c.document_id for c in chunks],
                    ground_truth=record.ground_truth,
                    abstained=bool(state.get("abstained", False)),
                )
            )
        except Exception as exc:                       # noqa: BLE001
            outcomes.append(
                CragOutcome(
                    record_id=record.id,
                    question=record.query,
                    answer="",
                    contexts=[],
                    context_doc_ids=[],
                    ground_truth=record.ground_truth,
                    abstained=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return outcomes, governance.model_dump()


def _local_judge(settings: Settings):
    """RAGAS judge + embeddings pinned to local Ollama (air-gapped)."""
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    llm = LangchainLLMWrapper(
        ChatOllama(
            model=settings.agent_reasoning_model,
            base_url=settings.ollama_base_url,
            temperature=0.0,       # a judge must be reproducible
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model="nomic-embed-text", base_url=settings.ollama_base_url)
    )
    return llm, embeddings


def score_with_ragas(
    outcomes: Sequence[CragOutcome],
    settings: Settings | None = None,
) -> dict[str, float]:
    """Score answers for faithfulness and answer relevancy. Returns {} if unavailable.

    Abstentions and errors are excluded. An abstention is the *correct* behaviour
    for an unanswerable question, but faithfulness has no claims to verify
    against, so including it would drag the mean down for doing the right thing.
    """
    settings = settings or get_settings()

    scorable = [o for o in outcomes if o.answer.strip() and not o.abstained and not o.error]
    if not scorable:
        return {"_scored": 0.0, "_abstained": float(sum(o.abstained for o in outcomes))}

    try:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError as exc:
        # Report the ACTUAL import error, never a guess. "ragas not installed"
        # was wrong and cost real debugging time: ragas was installed, but it
        # imports langchain_community.chat_models.vertexai at module load, which
        # langchain-community >= 0.4 removed. A missing package and a broken
        # dependency tree need completely different fixes, so say which it is.
        print(f"    generation metrics unavailable: {type(exc).__name__}: {exc}", flush=True)
        if "langchain_community" in str(exc):
            print(
                "    ragas requires langchain-community < 0.4, which conflicts with "
                "agent-hub's langchain-core 1.x stack — they cannot share a venv. "
                "See eval/README.md.",
                flush=True,
            )
        else:
            print("    install with: uv pip install -e '.[eval]'", flush=True)
        return {}

    llm, embeddings = _local_judge(settings)

    rows = [
        {
            "user_input": o.question,
            "response": o.answer,
            "retrieved_contexts": o.contexts or [""],
            "reference": o.ground_truth,
        }
        for o in scorable
    ]

    try:
        # RAGAS >= 0.2
        from ragas.dataset_schema import EvaluationDataset

        dataset = EvaluationDataset.from_list(rows)
    except Exception:                                  # noqa: BLE001
        # RAGAS 0.1 column names.
        from datasets import Dataset

        dataset = Dataset.from_dict(
            {
                "question": [r["user_input"] for r in rows],
                "answer": [r["response"] for r in rows],
                "contexts": [r["retrieved_contexts"] for r in rows],
                "ground_truth": [r["reference"] for r in rows],
            }
        )

    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
        )
    except Exception as exc:                           # noqa: BLE001
        print(f"    ragas evaluate failed: {type(exc).__name__}: {exc}", flush=True)
        return {}

    scores: dict[str, float] = {}
    for key, value in dict(result).items():
        try:
            scores[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    scores["_scored"] = float(len(scorable))
    scores["_abstained"] = float(sum(o.abstained for o in outcomes))
    return scores


def dump_outcomes(path: pathlib.Path, outcomes: Sequence[CragOutcome], lang: str) -> None:
    """Write CRAG results to JSON for the isolated RAGAS scorer to pick up.

    This file is the process boundary. RAGAS cannot be imported in this venv,
    so stage 2 runs elsewhere and communicates only through this JSON — it must
    stay plain data, with no agent_hub types leaking into it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lang": lang,
        "outcomes": [
            {
                "record_id": o.record_id,
                "question": o.question,
                "answer": o.answer,
                "contexts": o.contexts,
                "ground_truth": o.ground_truth,
                "abstained": o.abstained,
                "error": o.error,
            }
            for o in outcomes
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def run(
    lang: str,
    limit: int | None = None,
    know_hub_host: str = "know-hub",
    settings: Settings | None = None,
    scorer: str = "judge",
    dump_path: pathlib.Path | None = None,
) -> tuple[dict[str, float], list[CragOutcome]]:
    """Drive CRAG over a golden set and score the answers.

    `scorer="judge"` (default) uses the in-repo rubric judge, which shares this
    venv. `scorer="ragas"` only works where RAGAS is importable — in this stack
    that means the isolated ragas-runner container, not here.
    """
    settings = settings or get_settings()
    records = load_golden(lang)
    if limit:
        records = records[:limit]
    outcomes, _config = await collect_outcomes(
        records, settings, know_hub_host=know_hub_host
    )

    if dump_path is not None:
        dump_outcomes(dump_path, outcomes, lang)

    if scorer == "ragas":
        return score_with_ragas(outcomes, settings), outcomes

    from agent_hub.eval.judge import score_answers

    scores, _judged = await score_answers(outcomes, settings)
    return scores, outcomes


def main() -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description="Generation eval over live CRAG")
    parser.add_argument("--lang", default="en", choices=["en", "ar"])
    parser.add_argument("--limit", type=int, default=None,
                        help="cap questions (a local judge is slow)")
    parser.add_argument("--know-hub-host", default="know-hub")
    parser.add_argument("--scorer", default="judge", choices=["judge", "ragas"],
                        help="'ragas' requires the isolated ragas-runner venv")
    parser.add_argument("--dump-outcomes", metavar="PATH", default=None,
                        help="write CRAG outcomes to JSON for the isolated RAGAS stage "
                             "(use eval/outcomes/, NOT eval/reports/ — the gate globs reports/)")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()

    scores, outcomes = asyncio.run(
        run(
            args.lang,
            args.limit,
            know_hub_host=args.know_hub_host,
            scorer=args.scorer,
            dump_path=pathlib.Path(args.dump_outcomes) if args.dump_outcomes else None,
        )
    )

    if args.json:
        print(json.dumps({"scores": scores, "n": len(outcomes)}, ensure_ascii=False, indent=2))
    else:
        print(f"\n  {args.lang}: {len(outcomes)} questions")
        for key, value in sorted(scores.items()):
            print(f"    {key:<24} {value:.3f}")
        failed = [o for o in outcomes if o.error]
        if failed:
            print(f"    {len(failed)} question(s) errored:")
            for o in failed[:5]:
                print(f"      {o.record_id}: {o.error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
