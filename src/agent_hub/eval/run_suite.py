"""Eval suite orchestrator — the entry point behind `make evals`.

    python -m agent_hub.eval.run_suite --suite all

Stages: seed -> retrieval variants -> (optional) RAGAS generation -> report.
Each stage is independently skippable so a failure in the slow generation stage
never costs you the retrieval numbers, which are the cheap and reliable half.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from agent_hub.config import get_settings
from agent_hub.eval import configure_stdout
from agent_hub.eval import report as reporting
from agent_hub.eval.datasets import LANGUAGES, load_golden
from agent_hub.eval.knowhub_client import KnowHubEvalClient
from agent_hub.eval.retrieval import RunResult, run_variant, standard_variants
from agent_hub.eval.seed import seed

SUITES = ("retrieval", "ragas", "all")


def _wait_for_knowhub(client: KnowHubEvalClient, timeout_s: float = 180.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if client.ready():
            return True
        time.sleep(3.0)
    return False


def _stage(number: int, total: int, label: str) -> None:
    print(f"\n[{number}/{total}] {label}", flush=True)


def main() -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description="Run the brainwave eval suite")
    parser.add_argument("--suite", default="all", choices=SUITES)
    parser.add_argument("--know-hub-url", default="http://know-hub:3002")
    parser.add_argument("--know-hub-host", default="know-hub", help="gRPC host for CRAG")
    parser.add_argument("--k", type=int, default=5, help="cutoff for recall@k / precision@k")
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--skip-seed", action="store_true")
    parser.add_argument("--force-seed", action="store_true")
    parser.add_argument("--holdout", action="store_true",
                        help="score ONLY held-out records (the honest number)")
    parser.add_argument("--ragas-limit", type=int, default=6,
                        help="questions per language for generation; a local judge is slow")
    parser.add_argument("--scorer", default="judge", choices=["judge", "ragas"],
                        help="'ragas' only works in the isolated ragas-runner venv")
    args = parser.parse_args()

    run_retrieval = args.suite in ("retrieval", "all")
    run_ragas = args.suite in ("ragas", "all")
    total_stages = 2 + int(run_retrieval) + int(run_ragas)
    stage = 1

    settings = get_settings()
    client = KnowHubEvalClient(base_url=args.know_hub_url)

    _stage(stage, total_stages, f"waiting for know-hub at {args.know_hub_url}")
    stage += 1
    if not _wait_for_knowhub(client):
        print("  know-hub never became ready — is the eval stack up?", file=sys.stderr)
        return 2
    print("  ready", flush=True)

    _stage(stage, total_stages, "seeding corpus")
    stage += 1
    if args.skip_seed:
        print("  skipped (--skip-seed)", flush=True)
    else:
        result = seed(client, LANGUAGES, force=args.force_seed)
        print(f"  indexed {len(result.indexed)}/{len(result.statuses)}", flush=True)
        if result.not_indexed:
            print("  WARNING — not searchable:", file=sys.stderr)
            for title, status in sorted(result.not_indexed.items()):
                print(f"    {title}: {status}", file=sys.stderr)

    records_by_lang = {
        lang: load_golden(lang, args.dataset_version, only_holdout=args.holdout)
        for lang in LANGUAGES
    }
    runs: dict[str, RunResult] = {}

    if run_retrieval:
        _stage(stage, total_stages, "retrieval variants")
        stage += 1
        for lang in LANGUAGES:
            records = records_by_lang[lang]
            for variant in standard_variants(lang):
                print(f"  {variant.name:<28} {variant.description}", flush=True)
                runs[variant.name] = run_variant(client, records, variant, k=args.k)

    generation: dict[str, dict[str, float]] = {}
    if run_ragas:
        _stage(stage, total_stages,
               f"generation ({args.scorer} judge, <={args.ragas_limit} q/lang)")
        stage += 1
        from agent_hub.eval.ragas_eval import run as ragas_run

        for lang in LANGUAGES:
            print(f"  {lang}:", flush=True)
            try:
                scores, _outcomes = asyncio.run(
                    ragas_run(
                        lang,
                        limit=args.ragas_limit,
                        know_hub_host=args.know_hub_host,
                        settings=settings,
                        scorer=args.scorer,
                    )
                )
                if scores:
                    generation[lang] = scores
            except Exception as exc:                   # noqa: BLE001
                # Never let the slow, flaky half sink the run.
                print(f"    generation eval failed: {type(exc).__name__}: {exc}",
                      file=sys.stderr)

    # --- report ---------------------------------------------------------
    print()
    if runs:
        print(reporting.render_variants(runs))
        print(reporting.render_comparisons(runs))
        for name in ("ar:baseline", "ar:both-norm"):
            if name in runs:
                print(reporting.render_strata(runs[name], records_by_lang["ar"]))
    if generation:
        print(reporting.render_generation(generation, judge=args.scorer))
    if runs:
        print(reporting.render_diagnosis(runs, generation=generation))

    payload = reporting.build_payload(
        runs=runs,
        records_by_lang=records_by_lang,
        generation=generation,
        config_snapshot={
            "k": args.k,
            "holdout_only": args.holdout,
            "reasoning_model": settings.agent_reasoning_model,
            "ollama_base_url": settings.ollama_base_url,
            "crag_top_k": settings.crag_top_k,
            "crag_min_score": settings.crag_min_score,
            "cross_encoder_enabled": settings.cross_encoder_enabled,
            "judge": args.scorer,
            "judge_model": settings.agent_reasoning_model,
        },
        dataset_version=args.dataset_version,
    )
    path = reporting.write_report(payload)
    print(f"\n  report -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
