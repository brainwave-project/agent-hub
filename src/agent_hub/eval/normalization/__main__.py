"""python -m agent_hub.eval.normalization --help"""

import argparse
import json
import os
import sys
from pathlib import Path

from agent_hub.eval import configure_stdout
from agent_hub.eval.datasets import EVAL_DIR

from .benchmark import BenchmarkStore, RunConfig, options
from .prepare import CORPUS_REVISION, TOPICS_REVISION, prepare_miracl


def parser():
    result = argparse.ArgumentParser(
        description="Controlled Arabic normalization retrieval benchmark"
    )
    result.add_argument(
        "--root",
        type=Path,
        default=Path(os.getenv("EVAL_LAB_DIR", EVAL_DIR / "lab_store")) / "normalization",
        help="Shared benchmark store used by the Eval Lab API",
    )
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("options", help="Print exact transforms, tokenizer and run defaults")
    commands.add_parser("list", help="List imported datasets and completed run summaries")
    importer = commands.add_parser("import", help="Validate and import local UTF-8 files")
    importer.add_argument("--name", required=True)
    importer.add_argument("--corpus", type=Path, required=True)
    importer.add_argument("--queries", type=Path, required=True)
    importer.add_argument("--qrels", type=Path, required=True)
    importer.add_argument("--provenance", type=Path)
    prepare = commands.add_parser("prepare-miracl", help="Stream official pinned Arabic dev files")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    prepare.add_argument("--split", choices=["train", "dev"], default="dev")
    prepare.add_argument(
        "--query-limit",
        type=int,
        help="Queries selected by seeded hash; pilot default 100; full default all",
    )
    prepare.add_argument("--distractors", type=int, default=1000)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--dataset-revision", default=TOPICS_REVISION)
    prepare.add_argument("--corpus-revision", default=CORPUS_REVISION)
    run = commands.add_parser("run", help="Run paired raw-versus-normalized conditions")
    run.add_argument("--dataset", required=True)
    run.add_argument("--retrievers", nargs="+", choices=["bm25", "dense"], default=["bm25"])
    run.add_argument("--presets", nargs="+", default=RunConfig().presets)
    run.add_argument("--normalization-side", choices=["both", "query"], default="both")
    run.add_argument("--bootstrap-samples", type=int, default=1000)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--dense-model")
    run.add_argument("--dense-revision")
    run.add_argument("--dense-batch-size", type=int, default=32)
    run.add_argument("--dense-max-seq-length", type=int, default=256)
    run.add_argument("--query-prefix", default="")
    run.add_argument("--document-prefix", default="")
    export = commands.add_parser("export", help="Export TREC rank files for a completed run")
    export.add_argument("--run", required=True)
    export.add_argument("--output", type=Path, required=True)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)

    def progress(event):
        print(json.dumps(event, ensure_ascii=False), file=sys.stderr, flush=True)

    try:
        if args.command == "options":
            value = options()
        elif args.command == "prepare-miracl":
            value = prepare_miracl(
                args.output,
                args.mode,
                args.query_limit,
                args.distractors,
                args.seed,
                args.dataset_revision,
                args.corpus_revision,
                progress,
                split=args.split,
            )
        else:
            store = BenchmarkStore(args.root)
            if args.command == "import":
                if (args.corpus.parent / "INCOMPLETE").exists():
                    raise ValueError("MIRACL preparation is incomplete; do not import partial data")
                provenance = (
                    json.loads(args.provenance.read_text(encoding="utf-8"))
                    if args.provenance
                    else None
                )
                value = store.import_files(
                    args.name, args.corpus, args.queries, args.qrels, provenance
                )
            elif args.command == "list":
                value = {"datasets": store.list_datasets(), "runs": store.list_runs()}
            elif args.command == "export":
                value = {"output": str(store.export_trec(args.run, args.output))}
            else:
                config = RunConfig(
                    **{
                        key: getattr(args, key)
                        for key in (
                            "retrievers",
                            "presets",
                            "normalization_side",
                            "bootstrap_samples",
                            "seed",
                            "dense_model",
                            "dense_revision",
                            "dense_batch_size",
                            "dense_max_seq_length",
                            "query_prefix",
                            "document_prefix",
                        )
                    }
                )
                report = store.run(args.dataset, config, progress)
                value = {
                    "run_id": report["run_id"],
                    "dataset_id": report["dataset_id"],
                    "report_path": str(store.root / "runs" / f"{report['run_id']}.json"),
                    "conditions": [
                        {
                            "retriever": c["retriever"],
                            "preset": c["preset"],
                            "metrics": c["metrics"],
                        }
                        for c in report["conditions"]
                    ],
                }
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        print(f"Normalization experiment failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    configure_stdout()
    raise SystemExit(main())
