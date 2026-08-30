"""Regression gate — compare a report against a baseline and fail the build.

The policy question a gate answers is not "did the number drop?" but "did it
drop by more than this eval can resolve?". A 20-question set produces a
confidence interval several points wide; blocking on any decrease turns the gate
into a coin flip that erodes trust until people start bypassing it.

So the rule here is: **block when the mean falls below the baseline by more
than `tolerance`.** Default tolerance is 0.05 — deliberately loose, because a
gate that cries wolf gets disabled, and a disabled gate protects nothing.

    python -m agent_hub.eval.gate --baseline baselines/v1.json --report reports/<t>.json
    python -m agent_hub.eval.gate --promote reports/<t>.json --baseline baselines/v1.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

from agent_hub.eval import configure_stdout
from agent_hub.eval.datasets import EVAL_DIR

BASELINES_DIR = EVAL_DIR / "baselines"

#: Variant/metric pairs the gate enforces. Everything else is reported but does
#: not block — a gate should guard the few numbers you would actually roll back
#: a release for.
GATED: tuple[tuple[str, str], ...] = (
    ("en:baseline", "recall@k"),
    ("en:baseline", "ndcg@k"),
    ("ar:baseline", "recall@k"),
    ("ar:both-norm", "recall@k"),
)


def _is_report(path: pathlib.Path) -> bool:
    """A report has a `variants` block. Other JSON in the directory does not.

    Guards against promoting the wrong file: stage-1 CRAG outcome dumps used to
    land in reports/ and were happily adopted as a baseline, after which the
    gate compared against a file containing none of the gated metrics and
    announced GATE PASSED. A gate that silently checks nothing is worse than no
    gate, so the shape is verified rather than assumed.
    """
    try:
        return isinstance(json.loads(path.read_text(encoding="utf-8")).get("variants"), dict)
    except (json.JSONDecodeError, OSError, AttributeError):
        return False


def _latest_report() -> pathlib.Path | None:
    reports = sorted(p for p in (EVAL_DIR / "reports").glob("*.json") if _is_report(p))
    return reports[-1] if reports else None


def _metric(payload: dict, variant: str, metric: str) -> float | None:
    try:
        return float(payload["variants"][variant]["metrics"][metric]["mean"])
    except (KeyError, TypeError, ValueError):
        return None


def compare(baseline: dict, report: dict, tolerance: float) -> tuple[bool, list[str]]:
    lines: list[str] = []
    failed = False
    checked = 0

    for variant, metric in GATED:
        base_value = _metric(baseline, variant, metric)
        new_value = _metric(report, variant, metric)

        if base_value is None:
            lines.append(f"  {variant:<26} {metric:<10} — not in baseline, skipped")
            continue
        checked += 1
        if new_value is None:
            lines.append(f"  {variant:<26} {metric:<10} MISSING from report -> FAIL")
            failed = True
            continue

        delta = new_value - base_value
        if delta < -tolerance:
            lines.append(
                f"  {variant:<26} {metric:<10} {base_value:.3f} -> {new_value:.3f} "
                f"({delta:+.3f}) -> FAIL"
            )
            failed = True
        else:
            lines.append(
                f"  {variant:<26} {metric:<10} {base_value:.3f} -> {new_value:.3f} "
                f"({delta:+.3f}) -> ok"
            )

    if checked == 0:
        # Every gated metric was missing from the baseline. Passing here would
        # report success for having verified nothing.
        lines.append("")
        lines.append(
            "  None of the gated metrics exist in the baseline — it is not a valid "
            "eval report. Re-promote with `make evals-promote`."
        )
        failed = True

    return failed, lines


def main() -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description="Eval regression gate")
    parser.add_argument("--baseline", default=str(BASELINES_DIR / "v1.json"))
    parser.add_argument("--report", default=None, help="defaults to the newest report")
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--promote", metavar="REPORT",
                        help="copy REPORT over the baseline and exit")
    parser.add_argument("--promote-latest", action="store_true",
                        help="promote the newest report (no shell globbing needed)")
    args = parser.parse_args()

    baseline_path = pathlib.Path(args.baseline)

    if args.promote or args.promote_latest:
        source = pathlib.Path(args.promote) if args.promote else _latest_report()
        if source is None:
            print("no reports to promote — run `make evals` first", file=sys.stderr)
            return 2
        if not source.exists():
            print(f"no such report: {source}", file=sys.stderr)
            return 2
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, baseline_path)
        print(f"promoted {source.name} -> {baseline_path}")
        return 0

    report_path = pathlib.Path(args.report) if args.report else _latest_report()
    if report_path is None or not report_path.exists():
        print("no report to gate on — run `make evals` first", file=sys.stderr)
        return 2

    if not baseline_path.exists():
        print(f"no baseline at {baseline_path}.")
        print(f"Establish one with:\n  python -m agent_hub.eval.gate --promote {report_path}")
        return 0                                       # nothing to regress against yet

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    print(f"baseline: {baseline_path.name}  ({baseline.get('timestamp', '?')})")
    print(f"report:   {report_path.name}  ({report.get('timestamp', '?')})")
    print(f"tolerance: {args.tolerance:+.3f}\n")

    if baseline.get("dataset_version") != report.get("dataset_version"):
        print(
            f"  dataset version changed "
            f"({baseline.get('dataset_version')} -> {report.get('dataset_version')}). "
            "Scores across versions are not comparable; promote a new baseline.",
            file=sys.stderr,
        )
        return 2

    failed, lines = compare(baseline, report, args.tolerance)
    print("\n".join(lines))
    print()
    if failed:
        print("GATE FAILED — a gated metric regressed beyond tolerance.")
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
