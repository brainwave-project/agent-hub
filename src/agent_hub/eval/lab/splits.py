"""Deterministic, stratified, group-aware dev/holdout splitting.

Three properties, each preventing a specific way a holdout number lies:

* **Deterministic.** Same items + same config + same seed -> same split, on any
  machine. A split you cannot regenerate is a split you cannot audit.
* **Stratified.** Each stratum (e.g. each difficulty) is split separately, so
  the holdout is not accidentally all-easy. Otherwise dev and holdout measure
  different populations and the "generalisation gap" is just a composition gap.
* **Grouped.** With `group_by="primary_doc"`, every question whose best answer is
  the same document lands on the same side. If three questions about the leave
  policy straddle the split, tuning on the dev two teaches the system about the
  holdout one: that is leakage, even though no question appears twice.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

from agent_hub.eval.lab.models import GoldenItem, SplitConfig

__all__ = ["SplitSummary", "assign_split", "summarize", "stratum_of", "group_of"]


@dataclass
class SplitSummary:
    dev: int = 0
    holdout: int = 0
    unassigned: int = 0
    groups: int = 0
    pinned: int = 0
    #: stratum -> {"dev": n, "holdout": n}
    strata: dict[str, dict[str, int]] = field(default_factory=dict)
    #: groups whose items sit on both sides of the split
    straddling_groups: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        total = self.dev + self.holdout
        return {
            "dev": self.dev,
            "holdout": self.holdout,
            "unassigned": self.unassigned,
            "holdout_fraction": (self.holdout / total) if total else 0.0,
            "groups": self.groups,
            "pinned": self.pinned,
            "strata": self.strata,
            "straddling_groups": self.straddling_groups,
        }


def stratum_of(item: GoldenItem, attrs: list[str]) -> str:
    if not attrs:
        return "all"
    return " | ".join(f"{a}={getattr(item, a)}" for a in attrs)


def group_of(item: GoldenItem, config: SplitConfig) -> str:
    return item.primary_doc if config.group_by == "primary_doc" else item.id


def assign_split(
    items: list[GoldenItem], config: SplitConfig
) -> tuple[list[GoldenItem], SplitSummary]:
    groups: dict[str, list[GoldenItem]] = {}
    for item in items:
        groups.setdefault(group_of(item, config), []).append(item)

    # A group's stratum is its majority stratum; ties break alphabetically.
    by_stratum: dict[str, list[str]] = {}
    for key, members in groups.items():
        counts = Counter(stratum_of(m, config.stratify_by) for m in members)
        stratum = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        by_stratum.setdefault(stratum, []).append(key)

    group_split: dict[str, str] = {}
    for stratum in sorted(by_stratum):
        keys = sorted(by_stratum[stratum])
        size = sum(len(groups[k]) for k in keys)
        target = round(config.holdout_fraction * size)
        holdout_count = 0

        # Pinned groups are decided first and count toward the target.
        free: list[str] = []
        for key in keys:
            pinned = [m.split for m in groups[key] if m.pinned and m.split]
            if pinned:
                group_split[key] = pinned[0]
                if pinned[0] == "holdout":
                    holdout_count += len(groups[key])
            else:
                free.append(key)

        # A str seed is hashed deterministically by random.Random, so the
        # shuffle depends only on (seed, stratum) — not on process or platform.
        random.Random(f"{config.seed}:{stratum}").shuffle(free)
        for key in free:
            n = len(groups[key])
            if abs(holdout_count + n - target) < abs(holdout_count - target):
                group_split[key] = "holdout"
                holdout_count += n
            else:
                group_split[key] = "dev"

    out: list[GoldenItem] = []
    for item in items:
        if item.pinned and item.split:
            out.append(item)
        else:
            out.append(item.model_copy(update={"split": group_split[group_of(item, config)]}))
    return out, summarize(out, config)


def summarize(items: list[GoldenItem], config: SplitConfig | None) -> SplitSummary:
    config = config or SplitConfig()
    summary = SplitSummary()
    sides: dict[str, set[str]] = {}
    for item in items:
        if item.pinned:
            summary.pinned += 1
        if item.split is None:
            summary.unassigned += 1
            continue
        if item.split == "dev":
            summary.dev += 1
        else:
            summary.holdout += 1
        stratum = stratum_of(item, config.stratify_by)
        summary.strata.setdefault(stratum, {"dev": 0, "holdout": 0})[item.split] += 1
        # Straddling is always measured by primary doc — the leakage that
        # matters — whatever grouping produced the split.
        sides.setdefault(item.primary_doc, set()).add(item.split)
    summary.groups = len({group_of(i, config) for i in items})
    summary.straddling_groups = sorted(k for k, v in sides.items() if len(v) > 1)
    return summary
