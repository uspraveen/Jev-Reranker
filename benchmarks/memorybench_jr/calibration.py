"""Calibration metrics for Jev heads: Brier, ECE, reliability curves, thresholds.

Binary outcomes per candidate, derived from MemoryBench-JR gold annotations:
- ``useful``: y=1 iff the candidate is the case's gold answer; p = utility/3.
- ``relevant``: y=1 iff gold answer; p = relevance/3.
- ``superseded``: y=1 iff gold label STALE; p = Noul superseded probability.
- ``conflict``: y=1 iff gold label CONFLICT; p = Noul conflict probability.

All inputs come from a committed predictions JSONL produced by a real run —
this module computes, it never simulates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

N_BINS = 10


@dataclass
class HeadOutcome:
    head: str
    probs: list[float]
    outcomes: list[int]


def collect_outcomes(predictions_path: Path) -> list[HeadOutcome]:
    useful = HeadOutcome("useful", [], [])
    relevant = HeadOutcome("relevant", [], [])
    superseded = HeadOutcome("superseded", [], [])
    conflict = HeadOutcome("conflict", [], [])
    with predictions_path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            gold_id = rec["gold_id"]
            gold_labels: dict[str, str] = rec["gold_labels"]
            for cid, j in rec["judgments"].items():
                useful.probs.append(min(1.0, j["utility"] / 3.0))
                useful.outcomes.append(1 if cid == gold_id else 0)
                relevant.probs.append(min(1.0, j["relevance"] / 3.0))
                relevant.outcomes.append(1 if cid == gold_id else 0)
                superseded.probs.append(j["superseded"])
                superseded.outcomes.append(1 if gold_labels.get(cid) == "STALE" else 0)
                conflict.probs.append(j["conflict"])
                conflict.outcomes.append(1 if gold_labels.get(cid) == "CONFLICT" else 0)
    return [useful, relevant, superseded, conflict]


def brier_score(probs: list[float], outcomes: list[int]) -> float:
    if not probs:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probs, outcomes, strict=True)) / len(probs)


def ece(probs: list[float], outcomes: list[int], n_bins: int = N_BINS) -> float:
    if not probs:
        return 0.0
    bin_sums_p = [0.0] * n_bins
    bin_sums_y = [0.0] * n_bins
    bin_counts = [0] * n_bins
    for p, y in zip(probs, outcomes, strict=True):
        b = min(n_bins - 1, int(p * n_bins))
        bin_sums_p[b] += p
        bin_sums_y[b] += y
        bin_counts[b] += 1
    total = len(probs)
    e = 0.0
    for i in range(n_bins):
        if bin_counts[i]:
            e += (bin_counts[i] / total) * abs(bin_sums_p[i] / bin_counts[i] - bin_sums_y[i] / bin_counts[i])
    return e


def reliability_curve(probs: list[float], outcomes: list[int], n_bins: int = N_BINS) -> list[dict[str, Any]]:
    curve: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        hi_incl = 1.0 if i == n_bins - 1 else hi
        if i == n_bins - 1:
            pairs = [(p, y) for p, y in zip(probs, outcomes, strict=True) if lo <= p <= hi_incl]
        else:
            pairs = [(p, y) for p, y in zip(probs, outcomes, strict=True) if lo <= p < hi]
        ps = [p for p, _ in pairs]
        ys = [y for _, y in pairs]
        curve.append(
            {
                "bin_low": round(lo, 2),
                "bin_high": round(hi, 2),
                "count": len(ps),
                "mean_predicted": round(sum(ps) / len(ps), 4) if ps else None,
                "observed_frequency": round(sum(ys) / len(ys), 4) if ys else None,
            }
        )
    return curve


def threshold_table(
    probs: list[float],
    outcomes: list[int],
    thresholds: list[float] | None = None,
) -> list[dict[str, float | int]]:
    """Binary decision table: predict positive iff p >= t."""
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rows: list[dict[str, float | int]] = []
    n = len(probs)
    for t in thresholds:
        tp = sum(1 for p, y in zip(probs, outcomes, strict=True) if p >= t and y == 1)
        fp = sum(1 for p, y in zip(probs, outcomes, strict=True) if p >= t and y == 0)
        fn = sum(1 for p, y in zip(probs, outcomes, strict=True) if p < t and y == 1)
        tn = n - tp - fp - fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        rows.append(
            {
                "threshold": t,
                "predicted_positive": tp + fp,
                "coverage": round((tp + fp) / n, 4) if n else 0.0,
                "accuracy": round((tp + tn) / n, 4) if n else 0.0,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
            }
        )
    return rows


def compute_all(predictions_path: Path) -> dict[str, Any]:
    heads = collect_outcomes(predictions_path)
    out: dict[str, Any] = {"source": predictions_path.name, "heads": {}}
    for h in heads:
        out["heads"][h.head] = {
            "n": len(h.probs),
            "positive_rate": round(sum(h.outcomes) / len(h.outcomes), 4) if h.outcomes else 0.0,
            "brier": round(brier_score(h.probs, h.outcomes), 4),
            "ece": round(ece(h.probs, h.outcomes), 4),
            "reliability_curve": reliability_curve(h.probs, h.outcomes),
            "threshold_table": threshold_table(h.probs, h.outcomes),
        }
    return out
