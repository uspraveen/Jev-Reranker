"""Context Precision / Recall / Efficiency for MemoryBench-JR (owner section 21).

Definitions used (documented in benchmarks/memorybench_jr/README.md):
- Context Precision  = |selected ∩ relevant| / |selected|
- Context Recall     = |selected ∩ relevant| / |relevant|
- Context Efficiency = sum(value of selected) / sum(value of the ORACLE
  selection under the same token budget), where the oracle maximizes total
  policy value with a full-knapsack DP over the same items (no labels used).
Efficiency = 1.0 means the budgeted selection matched the best possible.
"""

from __future__ import annotations

from jev_reranker.models import ContextSelection, RankedItem
from jev_reranker.reranker import candidate_tokens


def context_precision(selection: ContextSelection, relevant_ids: set[str]) -> float:
    if not selection.selected:
        return 0.0
    hits = sum(1 for it in selection.selected if it.candidate.id in relevant_ids)
    return hits / len(selection.selected)


def context_recall(selection: ContextSelection, relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for it in selection.selected if it.candidate.id in relevant_ids)
    return hits / len(relevant_ids)


def oracle_value(items: list[RankedItem], budget_tokens: int) -> float:
    """Max total value under the budget: DP knapsack over token costs."""
    costs = [candidate_tokens(it.candidate) for it in items]
    values = [it.value for it in items]
    cap = max(0, int(budget_tokens))
    dp = [0.0] * (cap + 1)
    for c, v in zip(costs, values, strict=True):
        if c <= 0:
            continue
        for t in range(cap, c - 1, -1):
            cand = dp[t - c] + v
            if cand > dp[t]:
                dp[t] = cand
    best = max(dp) if dp else 0.0
    return float(best)


def context_efficiency(selection: ContextSelection, items: list[RankedItem]) -> float:
    best = oracle_value([it for it in items if it.value > 0], selection.budget_tokens)
    got = sum(it.value for it in selection.selected)
    if best <= 0:
        return 1.0 if got <= 0 else 0.0
    ratio = min(1.0, got / best)
    return round(float(ratio), 4)
