"""JevReranker: drop-in reranker API with three modes.

- ``relevance``: classic query/passage rerank (policy value only).
- ``memory``: agent-memory triage — STALE/CONFLICT/UNCERTAIN labels surfaced,
  superseded + conflict heads fully active.
- ``context``: token-budget selection — greedy value-per-token knapsack over
  non-DROP items so the host agent always fits its window.

Async-first (``arerank``/``aselect``); sync wrappers (``rerank``/``select``)
for drop-in use. Exactly one judge call per invocation (or cache hit).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from jev_reranker.client import JevClient
from jev_reranker.judges import Judge
from jev_reranker.models import (
    Candidate,
    ContextSelection,
    FallbackType,
    Label,
    PolicyConfig,
    RankedItem,
    RerankResult,
)
from jev_reranker.policy import POLICY_VERSION, QUESTION_SCHEMA_VERSION

Mode = Literal["relevance", "memory", "context"]


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def candidate_tokens(c: Candidate) -> int:
    return c.token_estimate if c.token_estimate else estimate_tokens(c.text)


def select_for_budget(items: list[RankedItem], budget_tokens: int) -> ContextSelection:
    """Greedy value-per-token selection over usable items (never DROP)."""
    eligible = [it for it in items if it.label is not Label.DROP]
    eligible.sort(key=lambda it: -(it.value / max(1, candidate_tokens(it.candidate))))
    selected: list[RankedItem] = []
    excluded: list[RankedItem] = [it for it in items if it.label is Label.DROP]
    used = 0
    for it in eligible:
        cost = candidate_tokens(it.candidate)
        if used + cost <= budget_tokens:
            selected.append(it)
            used += cost
        else:
            excluded.append(it)
    selected.sort(key=lambda it: it.rank)  # restore relevance order for the prompt
    return ContextSelection(selected=selected, total_tokens=used, budget_tokens=budget_tokens, excluded=excluded)


class JevReranker:
    """Drop-in reranker. ``judge=None`` auto-picks LiveJevJudge iff key present."""

    def __init__(
        self,
        judge: Judge | None = None,
        model: str = "jev-latest",
        policy: PolicyConfig | None = None,
        cache_path: Path | str | None = None,
        fallback: FallbackType = "retrieval_order",
        default_top_k: int | None = None,
    ) -> None:
        self.client = JevClient(judge=judge, model=model, policy=policy, cache_path=cache_path, fallback=fallback)
        self.default_top_k = default_top_k

    @property
    def judge_calls(self) -> int:
        return self.client.calls_made

    async def arerank(
        self,
        query: str,
        candidates: list[Candidate],
        *,
        mode: Mode = "relevance",
        top_k: int | None = None,
        budget_tokens: int | None = None,
    ) -> RerankResult:
        loop = asyncio.get_running_loop()
        items, meta = await loop.run_in_executor(None, self.client.judge_once, query, candidates)
        k = top_k if top_k is not None else self.default_top_k
        shown = items[:k] if k is not None else items
        trace: dict[str, object] = {"judge": type(self.client.judge).__name__}
        if mode == "context" and budget_tokens is not None:
            selection = select_for_budget(items, budget_tokens)
            trace["selection"] = {
                "total_tokens": selection.total_tokens,
                "budget_tokens": budget_tokens,
                "excluded": [it.candidate.id for it in selection.excluded],
            }
        return RerankResult(
            request_id=str(meta["request_id"]),
            mode=mode,
            query=query,
            model=self.client.model,
            policy_version=self.client.policy.version or POLICY_VERSION,
            schema_version=QUESTION_SCHEMA_VERSION,
            items=shown,
            usage=dict(meta.get("usage", {})),
            cached=bool(meta.get("cached", False)),
            fallback_used=bool(meta.get("fallback_used", False)),
            trace=trace,
        )

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        *,
        mode: Mode = "relevance",
        top_k: int | None = None,
        budget_tokens: int | None = None,
    ) -> RerankResult:
        """Sync wrapper (drop-in for LangChain-style compressors)."""
        return asyncio.run(self.arerank(query, candidates, mode=mode, top_k=top_k, budget_tokens=budget_tokens))

    async def aselect(self, query: str, candidates: list[Candidate], *, budget_tokens: int) -> ContextSelection:
        items, _ = await asyncio.get_running_loop().run_in_executor(None, self.client.judge_once, query, candidates)
        return select_for_budget(items, budget_tokens)

    def select(self, query: str, candidates: list[Candidate], *, budget_tokens: int) -> ContextSelection:
        return asyncio.run(self.aselect(query, candidates, budget_tokens=budget_tokens))

    # Convenience: rerank plain strings.
    def rerank_texts(
        self,
        query: str,
        texts: list[str],
        mode: Mode = "relevance",
        top_k: int | None = None,
        budget_tokens: int | None = None,
    ) -> RerankResult:
        cands = [Candidate(id=f"c{i}", text=t, retrieval_rank=i) for i, t in enumerate(texts)]
        return self.rerank(query, cands, mode=mode, top_k=top_k, budget_tokens=budget_tokens)
