"""JevReranker: drop-in reranker API with three behavioral modes.

- ``relevance``: classic query/passage rerank — the relevance head ONLY
  (one Score question per candidate); value = normalized relevance.
- ``memory``: all four heads (relevance, utility, superseded, conflict);
  STALE/CONFLICT/UNCERTAIN labels surfaced.
- ``context``: memory heads PLUS token-budget selection with embedding-based
  near-duplicate suppression. Two-pass structure, documented:
    pass 1 (Jev): ONE batched call — 4 heads per candidate, ranked + labeled;
    pass 2 (local code): near-duplicate suppression (``dedup.py``), then a
    greedy value-per-token knapsack over the survivors. Pass 2 makes no
    remote calls.

Async-first (``arerank``/``aselect``) — the judge call itself is awaited on
the SDK's async client when the judge supports it; sync wrappers
(``rerank``/``select``) for drop-in use. Exactly one judge call per
invocation (or cache hit).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Coroutine, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal, TypeVar

from jev_reranker.client import JevClient
from jev_reranker.dedup import DEFAULT_DUP_THRESHOLD, suppress_near_duplicates
from jev_reranker.judges import AsyncJudge, Judge
from jev_reranker.models import (
    Candidate,
    ContextSelection,
    FallbackType,
    HeadName,
    Label,
    PolicyConfig,
    RankedItem,
    RerankResult,
    TelemetryCallback,
)
from jev_reranker.policy import POLICY_VERSION, QUESTION_SCHEMA_VERSION
from jev_reranker.rubric import Rubric

Mode = Literal["relevance", "memory", "context"]

_T = TypeVar("_T")


def as_candidates(items: Sequence[Candidate | str]) -> list[Candidate]:
    """Accept plain strings anywhere Candidates are expected (positional ids).

    Drop-in ergonomics: the #1 integration mistake is passing ``list[str]``
    to ``rerank``; that becomes valid input instead of a cryptic pydantic
    error. Candidate instances (and subclasses) pass through untouched.
    """
    out: list[Candidate] = []
    for i, item in enumerate(items):
        out.append(item if isinstance(item, Candidate) else Candidate(id=f"c{i}", text=item, retrieval_rank=i))
    return out


def run_coroutine_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run a coroutine from sync code, even under a running event loop.

    ``asyncio.run`` raises inside Jupyter, FastAPI handlers, or any host with
    a loop already running in this thread; there the coroutine executes on a
    private loop in a worker thread instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def candidate_tokens(c: Candidate) -> int:
    return c.token_estimate if c.token_estimate else estimate_tokens(c.text)


def select_for_budget(
    items: list[RankedItem],
    budget_tokens: int,
    *,
    dedup_threshold: float | None = None,
) -> ContextSelection:
    """Greedy value-per-token selection (pass 2 of context mode).

    Order of operations, all local code:
    1. drop label-DROP items;
    2. near-duplicate suppression (``dedup_threshold``; None = default 0.72,
       any negative value disables suppression);
    3. greedy knapsack by value-per-token under ``budget_tokens``.
    """
    eligible = [it for it in items if it.label is not Label.DROP]
    excluded: list[RankedItem] = [it for it in items if it.label is Label.DROP]
    if dedup_threshold is None:
        dedup_threshold = DEFAULT_DUP_THRESHOLD
    # Near-duplicate clustering prefers the highest-value member (e.g. the
    # current fact over its stale duplicate), then falls back to rank order.
    by_preference = sorted(eligible, key=lambda it: (-it.value, it.rank))
    if dedup_threshold >= 0:
        usable, suppressed = suppress_near_duplicates(by_preference, dedup_threshold)
    else:
        usable, suppressed = by_preference, []

    by_value_per_token = sorted(usable, key=lambda it: -(it.value / max(1, candidate_tokens(it.candidate))))
    selected: list[RankedItem] = []
    used = 0
    for it in by_value_per_token:
        cost = candidate_tokens(it.candidate)
        if used + cost <= budget_tokens:
            selected.append(it)
            used += cost
        else:
            excluded.append(it)
    selected.sort(key=lambda it: it.rank)  # restore relevance order for the prompt
    return ContextSelection(
        selected=selected,
        total_tokens=used,
        budget_tokens=budget_tokens,
        excluded=excluded,
        suppressed_near_duplicates=suppressed,
    )


class JevReranker:
    """Drop-in reranker. ``judge=None`` auto-picks LiveJevJudge iff key present."""

    def __init__(
        self,
        judge: Judge | AsyncJudge | None = None,
        model: str = "jev-latest",
        policy: PolicyConfig | None = None,
        cache_path: Path | str | None = None,
        fallback: FallbackType = "retrieval_order",
        default_top_k: int | None = None,
        on_event: TelemetryCallback | None = None,
        rubric: str | Rubric | None = None,
    ) -> None:
        self.client = JevClient(
            judge=judge,
            model=model,
            policy=policy,
            cache_path=cache_path,
            fallback=fallback,
            on_event=on_event,
            rubric=rubric,
        )
        self.default_top_k = default_top_k

    @property
    def judge_calls(self) -> int:
        return self.client.calls_made

    async def aclose(self) -> None:
        await self.client.aclose()

    async def arerank(
        self,
        query: str,
        candidates: Sequence[Candidate | str],
        *,
        mode: Mode = "relevance",
        top_k: int | None = None,
        budget_tokens: int | None = None,
        heads: tuple[HeadName, ...] | None = None,
    ) -> RerankResult:
        t0 = time.perf_counter()
        items, meta = await self.client.ajudge_once(query, as_candidates(candidates), mode=mode, heads=heads)
        total_ms = round((time.perf_counter() - t0) * 1000, 2)
        k = top_k if top_k is not None else self.default_top_k
        shown = items[:k] if k is not None else items
        trace: dict[str, object] = {"judge": type(self.client.judge).__name__}
        for note_key in ("dropped_by_limit", "dropped_by_token_budget"):
            if note_key in meta:
                trace[note_key] = meta[note_key]
        if mode == "context" and budget_tokens is not None:
            selection = select_for_budget(items, budget_tokens)
            trace["selection"] = {
                "total_tokens": selection.total_tokens,
                "budget_tokens": budget_tokens,
                "excluded": [it.candidate.id for it in selection.excluded],
                "suppressed_near_duplicates": [it.candidate.id for it in selection.suppressed_near_duplicates],
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
            latency_ms=meta.get("latency_ms", total_ms),
            cached=bool(meta.get("cached", False)),
            fallback_used=bool(meta.get("fallback_used", False)),
            trace=trace,
        )

    def rerank(
        self,
        query: str,
        candidates: Sequence[Candidate | str],
        *,
        mode: Mode = "relevance",
        top_k: int | None = None,
        budget_tokens: int | None = None,
        heads: tuple[HeadName, ...] | None = None,
    ) -> RerankResult:
        """Sync wrapper (drop-in for LangChain-style compressors).

        Safe to call from inside a running event loop (Jupyter, async hosts).
        """
        return run_coroutine_sync(
            self.arerank(query, candidates, mode=mode, top_k=top_k, budget_tokens=budget_tokens, heads=heads)
        )

    async def aselect(
        self,
        query: str,
        candidates: Sequence[Candidate | str],
        *,
        budget_tokens: int,
        dedup_threshold: float | None = None,
    ) -> ContextSelection:
        t0 = time.perf_counter()
        items, meta = await self.client.ajudge_once(query, as_candidates(candidates), mode="context")
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        selection = select_for_budget(items, budget_tokens, dedup_threshold=dedup_threshold)
        selection.usage = dict(meta.get("usage", {}))
        selection.latency_ms = latency_ms
        return selection

    def select(
        self,
        query: str,
        candidates: Sequence[Candidate | str],
        *,
        budget_tokens: int,
        dedup_threshold: float | None = None,
    ) -> ContextSelection:
        return run_coroutine_sync(
            self.aselect(query, candidates, budget_tokens=budget_tokens, dedup_threshold=dedup_threshold)
        )

    # Convenience: rerank plain strings (also accepted directly by rerank/select).
    def rerank_texts(
        self,
        query: str,
        texts: list[str],
        mode: Mode = "relevance",
        top_k: int | None = None,
        budget_tokens: int | None = None,
        heads: tuple[HeadName, ...] | None = None,
    ) -> RerankResult:
        return self.rerank(query, texts, mode=mode, top_k=top_k, budget_tokens=budget_tokens, heads=heads)
