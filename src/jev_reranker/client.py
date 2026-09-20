"""JevClient: one batched Jev call per rerank + resilience layers.

Resilience (host agent never goes brittle):
- candidate limits (MAX_CANDIDATES) + per-candidate char truncation
- aggregate token-based input limit (MAX_INPUT_TOKENS) with deterministic
  truncation in retrieval order
- timeouts + retries with exponential backoff on 429/5xx (in the judge)
- cache keyed on query+candidates+model+policy+schema versions+heads
- request IDs on every result + trace
- optional telemetry hook (``on_event``) for judge_call/cache_hit/fallback
- fallback='retrieval_order' or a custom callable when Jev is unreachable
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

from jev_reranker.cache import JudgmentCache, cache_key
from jev_reranker.judges import is_async_judge
from jev_reranker.models import (
    Candidate,
    FallbackType,
    HeadJudgments,
    HeadName,
    Label,
    PolicyConfig,
    RankedItem,
    TelemetryCallback,
)
from jev_reranker.policy import QUESTION_SCHEMA_VERSION, apply_policy, build_questions, heads_for_mode

MAX_CANDIDATES = 100
MAX_CANDIDATE_CHARS = 4000
MAX_QUERY_CHARS = 2000
MAX_INPUT_TOKENS = 12000  # aggregate query+candidates budget (~4 chars/token)
MIN_TRUNCATED_CHARS = 200


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " …[truncated]"


def retrieval_order_fallback(query: str, candidates: list[Candidate]) -> list[RankedItem]:
    """Fallback: keep original retrieval order with neutral judgments."""
    ordered = sorted(
        candidates,
        key=lambda c: (c.retrieval_rank if c.retrieval_rank is not None else 10**9, c.id),
    )
    items: list[RankedItem] = []
    for rank, cand in enumerate(ordered):
        j = HeadJudgments(
            relevance=1.5,
            relevance_confidence=0.0,  # zero confidence -> UNCERTAIN by construction
            utility=1.5,
            utility_confidence=0.0,
            superseded=0.0,
            conflict=0.0,
        )
        items.append(RankedItem(candidate=cand, judgments=j, value=0.0, label=Label.UNCERTAIN, rank=rank))
    return items


class JevClient:
    """Orchestrates a single batched judgment call + policy + resilience."""

    def __init__(
        self,
        judge: Any | None = None,
        model: str = "jev-latest",
        policy: PolicyConfig | None = None,
        cache: JudgmentCache | None = None,
        cache_path: Path | str | None = None,
        fallback: FallbackType = "retrieval_order",
        max_candidates: int = MAX_CANDIDATES,
        input_token_limit: int = MAX_INPUT_TOKENS,
        on_event: TelemetryCallback | None = None,
    ) -> None:
        self.policy = policy or PolicyConfig()
        self.model = model
        self.fallback = fallback
        self.max_candidates = max_candidates
        self.input_token_limit = input_token_limit
        self.on_event = on_event
        if judge is not None:
            self.judge = judge
            if not model or model == "jev-latest":
                self.model = getattr(judge, "model_name", model)
        elif os.environ.get("TYPESAFE_API_KEY"):
            from jev_reranker.judges import LiveJevJudge

            self.judge = LiveJevJudge(model=model)
        else:
            from jev_reranker.judges import OfflineJudge

            self.judge = OfflineJudge()
        if cache is not None:
            self.cache = cache
        elif cache_path is not None:
            self.cache = JudgmentCache(cache_path)
        else:
            self.cache = JudgmentCache()
        self.calls_made = 0  # remote judge invocations (excludes cache hits)
        self.cache_hits = 0

    # -- telemetry ---------------------------------------------------------
    def _emit(self, event: dict[str, Any]) -> None:
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:
                pass  # telemetry must never break the ranking path

    # -- internals ---------------------------------------------------------
    def _prepare(self, query: str, candidates: list[Candidate]) -> tuple[list[Candidate], dict[str, Any]]:
        """Candidate cap, per-candidate truncation, aggregate token budget."""
        notes: dict[str, Any] = {}
        if len(candidates) > self.max_candidates:
            ordered = sorted(
                candidates,
                key=lambda c: (c.retrieval_rank if c.retrieval_rank is not None else 10**9, c.id),
            )
            notes["dropped_by_limit"] = [c.id for c in candidates[self.max_candidates :]]
            candidates = ordered[: self.max_candidates]
        out: list[Candidate] = []
        for c in candidates:
            out.append(c.model_copy(update={"text": _truncate(c.text, MAX_CANDIDATE_CHARS)}))
        # Aggregate token budget, deterministic: walk retrieval order, shrink
        # the first over-budget candidate, drop everything after it.
        query_tokens = len(_truncate(query, MAX_QUERY_CHARS)) // 4
        used = query_tokens
        kept: list[Candidate] = []
        dropped_by_tokens: list[str] = []
        for c in out:
            cost = max(1, len(c.text) // 4)
            if used + cost <= self.input_token_limit:
                used += cost
                kept.append(c)
                continue
            remaining = self.input_token_limit - used
            if remaining * 4 >= MIN_TRUNCATED_CHARS:
                truncated = c.model_copy(update={"text": _truncate(c.text[: remaining * 4], remaining * 4)})
                kept.append(truncated)
                used = self.input_token_limit
            dropped_by_tokens.extend(x.id for x in out[len(kept) :])
            break
        if dropped_by_tokens:
            notes["dropped_by_token_budget"] = dropped_by_tokens
            self._emit({"event": "input_truncated", "dropped": dropped_by_tokens, "input_tokens": used})
        return kept, notes

    def _state(self, query: str, candidates: list[Candidate]) -> dict[str, Any]:
        return {
            "query": _truncate(query, MAX_QUERY_CHARS),
            "candidates": [
                {
                    "id": c.id,
                    "text": c.text,
                    **({"timestamp": c.timestamp} if c.timestamp else {}),
                    **({"source": c.source} if c.source else {}),
                }
                for c in candidates
            ],
        }

    def _parse(
        self,
        answers: dict[str, Any],
        candidates: list[Candidate],
        heads: tuple[HeadName, ...],
    ) -> dict[str, HeadJudgments]:
        judgments: dict[str, HeadJudgments] = {}
        for c in candidates:
            fields: dict[str, float] = {
                "relevance": 0.0,
                "relevance_confidence": 1.0,
                "utility": 0.0,
                "utility_confidence": 1.0,
                "superseded": 0.0,
                "conflict": 0.0,
            }
            if "rel" in heads:
                rel = answers[f"rel__{c.id}"]
                fields["relevance"] = float(rel["score"])
                fields["relevance_confidence"] = float(rel["confidence"])
            if "util" in heads:
                util = answers[f"util__{c.id}"]
                fields["utility"] = float(util["score"])
                fields["utility_confidence"] = float(util["confidence"])
            if "sup" in heads:
                fields["superseded"] = float(answers[f"sup__{c.id}"]["noul"])
            if "con" in heads:
                fields["conflict"] = float(answers[f"con__{c.id}"]["noul"])
            judgments[c.id] = HeadJudgments(**fields)
        return judgments

    def _fallback_items(self, query: str, candidates: list[Candidate]) -> list[RankedItem]:
        if callable(self.fallback):
            return self.fallback(query, candidates)
        return retrieval_order_fallback(query, candidates)

    # -- sync public -------------------------------------------------------
    def judge_once(
        self,
        query: str,
        candidates: list[Candidate],
        mode: str = "memory",
    ) -> tuple[list[RankedItem], dict[str, Any]]:
        """Run ONE batched judgment + policy. Returns (items, meta)."""
        heads = heads_for_mode(mode)
        request_id = f"jr_{uuid.uuid4().hex[:12]}"
        if not candidates:
            empty_meta = {"request_id": request_id, "cached": False, "fallback_used": False, "usage": {}}
            return [], {**empty_meta, "latency_ms": 0.0}
        cands, notes = self._prepare(query, candidates)
        key = cache_key(
            query,
            [(c.id, c.text) for c in cands],
            self.model,
            self.policy.version,
            QUESTION_SCHEMA_VERSION,
            heads,
        )
        hit = self.cache.get(key)
        if hit is not None:
            self.cache_hits += 1
            judgments = {cid: HeadJudgments(**j) for cid, j in hit["judgments"].items()}
            items = apply_policy(cands, judgments, self.policy, heads)
            self._emit({"event": "cache_hit", "request_id": request_id, "mode": mode})
            return items, {
                "request_id": request_id,
                "cached": True,
                "fallback_used": False,
                "usage": hit.get("usage", {}),
                "latency_ms": 0.0,
                **notes,
            }
        state = self._state(query, cands)
        questions = build_questions([c.id for c in cands], heads)
        t0 = time.perf_counter()
        try:
            answers, usage = self.judge.judge(state, questions, query, cands)
        except Exception:
            items = self._fallback_items(query, cands)
            self._emit({"event": "fallback", "request_id": request_id, "mode": mode})
            return items, {
                "request_id": request_id,
                "cached": False,
                "fallback_used": True,
                "usage": {},
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                **notes,
            }
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        judgments = self._parse(answers, cands, heads)
        self.calls_made += 1
        self.cache.set(key, {"judgments": {cid: j.model_dump() for cid, j in judgments.items()}, "usage": usage})
        items = apply_policy(cands, judgments, self.policy, heads)
        self._emit(
            {
                "event": "judge_call",
                "request_id": request_id,
                "mode": mode,
                "latency_ms": latency_ms,
                "n_candidates": len(cands),
                "n_questions": len(questions),
                "usage": usage,
            }
        )
        return items, {
            "request_id": request_id,
            "cached": False,
            "fallback_used": False,
            "usage": usage,
            "latency_ms": latency_ms,
            **notes,
        }

    # -- async public ------------------------------------------------------
    async def ajudge_once(
        self,
        query: str,
        candidates: list[Candidate],
        mode: str = "memory",
    ) -> tuple[list[RankedItem], dict[str, Any]]:
        """Async twin of :meth:`judge_once`.

        Uses the judge's native async path when available (AsyncLiveJevJudge);
        otherwise runs the sync judge in the default executor.
        """
        if not is_async_judge(self.judge):
            import asyncio

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self.judge_once(query, candidates, mode))
        heads = heads_for_mode(mode)
        request_id = f"jr_{uuid.uuid4().hex[:12]}"
        if not candidates:
            empty_meta = {"request_id": request_id, "cached": False, "fallback_used": False, "usage": {}}
            return [], {**empty_meta, "latency_ms": 0.0}
        cands, notes = self._prepare(query, candidates)
        key = cache_key(
            query,
            [(c.id, c.text) for c in cands],
            self.model,
            self.policy.version,
            QUESTION_SCHEMA_VERSION,
            heads,
        )
        hit = self.cache.get(key)
        if hit is not None:
            self.cache_hits += 1
            judgments = {cid: HeadJudgments(**j) for cid, j in hit["judgments"].items()}
            items = apply_policy(cands, judgments, self.policy, heads)
            self._emit({"event": "cache_hit", "request_id": request_id, "mode": mode})
            return items, {
                "request_id": request_id,
                "cached": True,
                "fallback_used": False,
                "usage": hit.get("usage", {}),
                "latency_ms": 0.0,
                **notes,
            }
        state = self._state(query, cands)
        questions = build_questions([c.id for c in cands], heads)
        t0 = time.perf_counter()
        try:
            answers, usage = await self.judge.judge(state, questions, query, cands)
        except Exception:
            items = self._fallback_items(query, cands)
            self._emit({"event": "fallback", "request_id": request_id, "mode": mode})
            return items, {
                "request_id": request_id,
                "cached": False,
                "fallback_used": True,
                "usage": {},
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                **notes,
            }
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        judgments = self._parse(answers, cands, heads)
        self.calls_made += 1
        self.cache.set(key, {"judgments": {cid: j.model_dump() for cid, j in judgments.items()}, "usage": usage})
        items = apply_policy(cands, judgments, self.policy, heads)
        self._emit(
            {
                "event": "judge_call",
                "request_id": request_id,
                "mode": mode,
                "latency_ms": latency_ms,
                "n_candidates": len(cands),
                "n_questions": len(questions),
                "usage": usage,
            }
        )
        return items, {
            "request_id": request_id,
            "cached": False,
            "fallback_used": False,
            "usage": usage,
            "latency_ms": latency_ms,
            **notes,
        }

    async def aclose(self) -> None:
        close = getattr(self.judge, "aclose", None)
        if close is not None:
            await close()
