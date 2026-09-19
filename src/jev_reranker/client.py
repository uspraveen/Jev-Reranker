"""JevClient: one batched Jev call per rerank + resilience layers.

Resilience (host agent never goes brittle):
- candidate limits (MAX_CANDIDATES) + per-candidate char truncation
- timeouts (via judge / SDK client)
- cache keyed on query+candidates+model+policy+schema versions
- request IDs on every result + trace
- fallback='retrieval_order' or a custom callable when Jev is unreachable
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from jev_reranker.cache import JudgmentCache, cache_key
from jev_reranker.judges import Judge, LiveJevJudge, OfflineJudge
from jev_reranker.models import Candidate, FallbackType, HeadJudgments, Label, PolicyConfig, RankedItem
from jev_reranker.policy import QUESTION_SCHEMA_VERSION, apply_policy, build_questions

MAX_CANDIDATES = 100
MAX_CANDIDATE_CHARS = 4000
MAX_QUERY_CHARS = 2000


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
            relevance=1.0,
            relevance_confidence=0.0,  # zero confidence -> UNCERTAIN by construction
            utility=1.0,
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
        judge: Judge | None = None,
        model: str = "jev-latest",
        policy: PolicyConfig | None = None,
        cache: JudgmentCache | None = None,
        cache_path: Path | str | None = None,
        fallback: FallbackType = "retrieval_order",
        max_candidates: int = MAX_CANDIDATES,
    ) -> None:
        self.policy = policy or PolicyConfig()
        self.model = model
        self.fallback = fallback
        self.max_candidates = max_candidates
        if judge is not None:
            self.judge = judge
            if not model or model == "jev-latest":
                self.model = getattr(judge, "model_name", model)
        elif os.environ.get("TYPESAFE_API_KEY"):
            self.judge = LiveJevJudge(model=model)
        else:
            self.judge = OfflineJudge()
        if cache is not None:
            self.cache = cache
        elif cache_path is not None:
            self.cache = JudgmentCache(cache_path)
        else:
            self.cache = JudgmentCache()
        self.calls_made = 0  # remote judge invocations (excludes cache hits)

    # -- internals ---------------------------------------------------------
    def _prepare(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        if len(candidates) > self.max_candidates:
            ordered = sorted(
                candidates,
                key=lambda c: (c.retrieval_rank if c.retrieval_rank is not None else 10**9, c.id),
            )
            candidates = ordered[: self.max_candidates]
        out: list[Candidate] = []
        for c in candidates:
            out.append(c.model_copy(update={"text": _truncate(c.text, MAX_CANDIDATE_CHARS)}))
        return out

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

    def _parse(self, answers: dict[str, Any], candidates: list[Candidate]) -> dict[str, HeadJudgments]:
        judgments: dict[str, HeadJudgments] = {}
        for c in candidates:
            rel = answers[f"rel__{c.id}"]
            util = answers[f"util__{c.id}"]
            sup = answers[f"sup__{c.id}"]
            con = answers[f"con__{c.id}"]
            judgments[c.id] = HeadJudgments(
                relevance=float(rel["score"]),
                relevance_confidence=float(rel["confidence"]),
                utility=float(util["score"]),
                utility_confidence=float(util["confidence"]),
                superseded=float(sup["noul"]),
                conflict=float(con["noul"]),
            )
        return judgments

    def _fallback_items(self, query: str, candidates: list[Candidate]) -> list[RankedItem]:
        if callable(self.fallback):
            return self.fallback(query, candidates)
        return retrieval_order_fallback(query, candidates)

    # -- public ------------------------------------------------------------
    def judge_once(
        self,
        query: str,
        candidates: list[Candidate],
    ) -> tuple[list[RankedItem], dict[str, Any]]:
        """Run ONE batched judgment + policy. Returns (items, meta)."""
        request_id = f"jr_{uuid.uuid4().hex[:12]}"
        if not candidates:
            return [], {"request_id": request_id, "cached": False, "fallback_used": False, "usage": {}}
        cands = self._prepare(query, candidates)
        key = cache_key(
            query,
            [(c.id, c.text) for c in cands],
            self.model,
            self.policy.version,
            QUESTION_SCHEMA_VERSION,
        )
        hit = self.cache.get(key)
        if hit is not None:
            judgments = {cid: HeadJudgments(**j) for cid, j in hit["judgments"].items()}
            items = apply_policy(cands, judgments, self.policy)
            return items, {
                "request_id": request_id,
                "cached": True,
                "fallback_used": False,
                "usage": hit.get("usage", {}),
            }
        state = self._state(query, cands)
        questions = build_questions([c.id for c in cands])
        try:
            answers, usage = self.judge.judge(state, questions, query, cands)
        except Exception:
            items = self._fallback_items(query, cands)
            return items, {
                "request_id": request_id,
                "cached": False,
                "fallback_used": True,
                "usage": {},
            }
        judgments = self._parse(answers, cands)
        self.calls_made += 1
        self.cache.set(key, {"judgments": {cid: j.model_dump() for cid, j in judgments.items()}, "usage": usage})
        items = apply_policy(cands, judgments, self.policy)
        return items, {"request_id": request_id, "cached": False, "fallback_used": False, "usage": usage}
