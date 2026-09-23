"""Cohere-shaped rerank surface: the ecosystem-standard contract.

``CohereCompatReranker.rerank(query, documents, top_n)`` accepts plain strings
(or ``{"text": ...}`` dicts) and returns the shape hosted rerank APIs return:
``{"id", "results": [{"index", "relevance_score", "document"}]}`` where
``index`` is positional over the input list and ``relevance_score`` is in
[0, 1] (raw relevance head / 3 — comparable across documents of one call,
not ratio-scale, mirroring Cohere's own caveat).

Jev-specific richness (labels, policy value, per-head judgments) travels in
``meta.jev`` so strict Cohere clients stay compatible. ``select`` exposes the
token-budget selection in the same plain-dict spirit.
"""

from __future__ import annotations

import uuid
from typing import Any

from jev_reranker.models import Candidate
from jev_reranker.reranker import JevReranker, Mode, run_coroutine_sync
from jev_reranker.rubric import Rubric

Doc = str | dict[str, Any]


def _doc_text(doc: Doc) -> str:
    return doc if isinstance(doc, str) else str(doc.get("text", ""))


def _candidates(documents: list[Doc]) -> list[Candidate]:
    # Positional ids: `index` in the response is the input-list position,
    # exactly the hosted-API contract (duplicates included).
    return [Candidate(id=str(i), text=_doc_text(d), retrieval_rank=i) for i, d in enumerate(documents)]


class CohereCompatReranker:
    """Drop-in rerank object speaking the hosted-API request/response shape."""

    def __init__(
        self,
        reranker: JevReranker | None = None,
        rubric: str | Rubric | None = None,
    ) -> None:
        self._rr = reranker or JevReranker(rubric=rubric)

    @property
    def model(self) -> str:
        return self._rr.client.model

    @property
    def judge_name(self) -> str:
        return type(self._rr.client.judge).__name__

    async def arerank(
        self,
        query: str,
        documents: list[Doc],
        top_n: int | None = None,
        mode: Mode = "relevance",
    ) -> dict[str, Any]:
        res = await self._rr.arerank(query, _candidates(documents), mode=mode, top_k=top_n)
        return {
            "id": res.request_id,
            "results": [
                {
                    "index": int(it.candidate.id),
                    "relevance_score": it.relevance_score,
                    "document": {"text": it.candidate.text},
                }
                for it in res.items
            ],
            "meta": {
                "model": res.model,
                "jev": {
                    "mode": res.mode,
                    "policy_version": res.policy_version,
                    "schema_version": res.schema_version,
                    "cached": res.cached,
                    "fallback_used": res.fallback_used,
                    "latency_ms": res.latency_ms,
                    "items": [
                        {"index": int(it.candidate.id), "label": it.label.value, "value": it.value}
                        for it in res.items
                    ],
                },
            },
        }

    def rerank(
        self,
        query: str,
        documents: list[Doc],
        top_n: int | None = None,
        mode: Mode = "relevance",
    ) -> dict[str, Any]:
        return run_coroutine_sync(self.arerank(query, documents, top_n=top_n, mode=mode))

    async def aselect(self, query: str, documents: list[Doc], budget_tokens: int) -> dict[str, Any]:
        sel = await self._rr.aselect(query, _candidates(documents), budget_tokens=budget_tokens)

        def entry(it: Any) -> dict[str, Any]:
            return {
                "index": int(it.candidate.id),
                "text": it.candidate.text,
                "relevance_score": it.relevance_score,
                "label": it.label.value,
                "value": it.value,
            }

        return {
            "id": f"js_{uuid.uuid4().hex[:12]}",
            "total_tokens": sel.total_tokens,
            "budget_tokens": sel.budget_tokens,
            "selected": [entry(it) for it in sel.selected],
            "excluded": [entry(it) for it in sel.excluded],
            "suppressed_near_duplicates": [entry(it) for it in sel.suppressed_near_duplicates],
            "meta": {"model": self.model, "latency_ms": sel.latency_ms, "usage": dict(sel.usage)},
        }

    def select(self, query: str, documents: list[Doc], budget_tokens: int) -> dict[str, Any]:
        return run_coroutine_sync(self.aselect(query, documents, budget_tokens=budget_tokens))
