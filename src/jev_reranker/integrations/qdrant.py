"""Qdrant integration: rerank ``ScoredPoint`` hits with Jev."""

from __future__ import annotations

from typing import Any

from jev_reranker.integrations.langchain import _candidates_from_texts
from jev_reranker.reranker import JevReranker


def rerank_qdrant_hits(
    query: str,
    hits: list[Any],
    reranker: JevReranker | None = None,
    top_k: int = 5,
) -> list[Any]:
    """Reorder Qdrant ``ScoredPoint`` hits with Jev; annotates payload with jev_value/label."""
    rr = reranker or JevReranker()
    texts = [(h.payload or {}).get("text", "") if hasattr(h, "payload") else str(h) for h in hits]
    metas = [{"id": str(getattr(h, "id", i)), "retrieval_score": getattr(h, "score", None)} for i, h in enumerate(hits)]
    result = rr.rerank(query, _candidates_from_texts(texts, metas), top_k=top_k)
    id2hit = {m["id"]: h for m, h in zip(metas, hits, strict=True)}
    out: list[Any] = []
    for item in result.items:
        h = id2hit[item.candidate.id]
        if hasattr(h, "payload") and isinstance(h.payload, dict):
            h.payload["jev_value"] = item.value
            h.payload["jev_label"] = item.label.value
        out.append(h)
    return out
