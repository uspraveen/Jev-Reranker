"""LangChain integration: ``JevRerankCompressor`` document compressor.

Duck-typed (no hard dependency): works with any object exposing
``page_content``/``metadata``. Optional extra: ``pip install jev-reranker[integrations]``.
"""

from __future__ import annotations

from typing import Any

from jev_reranker.models import Candidate
from jev_reranker.reranker import JevReranker, Mode


def _candidates_from_texts(texts: list[str], metadatas: list[dict[str, Any]] | None = None) -> list[Candidate]:
    cands: list[Candidate] = []
    for i, text in enumerate(texts):
        meta = (metadatas or [{}] * len(texts))[i] if metadatas else {}
        cands.append(
            Candidate(
                id=str(meta.get("id", f"c{i}")),
                text=text,
                source=meta.get("source"),
                timestamp=meta.get("timestamp"),
                retrieval_rank=i,
                retrieval_score=meta.get("retrieval_score"),
            )
        )
    return cands


class JevRerankCompressor:
    """LangChain document compressor (``pip install jev-reranker[integrations]``)."""

    def __init__(self, reranker: JevReranker | None = None, top_n: int = 5, mode: Mode = "relevance") -> None:
        self.reranker = reranker or JevReranker()
        self.top_n = top_n
        self.mode = mode

    def compress_documents(self, documents: list[Any], query: str) -> list[Any]:
        texts = [d.page_content if hasattr(d, "page_content") else str(d) for d in documents]
        metas = [d.metadata if hasattr(d, "metadata") else {} for d in documents]
        result = self.reranker.rerank(query, _candidates_from_texts(texts, metas), mode=self.mode, top_k=self.top_n)
        return _attach(documents, metas, result)

    async def acompress_documents(self, documents: list[Any], query: str) -> list[Any]:
        """Async twin — awaits the reranker directly instead of blocking."""
        texts = [d.page_content if hasattr(d, "page_content") else str(d) for d in documents]
        metas = [d.metadata if hasattr(d, "metadata") else {} for d in documents]
        result = await self.reranker.arerank(
            query, _candidates_from_texts(texts, metas), mode=self.mode, top_k=self.top_n
        )
        return _attach(documents, metas, result)


def _attach(documents: list[Any], metas: list[dict[str, Any]], result: Any) -> list[Any]:
    id2doc = {str(m.get("id", f"c{i}")): d for i, (d, m) in enumerate(zip(documents, metas, strict=True))}
    out: list[Any] = []
    for item in result.items:
        doc = id2doc[item.candidate.id]
        if hasattr(doc, "metadata"):
            doc.metadata["jev_value"] = item.value
            doc.metadata["jev_label"] = item.label.value
            doc.metadata["jev_relevance_score"] = item.relevance_score
        out.append(doc)
    return out
