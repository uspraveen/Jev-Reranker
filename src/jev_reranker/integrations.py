"""Framework integrations — all optional, all lazy-imported.

- LangChain: ``JevRerankCompressor`` (BaseDocumentCompressor API shape)
- LangGraph: ``memory_triage_node`` for agentic memory pipelines
- LlamaIndex: ``JevRerankPostprocessor`` (BaseNodePostprocessor shape)
- Qdrant: ``rerank_qdrant_hits`` helper over ``qdrant_client`` scored points

Each adapter degrades to a clear ImportError naming the missing extra.
"""

from __future__ import annotations

from typing import Any

from jev_reranker.models import Candidate, Label, PolicyConfig
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
        id2doc = {str(m.get("id", f"c{i}")): d for i, (d, m) in enumerate(zip(documents, metas, strict=True))}
        out: list[Any] = []
        for item in result.items:
            doc = id2doc[item.candidate.id]
            if hasattr(doc, "metadata"):
                doc.metadata["jev_value"] = item.value
                doc.metadata["jev_label"] = item.label.value
            out.append(doc)
        return out


def memory_triage_node(policy: PolicyConfig | None = None) -> Any:
    """LangGraph node: triage ``state['memories']`` for ``state['query']``.

    Returns ``{'ranked_memories': [...], 'dropped': [...], 'conflicts': [...]}``.
    """

    def node(state: dict[str, Any]) -> dict[str, Any]:
        reranker = JevReranker(policy=policy)
        texts: list[str] = state.get("memories", [])
        query: str = state.get("query", "")
        result = reranker.rerank(query, _candidates_from_texts(texts), mode="memory")
        return {
            "ranked_memories": [it.candidate.text for it in result.items if it.label not in (Label.DROP,)],
            "dropped": [it.candidate.id for it in result.items if it.label is Label.DROP],
            "conflicts": [it.candidate.id for it in result.items if it.label is Label.CONFLICT],
            "request_id": result.request_id,
        }

    return node


class JevRerankPostprocessor:
    """LlamaIndex node postprocessor (duck-typed; no hard dependency)."""

    def __init__(self, reranker: JevReranker | None = None, top_n: int = 5) -> None:
        self.reranker = reranker or JevReranker()
        self.top_n = top_n

    def postprocess_nodes(self, nodes: list[Any], query_str: str = "") -> list[Any]:
        texts = [n.get_content() if hasattr(n, "get_content") else str(n) for n in nodes]
        result = self.reranker.rerank(query_str, _candidates_from_texts(texts), top_k=self.top_n)
        kept = [nodes[int(item.candidate.id[1:])] for item in result.items if item.candidate.id[1:].isdigit()]
        for item in result.items:
            idx = int(item.candidate.id[1:]) if item.candidate.id[1:].isdigit() else -1
            if 0 <= idx < len(nodes) and hasattr(nodes[idx], "metadata"):
                nodes[idx].metadata["jev_value"] = item.value
        return kept if kept else [nodes[int(i.candidate.id[1:])] for i in result.items if i.candidate.id[1:].isdigit()]


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
