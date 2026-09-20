"""LlamaIndex integration: ``JevRerankPostprocessor`` node postprocessor."""

from __future__ import annotations

from typing import Any

from jev_reranker.integrations.langchain import _candidates_from_texts
from jev_reranker.reranker import JevReranker


class JevRerankPostprocessor:
    """LlamaIndex node postprocessor (duck-typed; no hard dependency)."""

    def __init__(self, reranker: JevReranker | None = None, top_n: int = 5) -> None:
        self.reranker = reranker or JevReranker()
        self.top_n = top_n

    def postprocess_nodes(self, nodes: list[Any], query_str: str = "") -> list[Any]:
        texts = [n.get_content() if hasattr(n, "get_content") else str(n) for n in nodes]
        result = self.reranker.rerank(query_str, _candidates_from_texts(texts), top_k=self.top_n)
        kept: list[Any] = []
        for item in result.items:
            idx = int(item.candidate.id[1:]) if item.candidate.id[1:].isdigit() else -1
            if 0 <= idx < len(nodes):
                if hasattr(nodes[idx], "metadata"):
                    nodes[idx].metadata["jev_value"] = item.value
                    nodes[idx].metadata["jev_label"] = item.label.value
                kept.append(nodes[idx])
        return kept
