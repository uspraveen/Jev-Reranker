"""LangGraph integration: ``memory_triage_node`` for agentic memory pipelines."""

from __future__ import annotations

from typing import Any

from jev_reranker.integrations.langchain import _candidates_from_texts
from jev_reranker.models import Label, PolicyConfig
from jev_reranker.reranker import JevReranker


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
