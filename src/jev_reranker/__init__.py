"""Jev-Reranker: decision-aware, calibrated context selection for AI agents.

Jev (TypeSafe System One model) makes judgments; deterministic Python policy
makes the ranking. Exactly ONE ``/v1/systemone`` call per rerank/select —
all candidates x all decision heads are batched into a single request.

Three modes: ``relevance`` (classic rerank), ``memory`` (agent memory
triage with superseded/conflict detection), ``context`` (token-budget
selection maximizing context value per token).
"""

from jev_reranker.client import JevClient
from jev_reranker.models import (
    Candidate,
    ContextSelection,
    Label,
    MemoryItem,
    PolicyConfig,
    RankedItem,
    RerankResult,
)
from jev_reranker.policy import POLICY_VERSION, QUESTION_SCHEMA_VERSION, apply_policy
from jev_reranker.reranker import JevReranker

__all__ = [
    "Candidate",
    "ContextSelection",
    "JevClient",
    "JevReranker",
    "Label",
    "MemoryItem",
    "POLICY_VERSION",
    "QUESTION_SCHEMA_VERSION",
    "PolicyConfig",
    "RankedItem",
    "RerankResult",
    "apply_policy",
]

__version__ = "0.1.0"
