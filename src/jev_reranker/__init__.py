"""Jev-Reranker: decision-aware, calibrated context selection for AI agents.

Jev (TypeSafe System One model) makes judgments; deterministic Python policy
makes the ranking. Exactly ONE ``/v1/systemone`` call per rerank/select —
all candidates x all active heads are batched into a single request.

Behavioral modes: ``relevance`` (relevance head only), ``memory`` (all four
heads with superseded/conflict detection), ``context`` (memory heads plus
token-budget selection and embedding-based near-duplicate suppression).
"""

from jev_reranker.client import JevClient
from jev_reranker.dedup import embed, similarity, suppress_near_duplicates
from jev_reranker.integrations.cohere_compat import CohereCompatReranker
from jev_reranker.judges import AsyncJudge, AsyncLiveJevJudge, Judge, LiveJevJudge, OfflineJudge
from jev_reranker.models import (
    Candidate,
    ContextSelection,
    HeadJudgments,
    Label,
    MemoryItem,
    PolicyConfig,
    RankedDocument,
    RankedItem,
    RerankResult,
    SelectionResult,
)
from jev_reranker.policy import POLICY_VERSION, QUESTION_SCHEMA_VERSION, apply_policy
from jev_reranker.reranker import JevReranker
from jev_reranker.rubric import (
    BUILTIN_RUBRICS,
    NoulHeadRubric,
    Rubric,
    ScoreHeadRubric,
    resolve_rubric,
)

__all__ = [
    "AsyncJudge",
    "AsyncLiveJevJudge",
    "BUILTIN_RUBRICS",
    "Candidate",
    "CohereCompatReranker",
    "ContextSelection",
    "HeadJudgments",
    "JevClient",
    "JevReranker",
    "Judge",
    "Label",
    "LiveJevJudge",
    "MemoryItem",
    "NoulHeadRubric",
    "OfflineJudge",
    "POLICY_VERSION",
    "QUESTION_SCHEMA_VERSION",
    "PolicyConfig",
    "RankedDocument",
    "RankedItem",
    "RerankResult",
    "Rubric",
    "ScoreHeadRubric",
    "SelectionResult",
    "apply_policy",
    "embed",
    "resolve_rubric",
    "similarity",
    "suppress_near_duplicates",
]

__version__ = "0.3.0"
