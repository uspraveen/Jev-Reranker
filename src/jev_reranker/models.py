"""Pydantic models for Jev-Reranker. Full type hints; no untyped defs."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Label(str, Enum):
    """Deterministic policy labels assigned per candidate."""

    USE = "USE"  # top-ranked / above use threshold: feed to the agent
    KEEP = "KEEP"  # useful background: keep if budget allows
    DROP = "DROP"  # below value threshold
    STALE = "STALE"  # superseded by newer information
    CONFLICT = "CONFLICT"  # contradicts other candidates / known facts
    UNCERTAIN = "UNCERTAIN"  # judge confidence below gate: needs review


class Candidate(BaseModel):
    """One retrieval candidate to be judged."""

    id: str = Field(description="Stable candidate id (used as question key prefix)")
    text: str = Field(description="Candidate text judged by Jev")
    timestamp: str | None = Field(default=None, description="ISO timestamp, for recency/superseded reasoning")
    source: str | None = Field(default=None, description="Origin, e.g. memory store, retriever name")
    retrieval_rank: int | None = Field(default=None, description="Original retriever rank (0-based)")
    retrieval_score: float | None = Field(default=None, description="Original retriever score, if any")
    token_estimate: int | None = Field(default=None, description="Precomputed token estimate; else ~len/4")


class MemoryItem(Candidate):
    """A Candidate that is an agent memory (adds writer/recency metadata)."""

    memory_type: str | None = Field(default=None, description="episodic | semantic | procedural | ...")
    agent_id: str | None = Field(default=None)


class HeadJudgments(BaseModel):
    """Raw Jev judgments for one candidate (one Score/Score/Noul/Noul head each)."""

    relevance: float = Field(ge=0.0, le=2.0, description="Score head 0..2")
    relevance_confidence: float = Field(ge=0.0, le=1.0)
    utility: float = Field(ge=0.0, le=2.0, description="Score head 0..2")
    utility_confidence: float = Field(ge=0.0, le=1.0)
    superseded: float = Field(ge=0.0, le=1.0, description="Noul head: P(memory is superseded)")
    conflict: float = Field(ge=0.0, le=1.0, description="Noul head: P(conflicts with other candidates)")

    @property
    def min_confidence(self) -> float:
        # Noul heads carry no confidence; gate on the two Score confidences.
        return min(self.relevance_confidence, self.utility_confidence)


class PolicyConfig(BaseModel):
    """Transparent, configurable coefficients for the deterministic policy.

    Confidence NEVER multiplies into the score — it only gates labels/actions.
    """

    version: str = Field(default="v1")
    w_relevance: float = 0.55
    w_utility: float = 0.45
    w_superseded: float = 0.80  # penalty weight
    w_conflict: float = 0.60  # penalty weight
    use_threshold: float = 0.55  # value >= this -> USE
    drop_threshold: float = 0.25  # value < this -> DROP (else KEEP)
    stale_threshold: float = 0.65  # superseded noul >= this (+confident) -> STALE
    conflict_threshold: float = 0.65  # conflict noul >= this (+confident) -> CONFLICT
    confidence_gate: float = 0.45  # min head confidence below this -> UNCERTAIN


class RankedItem(BaseModel):
    """One candidate after policy application."""

    candidate: Candidate
    judgments: HeadJudgments
    value: float = Field(description="Deterministic policy value (higher is better)")
    label: Label
    rank: int = Field(description="0-based final rank")


class RerankResult(BaseModel):
    """Full result of one rerank/select call."""

    request_id: str
    mode: Literal["relevance", "memory", "context"]
    query: str
    model: str
    policy_version: str
    schema_version: str
    items: list[RankedItem]  # sorted best-first
    usage: dict[str, int] = Field(default_factory=dict)
    cached: bool = False
    fallback_used: bool = False
    trace: dict[str, Any] = Field(default_factory=dict)


class ContextSelection(BaseModel):
    """Token-budget selection result (context mode)."""

    selected: list[RankedItem]
    total_tokens: int
    budget_tokens: int
    excluded: list[RankedItem]


FallbackKind = Literal["retrieval_order"]
FallbackType = FallbackKind | Callable[[str, list[Candidate]], list[RankedItem]]
"""Fallback when Jev is unreachable: 'retrieval_order' or a custom callable."""
