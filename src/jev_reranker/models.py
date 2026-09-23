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
    """A Candidate that is an agent memory: writer/recency/provenance metadata.

    Metadata is first-class *input*: it feeds deterministic code (e.g.
    ``PolicyConfig.source_priority`` tie-breaking) and is part of the state
    Jev sees, but it is never itself a question to Jev.
    """

    memory_type: str | None = Field(default=None, description="episodic | semantic | procedural | ...")
    agent_id: str | None = Field(default=None)
    created_at: str | None = Field(default=None, description="ISO-8601 write time of the memory")
    importance: float | None = Field(default=None, ge=0.0, le=1.0, description="Writer-assigned importance 0..1")
    entity_ids: list[str] = Field(default_factory=list, description="Entities this memory is about")
    session_id: str | None = Field(default=None, description="Conversation/session the memory came from")


class HeadJudgments(BaseModel):
    """Raw Jev judgments for one candidate (Score 0-3 x2, Noul 0-1 x2)."""

    relevance: float = Field(ge=0.0, le=3.0, description="Score head 0..3")
    relevance_confidence: float = Field(ge=0.0, le=1.0)
    utility: float = Field(ge=0.0, le=3.0, description="Score head 0..3")
    utility_confidence: float = Field(ge=0.0, le=1.0)
    superseded: float = Field(ge=0.0, le=1.0, description="Noul head: P(memory is superseded)")
    conflict: float = Field(ge=0.0, le=1.0, description="Noul head: P(conflicts with other candidates)")

    @property
    def min_confidence(self) -> float:
        # Noul heads carry no confidence; gate on the two Score confidences.
        return min(self.relevance_confidence, self.utility_confidence)


HeadName = Literal["rel", "util", "sup", "con"]
MODE_HEADS: dict[str, tuple[HeadName, ...]] = {
    "relevance": ("rel",),
    "memory": ("rel", "util", "sup", "con"),
    "context": ("rel", "util", "sup", "con"),
}


class PolicyConfig(BaseModel):
    """Transparent, configurable coefficients for the deterministic policy.

    Default scoring (owner section 7): ``U = 0.55*R + 0.45*V`` and
    ``J = U*(1-0.75*S)*(1-0.25*C)`` — the multiplicative penalty form.
    ``penalty_form="additive"`` keeps the legacy ``U - 0.80*S - 0.60*C``.

    Confidence NEVER multiplies into the score — it only gates labels/actions.
    ``source_priority`` is deterministic metadata applied in code; never a
    question to Jev.
    """

    version: str = Field(default="v2")
    # Composite value (owner section 7): U_i = u_relevance*R_i + u_utility*V_i,
    # R/V normalized 0..3 -> 0..1.
    u_relevance: float = 0.55
    u_utility: float = 0.45
    penalty_form: Literal["multiplicative", "additive"] = "multiplicative"
    # Multiplicative form: J = U*(1-superseded_penalty*S)*(1-conflict_penalty*C)
    superseded_penalty: float = 0.75
    conflict_penalty: float = 0.25
    # Legacy additive form: J = U - w_superseded*S - w_conflict*C
    w_superseded: float = 0.80
    w_conflict: float = 0.60
    use_threshold: float = 0.55  # value >= this -> USE
    drop_threshold: float = 0.25  # value < this -> DROP (else KEEP)
    stale_threshold: float = 0.65  # superseded noul >= this -> STALE
    conflict_threshold: float = 0.65  # conflict noul >= this -> CONFLICT
    confidence_gate: float = 0.45  # score-head confidence below this -> UNCERTAIN (value labels only)
    # Deterministic source tie-breaking: higher wins when values tie.
    source_priority: dict[str, float] = Field(default_factory=dict)
    default_source_priority: float = 0.5

    def source_priority_of(self, source: str | None) -> float:
        if source is None:
            return self.default_source_priority
        return self.source_priority.get(source, self.default_source_priority)


class RankedItem(BaseModel):
    """One candidate after policy application."""

    candidate: Candidate
    judgments: HeadJudgments
    value: float = Field(description="Deterministic policy value (higher is better)")
    label: Label
    rank: int = Field(description="0-based final rank")

    @property
    def relevance_score(self) -> float:
        """Ecosystem-standard 0..1 relevance (raw relevance head / 3).

        Independent of the policy value: it does NOT mix in utility or
        superseded/conflict penalties. Use for drop-in rerank comparisons;
        use ``value`` for policy decisions.
        """
        return self.judgments.relevance / 3.0

    @property
    def decisions(self) -> HeadJudgments:
        """Owner-section-9 name for the per-candidate head decisions."""
        return self.judgments


class RankedDocument(BaseModel):
    """Document-shaped view of one ranked candidate (rerank mode)."""

    id: str
    text: str
    score: float = Field(description="Policy value of the document")
    relevance_score: float = Field(description="Raw relevance head, normalized 0..1")
    label: Label
    rank: int
    decisions: HeadJudgments
    metadata: dict[str, Any] = Field(default_factory=dict)


def _split(items: list[RankedItem], labels: frozenset[Label]) -> list[RankedItem]:
    return [it for it in items if it.label in labels]


class RerankResult(BaseModel):
    """Full result of one rerank/select call.

    Splits are disjoint and cover ``items``:
    selected = {USE, KEEP}; rejected = {DROP, STALE};
    uncertain = {UNCERTAIN}; conflicts = {CONFLICT}.
    """

    request_id: str
    mode: Literal["relevance", "memory", "context"]
    query: str
    model: str
    policy_version: str
    schema_version: str
    items: list[RankedItem]  # sorted best-first
    usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: float | None = Field(default=None, description="Judge-call wall time in ms")
    cached: bool = False
    fallback_used: bool = False
    trace: dict[str, Any] = Field(default_factory=dict)

    @property
    def labels(self) -> list[str]:
        return [it.label.value for it in self.items]

    @property
    def selected(self) -> list[RankedItem]:
        return _split(self.items, frozenset({Label.USE, Label.KEEP}))

    @property
    def rejected(self) -> list[RankedItem]:
        return _split(self.items, frozenset({Label.DROP, Label.STALE}))

    @property
    def uncertain(self) -> list[RankedItem]:
        return _split(self.items, frozenset({Label.UNCERTAIN}))

    @property
    def conflicts(self) -> list[RankedItem]:
        return _split(self.items, frozenset({Label.CONFLICT}))

    @property
    def documents(self) -> list[RankedDocument]:
        """Rerank-mode view: plain documents with decisions attached."""
        return [
            RankedDocument(
                id=it.candidate.id,
                text=it.candidate.text,
                score=it.value,
                relevance_score=it.relevance_score,
                label=it.label,
                rank=it.rank,
                decisions=it.judgments,
                metadata={
                    k: v
                    for k, v in {
                        "source": it.candidate.source,
                        "timestamp": it.candidate.timestamp,
                        "retrieval_rank": it.candidate.retrieval_rank,
                    }.items()
                    if v is not None
                },
            )
            for it in self.items
        ]


SelectionResult = RerankResult
"""Owner-section-9 alias: results of selection are rich rerank results."""


class ContextSelection(BaseModel):
    """Token-budget selection result (context mode)."""

    selected: list[RankedItem]
    total_tokens: int
    budget_tokens: int
    excluded: list[RankedItem]
    suppressed_near_duplicates: list[RankedItem] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: float | None = None

    @property
    def uncertain(self) -> list[RankedItem]:
        return _split(self.selected, frozenset({Label.UNCERTAIN}))

    @property
    def conflicts(self) -> list[RankedItem]:
        return _split(self.selected, frozenset({Label.CONFLICT}))


FallbackKind = Literal["retrieval_order"]
FallbackType = FallbackKind | Callable[[str, list[Candidate]], list[RankedItem]]
"""Fallback when Jev is unreachable: 'retrieval_order' or a custom callable."""

TelemetryCallback = Callable[[dict[str, Any]], None]
"""Optional hook receiving structured events (judge_call, cache_hit, fallback, ...)."""
