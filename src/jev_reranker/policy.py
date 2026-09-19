"""Deterministic ranking policy.

Jev makes judgments (relevance Score, utility Score, superseded Noul,
conflict Noul). This module — pure Python, no model calls — turns those
judgments into a ranking. Transparent, configurable coefficients, versioned.

Hard rule: confidence NEVER multiplies into the score. It only gates
labels (below-gate items become UNCERTAIN instead of acting on the value).
"""

from __future__ import annotations

from jev_reranker.models import Candidate, HeadJudgments, Label, PolicyConfig, RankedItem

POLICY_VERSION = "v1"
QUESTION_SCHEMA_VERSION = "v1"

# Question-schema: 4 heads per candidate, all asked in ONE /v1/systemone call.
RELEVANCE_LEVELS = ["irrelevant to the query", "partially relevant background", "directly answers the query"]
UTILITY_LEVELS = ["no actionable content", "useful background only", "directly usable to act"]


def question_key(kind: str, candidate_id: str) -> str:
    """Question map key for head ``kind`` of ``candidate_id``."""
    return f"{kind}__{candidate_id}"


def build_questions(candidate_ids: list[str]) -> dict[str, dict[str, object]]:
    """Build the 4N question payload (JSON-serializable) for one batched call."""
    questions: dict[str, dict[str, object]] = {}
    for cid in candidate_ids:
        questions[question_key("rel", cid)] = {
            "type": "score",
            "instructions": f"How relevant is candidate [{cid}] to the query?",
            "criteria": RELEVANCE_LEVELS,
        }
        questions[question_key("util", cid)] = {
            "type": "score",
            "instructions": f"How actionable/useful is candidate [{cid}] for acting on the query?",
            "criteria": UTILITY_LEVELS,
        }
        questions[question_key("sup", cid)] = {
            "type": "noul",
            "instructions": f"Is candidate [{cid}] superseded or stale given the query and the other candidates?",
            "criteria": {
                "true": "A newer candidate (or the query itself) replaces or invalidates it",
                "false": "It still stands on its own",
            },
        }
        questions[question_key("con", cid)] = {
            "type": "noul",
            "instructions": f"Does candidate [{cid}] conflict with the query or the other candidates?",
            "criteria": {
                "true": "It asserts something contradicted elsewhere",
                "false": "It is consistent with the rest",
            },
        }
    return questions


def policy_value(j: HeadJudgments, cfg: PolicyConfig) -> float:
    """Deterministic value in roughly [-1.4, 1.0]. Higher is better.

    Scores are normalized 0..2 -> 0..1. Confidence is deliberately absent.
    """
    rel = j.relevance / 2.0
    util = j.utility / 2.0
    return cfg.w_relevance * rel + cfg.w_utility * util - cfg.w_superseded * j.superseded - cfg.w_conflict * j.conflict


def assign_label(j: HeadJudgments, value: float, cfg: PolicyConfig) -> Label:
    """Label assignment. Confidence gates; it never scales the value."""
    confident = j.min_confidence >= cfg.confidence_gate
    if not confident:
        return Label.UNCERTAIN
    if j.conflict >= cfg.conflict_threshold:
        return Label.CONFLICT
    if j.superseded >= cfg.stale_threshold:
        return Label.STALE
    if value >= cfg.use_threshold:
        return Label.USE
    if value < cfg.drop_threshold:
        return Label.DROP
    return Label.KEEP


def apply_policy(
    candidates: list[Candidate],
    judgments: dict[str, HeadJudgments],
    cfg: PolicyConfig | None = None,
) -> list[RankedItem]:
    """Rank candidates best-first. Pure function — no I/O, no model calls."""
    cfg = cfg or PolicyConfig()
    items: list[RankedItem] = []
    for cand in candidates:
        j = judgments[cand.id]
        value = policy_value(j, cfg)
        items.append(RankedItem(candidate=cand, judgments=j, value=value, label=assign_label(j, value, cfg), rank=0))
    # Stable sort: value desc, then confident-first, then retrieval_rank, then id.
    items.sort(
        key=lambda it: (
            -it.value,
            0 if it.label is not Label.UNCERTAIN else 1,
            it.candidate.retrieval_rank if it.candidate.retrieval_rank is not None else 10**9,
            it.candidate.id,
        )
    )
    for rank, item in enumerate(items):
        item.rank = rank
    return items
