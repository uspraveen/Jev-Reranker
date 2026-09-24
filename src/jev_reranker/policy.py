"""Deterministic ranking policy.

Jev makes judgments (relevance Score 0-3, utility Score 0-3, superseded Noul,
conflict Noul). This module - pure Python, no model calls - turns those
judgments into a ranking. Transparent, configurable coefficients, versioned.

Hard rules:
- confidence NEVER multiplies into the score; it only gates labels
  (below-gate items become UNCERTAIN instead of acting on the value);
- ``source_priority`` metadata tie-breaking happens here in code, never as a
  question to Jev.
"""

from __future__ import annotations

from jev_reranker.models import (
    MODE_HEADS,
    Candidate,
    HeadJudgments,
    HeadName,
    Label,
    PolicyConfig,
    RankedItem,
)
from jev_reranker.rubric import AGENT_MEMORY, Rubric, resolve_rubric

POLICY_VERSION = "v2"
QUESTION_SCHEMA_VERSION = "v2"

# Question-schema: up to 4 heads per candidate, all asked in ONE
# /v1/systemone call. Score heads use the owner's 0-3 rubric (4 levels).
# The question texts themselves are domain-configurable (see rubric.py);
# these aliases expose the default agent_memory criteria.
RELEVANCE_LEVELS = list(AGENT_MEMORY.rel.criteria)
UTILITY_LEVELS = list(AGENT_MEMORY.util.criteria)

HEAD_KINDS: dict[str, str] = {"rel": "score", "util": "score", "sup": "noul", "con": "noul"}


def question_key(kind: str, candidate_id: str) -> str:
    """Question map key for head ``kind`` of ``candidate_id``."""
    return f"{kind}__{candidate_id}"


def build_questions(
    candidate_ids: list[str],
    heads: tuple[HeadName, ...] = ("rel", "util", "sup", "con"),
    rubric: str | Rubric | None = None,
) -> dict[str, dict[str, object]]:
    """Build the (heads x N) question payload (JSON-serializable) for one batched call.

    ``rubric`` selects the question texts: ``None`` -> the default
    ``agent_memory`` preset, a preset name, or a custom :class:`Rubric`.
    """
    rb = resolve_rubric(rubric)
    questions: dict[str, dict[str, object]] = {}
    for cid in candidate_ids:
        for head in heads:
            key = question_key(head, cid)
            spec = rb.head(head)
            instructions = spec.instructions.replace("[cid]", f"[{cid}]")
            if spec.kind == "score":
                questions[key] = {
                    "type": "score",
                    "instructions": instructions,
                    "criteria": list(spec.criteria),
                }
            else:
                questions[key] = {
                    "type": "noul",
                    "instructions": instructions,
                    "criteria": dict(spec.criteria),
                }
    return questions


def heads_for_mode(mode: str) -> tuple[HeadName, ...]:
    """Heads judged per mode: relevance-only, or all four for memory/context."""
    return MODE_HEADS[mode]


ALL_HEADS: tuple[HeadName, ...] = ("rel", "util", "sup", "con")


def policy_value(j: HeadJudgments, cfg: PolicyConfig, heads: tuple[HeadName, ...] = ALL_HEADS) -> float:
    """Deterministic value; higher is better.

    Scores are normalized 0..3 -> 0..1. Confidence is deliberately absent.

    - relevance-only mode (``heads == ("rel",)``): value = R.
    - default (owner section 7): U = 0.55*R + 0.45*V,
      J = U*(1-0.75*S)*(1-0.25*C)  [multiplicative, ``penalty_form`` default]
      or J = U - 0.80*S - 0.60*C   [legacy additive].
    """
    rel = j.relevance / 3.0
    if heads == ("rel",):
        return rel
    util = j.utility / 3.0
    u = cfg.u_relevance * rel + cfg.u_utility * util
    sup_pen = j.superseded if "sup" in heads else 0.0
    con_pen = j.conflict if "con" in heads else 0.0
    if cfg.penalty_form == "multiplicative":
        return u * (1.0 - cfg.superseded_penalty * sup_pen) * (1.0 - cfg.conflict_penalty * con_pen)
    return u - cfg.w_superseded * sup_pen - cfg.w_conflict * con_pen


def _judged_confidence(j: HeadJudgments, heads: tuple[HeadName, ...]) -> float:
    confs = []
    if "rel" in heads:
        confs.append(j.relevance_confidence)
    if "util" in heads:
        confs.append(j.utility_confidence)
    return min(confs) if confs else 0.0


def assign_label(
    j: HeadJudgments,
    value: float,
    cfg: PolicyConfig,
    heads: tuple[HeadName, ...] = ("rel", "util", "sup", "con"),
) -> Label:
    """Label assignment.

    Head flags gate on their OWN probability: superseded/conflict are Noul
    probabilities, so STALE/CONFLICT fire when the head crosses its threshold
    (dominant head wins when both fire) - never on Score-head confidence.
    Score-head confidence gates only the VALUE labels: below the gate an item
    becomes UNCERTAIN instead of acting on its value. Confidence never scales
    the value itself.
    """
    sup_fired = "sup" in heads and j.superseded >= cfg.stale_threshold
    con_fired = "con" in heads and j.conflict >= cfg.conflict_threshold
    if sup_fired and con_fired:
        return Label.CONFLICT if j.conflict >= j.superseded else Label.STALE
    if con_fired:
        return Label.CONFLICT
    if sup_fired:
        return Label.STALE
    if _judged_confidence(j, heads) < cfg.confidence_gate:
        return Label.UNCERTAIN
    if value >= cfg.use_threshold:
        return Label.USE
    if value < cfg.drop_threshold:
        return Label.DROP
    return Label.KEEP


def apply_policy(
    candidates: list[Candidate],
    judgments: dict[str, HeadJudgments],
    cfg: PolicyConfig | None = None,
    heads: tuple[HeadName, ...] = ("rel", "util", "sup", "con"),
) -> list[RankedItem]:
    """Rank candidates best-first. Pure function - no I/O, no model calls.

    Deterministic ``source_priority`` metadata is a tie-breaker (after value
    and confidence, before retrieval order): never a question to Jev.
    """
    cfg = cfg or PolicyConfig()
    items: list[RankedItem] = []
    for cand in candidates:
        j = judgments[cand.id]
        value = policy_value(j, cfg, heads)
        label = assign_label(j, value, cfg, heads)
        items.append(RankedItem(candidate=cand, judgments=j, value=value, label=label, rank=0))
    # Stable sort: value desc, confident-first, source_priority desc, retrieval_rank, id.
    items.sort(
        key=lambda it: (
            -it.value,
            0 if it.label is not Label.UNCERTAIN else 1,
            -cfg.source_priority_of(it.candidate.source),
            it.candidate.retrieval_rank if it.candidate.retrieval_rank is not None else 10**9,
            it.candidate.id,
        )
    )
    for rank, item in enumerate(items):
        item.rank = rank
    return items
