"""Deterministic ranking policy.

Jev makes judgments (relevance Score 0-3, utility Score 0-3, superseded Noul,
conflict Noul). This module — pure Python, no model calls — turns those
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

POLICY_VERSION = "v2"
QUESTION_SCHEMA_VERSION = "v2"

# Question-schema: up to 4 heads per candidate, all asked in ONE
# /v1/systemone call. Score heads use the owner's 0-3 rubric (4 levels).
RELEVANCE_LEVELS = [
    "irrelevant to the query",
    "partially relevant background",
    "mostly relevant, on-topic",
    "directly answers the query",
]
UTILITY_LEVELS = [
    "no actionable content",
    "useful background only",
    "mostly actionable",
    "directly usable to act",
]

HEAD_KINDS: dict[str, str] = {"rel": "score", "util": "score", "sup": "noul", "con": "noul"}


def question_key(kind: str, candidate_id: str) -> str:
    """Question map key for head ``kind`` of ``candidate_id``."""
    return f"{kind}__{candidate_id}"


def build_questions(
    candidate_ids: list[str],
    heads: tuple[HeadName, ...] = ("rel", "util", "sup", "con"),
) -> dict[str, dict[str, object]]:
    """Build the (heads x N) question payload (JSON-serializable) for one batched call."""
    questions: dict[str, dict[str, object]] = {}
    for cid in candidate_ids:
        for head in heads:
            key = question_key(head, cid)
            if head == "rel":
                questions[key] = {
                    "type": "score",
                    "instructions": f"How relevant is candidate [{cid}] to the query?",
                    "criteria": RELEVANCE_LEVELS,
                }
            elif head == "util":
                questions[key] = {
                    "type": "score",
                    "instructions": f"How actionable/useful is candidate [{cid}] for acting on the query?",
                    "criteria": UTILITY_LEVELS,
                }
            elif head == "sup":
                questions[key] = {
                    "type": "noul",
                    "instructions": (
                        f"Is the fact in candidate [{cid}] superseded — that is, replaced or invalidated "
                        f"by newer information about the same fact (in the other candidates or the query)?"
                    ),
                    "criteria": {
                        "true": "A newer piece of information replaces or invalidates this same fact",
                        "false": "This fact still stands on its own (irrelevance alone does not make it superseded)",
                    },
                }
            else:  # con
                questions[key] = {
                    "type": "noul",
                    "instructions": f"Does candidate [{cid}] conflict with the query or the other candidates?",
                    "criteria": {
                        "true": "It asserts something contradicted elsewhere",
                        "false": "It is consistent with the rest",
                    },
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
    (dominant head wins when both fire) — never on Score-head confidence.
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
    """Rank candidates best-first. Pure function — no I/O, no model calls.

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
