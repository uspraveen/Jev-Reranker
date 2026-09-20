"""Policy unit tests: scoring math (both penalty forms), labels, confidence gating, source priority."""

from jev_reranker.models import Candidate, HeadJudgments, MemoryItem, PolicyConfig
from jev_reranker.policy import apply_policy, build_questions, policy_value


def _j(**kw: object) -> object:
    base: dict[str, object] = {
        "relevance": 2.4,  # 0-3 rubric
        "relevance_confidence": 0.9,
        "utility": 2.1,
        "utility_confidence": 0.9,
        "superseded": 0.0,
        "conflict": 0.0,
    }
    base.update(kw)
    return base


def _judged(**kw: object) -> HeadJudgments:
    return HeadJudgments(**_j(**kw))  # type: ignore[arg-type]


def _c(cid: str, rank: int = 0, source: str | None = None) -> Candidate:
    return Candidate(id=cid, text=f"text {cid}", retrieval_rank=rank, source=source)


def test_value_weights_scores_not_confidence() -> None:
    cfg = PolicyConfig()
    hi = _judged(relevance=3.0, utility=3.0)
    lo = _judged(relevance=0.0, utility=0.0)
    assert policy_value(hi, cfg) > policy_value(lo, cfg)
    # Confidence must not change the value.
    unconf = _judged(relevance=3.0, utility=3.0, relevance_confidence=0.1, utility_confidence=0.1)
    assert policy_value(unconf, cfg) == policy_value(hi, cfg)


def test_owner_formula_multiplicative_exact() -> None:
    """J = (0.55*R + 0.45*V) * (1-0.75*S) * (1-0.25*C), R/V normalized /3."""
    cfg = PolicyConfig()
    j = _judged(relevance=3.0, utility=3.0, superseded=0.8, conflict=0.4)
    expected = (0.55 * 1.0 + 0.45 * 1.0) * (1 - 0.75 * 0.8) * (1 - 0.25 * 0.4)
    assert abs(policy_value(j, cfg) - expected) < 1e-9


def test_additive_legacy_form_named_alternative() -> None:
    cfg = PolicyConfig(penalty_form="additive")
    j = _judged(relevance=3.0, utility=3.0, superseded=0.5, conflict=0.5)
    expected = 1.0 - 0.80 * 0.5 - 0.60 * 0.5
    assert abs(policy_value(j, cfg) - expected) < 1e-9


def test_penalties_superseded_conflict() -> None:
    cfg = PolicyConfig()
    clean = _judged()
    stale = _judged(superseded=1.0)
    conf = _judged(conflict=1.0)
    assert policy_value(stale, cfg) < policy_value(clean, cfg)
    assert policy_value(conf, cfg) < policy_value(clean, cfg)


def test_labels_use_keep_drop() -> None:
    cfg = PolicyConfig()
    items = apply_policy(
        [_c("a"), _c("b"), _c("c")],
        {
            "a": _judged(relevance=3.0, utility=3.0),
            "b": _judged(relevance=1.5, utility=1.5),
            "c": _judged(relevance=0.0, utility=0.0),
        },
        cfg,
    )
    assert items[0].candidate.id == "a" and items[0].label.value == "USE"
    assert items[-1].candidate.id == "c" and items[-1].label.value == "DROP"


def test_confidence_gates_to_uncertain() -> None:
    cfg = PolicyConfig()
    items = apply_policy([_c("a")], {"a": _judged(relevance=3.0, utility=3.0, relevance_confidence=0.1)}, cfg)
    assert items[0].label.value == "UNCERTAIN"


def test_stale_conflict_labels_beat_value() -> None:
    cfg = PolicyConfig()
    items = apply_policy(
        [_c("a")],
        {"a": _judged(relevance=3.0, utility=3.0, superseded=0.9, conflict=0.9)},
        cfg,
    )
    # Tie on threshold: the dominant head (conflict >= superseded) wins.
    assert items[0].label.value == "CONFLICT"


def test_dominant_head_wins_when_both_flags_fire() -> None:
    cfg = PolicyConfig()
    stale_dominant = _judged(superseded=0.95, conflict=0.7)
    conflict_dominant = _judged(superseded=0.7, conflict=0.95)
    assert apply_policy([_c("a")], {"a": stale_dominant}, cfg)[0].label.value == "STALE"
    assert apply_policy([_c("b")], {"b": conflict_dominant}, cfg)[0].label.value == "CONFLICT"


def test_head_flags_bypass_score_confidence_gate() -> None:
    """STALE/CONFLICT are Noul-probability flags; score confidence gates only value labels."""
    cfg = PolicyConfig()
    items = apply_policy(
        [_c("a")],
        {"a": _judged(relevance=3.0, utility=3.0, relevance_confidence=0.1, superseded=0.9)},
        cfg,
    )
    assert items[0].label.value == "STALE"  # not UNCERTAIN: the flag fired on its own head


def test_ranks_sequential_best_first() -> None:
    cfg = PolicyConfig()
    cands = [_c(f"c{i}", rank=5 - i) for i in range(3)]
    judgments = {c.id: _judged(relevance=float(i), utility=1.0) for i, c in enumerate(cands)}
    items = apply_policy(cands, judgments, cfg)
    assert [it.rank for it in items] == [0, 1, 2]
    assert items[0].value >= items[1].value >= items[2].value


def test_source_priority_breaks_value_ties() -> None:
    cfg = PolicyConfig(source_priority={"runbook": 0.9, "chatter": 0.1}, default_source_priority=0.5)
    a = MemoryItem(id="a", text="same text", retrieval_rank=0, source="chatter")
    b = MemoryItem(id="b", text="same text", retrieval_rank=1, source="runbook")
    judgments = {
        "a": _judged(relevance=3.0, utility=3.0),
        "b": _judged(relevance=3.0, utility=3.0),
    }
    items = apply_policy([a, b], judgments, cfg)
    assert items[0].candidate.id == "b"  # higher source priority wins the tie
    assert items[0].value == items[1].value  # ...without changing the values


def test_build_questions_head_subset() -> None:
    rel_only = build_questions(["a"], heads=("rel",))
    assert list(rel_only) == ["rel__a"]
    all_four = build_questions(["a"])
    assert sorted(all_four) == ["con__a", "rel__a", "sup__a", "util__a"]
