"""Policy unit tests: scoring math, labels, confidence gating."""

from jev_reranker.models import Candidate, HeadJudgments, Label, PolicyConfig
from jev_reranker.policy import apply_policy, policy_value


def _j(**kw: object) -> HeadJudgments:
    base: dict[str, object] = {
        "relevance": 1.6,
        "relevance_confidence": 0.9,
        "utility": 1.4,
        "utility_confidence": 0.9,
        "superseded": 0.0,
        "conflict": 0.0,
    }
    base.update(kw)
    return HeadJudgments(**base)  # type: ignore[arg-type]


def _c(cid: str, rank: int = 0) -> Candidate:
    return Candidate(id=cid, text=f"text {cid}", retrieval_rank=rank)


def test_value_weights_scores_not_confidence() -> None:
    cfg = PolicyConfig()
    hi = _j(relevance=2.0, utility=2.0)
    lo = _j(relevance=0.0, utility=0.0)
    assert policy_value(hi, cfg) > policy_value(lo, cfg)
    # Confidence must not change the value.
    unconf = _j(relevance=2.0, utility=2.0, relevance_confidence=0.1, utility_confidence=0.1)
    assert policy_value(unconf, cfg) == policy_value(hi, cfg)


def test_penalties_superseded_conflict() -> None:
    cfg = PolicyConfig()
    clean = _j()
    stale = _j(superseded=1.0)
    conf = _j(conflict=1.0)
    assert policy_value(stale, cfg) < policy_value(clean, cfg)
    assert policy_value(conf, cfg) < policy_value(clean, cfg)


def test_labels_use_keep_drop() -> None:
    cfg = PolicyConfig()
    items = apply_policy(
        [_c("a"), _c("b"), _c("c")],
        {"a": _j(relevance=2.0, utility=2.0), "b": _j(relevance=1.0, utility=1.0), "c": _j(relevance=0.0, utility=0.0)},
        cfg,
    )
    assert items[0].candidate.id == "a" and items[0].label is Label.USE
    assert items[-1].candidate.id == "c" and items[-1].label is Label.DROP


def test_confidence_gates_to_uncertain() -> None:
    cfg = PolicyConfig()
    items = apply_policy([_c("a")], {"a": _j(relevance=2.0, utility=2.0, relevance_confidence=0.1)}, cfg)
    assert items[0].label is Label.UNCERTAIN


def test_stale_conflict_labels_beat_value() -> None:
    cfg = PolicyConfig()
    items = apply_policy(
        [_c("a")],
        {"a": _j(relevance=2.0, utility=2.0, superseded=0.9, conflict=0.9)},
        cfg,
    )
    # Conflict takes precedence over stale.
    assert items[0].label is Label.CONFLICT


def test_ranks_sequential_best_first() -> None:
    cfg = PolicyConfig()
    cands = [_c(f"c{i}", rank=5 - i) for i in range(3)]
    judgments = {c.id: _j(relevance=float(i), utility=1.0) for i, c in enumerate(cands)}
    items = apply_policy(cands, judgments, cfg)
    assert [it.rank for it in items] == [0, 1, 2]
    assert items[0].value >= items[1].value >= items[2].value
