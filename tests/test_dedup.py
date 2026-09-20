"""Dedup module tests: deterministic embeddings, cosine, suppression."""

from jev_reranker.dedup import DEFAULT_DUP_THRESHOLD, embed, similarity, suppress_near_duplicates
from jev_reranker.models import Candidate, HeadJudgments, Label, RankedItem


def _item(cid: str, text: str, rank: int = 0) -> RankedItem:
    return RankedItem(
        candidate=Candidate(id=cid, text=text, retrieval_rank=rank),
        judgments=HeadJudgments(
            relevance=3.0,
            relevance_confidence=0.9,
            utility=3.0,
            utility_confidence=0.9,
            superseded=0.0,
            conflict=0.0,
        ),
        value=0.9,
        label=Label.USE,
        rank=rank,
    )


def test_embed_deterministic_and_normalized() -> None:
    a1 = embed("refunds settle within five business days")
    a2 = embed("refunds settle within five business days")
    assert a1 == a2
    assert abs(sum(v * v for v in a1) - 1.0) < 1e-6


def test_similarity_high_for_paraphrase_low_for_unrelated() -> None:
    gold = "Refunds settle within 5 business days."
    para = "Refunds settle within five business days!"
    unrelated = "The quarterly all-hands meeting is on Thursday at noon."
    assert similarity(gold, para) >= DEFAULT_DUP_THRESHOLD
    assert similarity(gold, unrelated) < 0.5


def test_suppress_keeps_first_of_cluster_in_rank_order() -> None:
    items = [
        _item("gold", "Refunds settle within 5 business days.", 0),
        _item("dup", "Refunds settle within five business days.", 1),
        _item("other", "Deploy freeze every Friday 18:00 UTC.", 2),
    ]
    kept, suppressed = suppress_near_duplicates(items)
    assert [it.candidate.id for it in kept] == ["gold", "other"]
    assert [it.candidate.id for it in suppressed] == ["dup"]


def test_suppress_disabled_with_zero_threshold() -> None:
    items = [_item("a", "same"), _item("b", "same", 1)]
    kept, suppressed = suppress_near_duplicates(items, threshold=0.0)
    assert len(kept) == 1 and len(suppressed) == 1  # identical texts still suppress at 0.0
    kept2, suppressed2 = suppress_near_duplicates(items, threshold=-1.0)
    assert len(kept2) == 2 and not suppressed2
