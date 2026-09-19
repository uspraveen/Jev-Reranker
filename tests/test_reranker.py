"""Reranker mode tests + cache + budget selection."""

from jev_reranker.judges import OfflineJudge
from jev_reranker.models import Candidate, Label
from jev_reranker.reranker import JevReranker, select_for_budget


def _rr() -> tuple[JevReranker, OfflineJudge]:
    judge = OfflineJudge()
    return JevReranker(judge=judge), judge


def test_relevance_mode_top_k() -> None:
    rr, judge = _rr()
    texts = ["deploy payments service to prod tonight", "lunch menu", "payments deploy runbook kubectl"]
    res = rr.rerank_texts("how do I deploy payments?", texts, mode="relevance", top_k=2)
    assert len(res.items) == 2
    assert judge.calls == 1
    assert res.mode == "relevance"
    assert res.request_id.startswith("jr_")


def test_memory_mode_surfaces_stale() -> None:
    rr, _ = _rr()
    cands = [
        Candidate(id="new", text="refunds settle within 5 business days (updated 2026-09-10)", retrieval_rank=1),
        Candidate(id="old", text="OUTDATED deprecated old version: refunds settle in 30 days", retrieval_rank=0),
    ]
    res = rr.rerank("how long do refunds take?", cands, mode="memory")
    labels = {it.candidate.id: it.label for it in res.items}
    assert labels["old"] is Label.STALE
    assert res.items[0].candidate.id == "new"


def test_memory_mode_surfaces_conflict() -> None:
    rr, _ = _rr()
    cands = [
        Candidate(id="a", text="search API allows 120 requests per minute", retrieval_rank=0),
        Candidate(
            id="b", text="However, that is wrong: search API never allows more than 5 requests", retrieval_rank=1
        ),
    ]
    res = rr.rerank("search API rate limit?", cands, mode="memory")
    assert any(it.label is Label.CONFLICT for it in res.items)


def test_context_mode_budget_respected() -> None:
    rr, _ = _rr()
    cands = [
        Candidate(id=f"c{i}", text=f"deploy payments service detail number {i} " * 10, retrieval_rank=i)
        for i in range(6)
    ]
    sel = rr.select("deploy payments?", cands, budget_tokens=200)
    assert sel.total_tokens <= 200
    assert len(sel.selected) + len(sel.excluded) == 6
    assert all(it.label is not Label.DROP for it in sel.selected)


def test_async_parity() -> None:
    import asyncio

    rr, _ = _rr()
    cands = [Candidate(id="a", text="refunds settle within 5 business days", retrieval_rank=0)]
    res = asyncio.run(rr.arerank("refunds?", cands, mode="relevance"))
    assert res.items[0].candidate.id == "a"


def test_select_for_budget_greedy_value_per_token() -> None:
    from jev_reranker.models import HeadJudgments, RankedItem

    def item(cid: str, value: float, words: int) -> RankedItem:
        text = ("w " * words).strip()
        return RankedItem(
            candidate=Candidate(id=cid, text=text, retrieval_rank=0),
            judgments=HeadJudgments(
                relevance=1.0,
                relevance_confidence=0.9,
                utility=1.0,
                utility_confidence=0.9,
                superseded=0.0,
                conflict=0.0,
            ),
            value=value,
            label=Label.KEEP,
            rank=0,
        )

    sel = select_for_budget([item("big", 0.9, 400), item("small", 0.5, 10)], budget_tokens=100)
    assert [it.candidate.id for it in sel.selected] == ["small"]
