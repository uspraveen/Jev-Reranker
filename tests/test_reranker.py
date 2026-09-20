"""Reranker mode tests: three distinct behaviors, budget, dedup, async, result objects."""

import asyncio
from typing import Any

from jev_reranker.judges import JudgeResult, OfflineJudge
from jev_reranker.models import Candidate, Label, MemoryItem, RerankResult
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


def test_relevance_mode_value_is_normalized_relevance() -> None:
    rr, _ = _rr()
    res = rr.rerank_texts("deploy payments?", ["deploy payments service now"], mode="relevance")
    top = res.items[0]
    assert 0.0 < top.value <= 1.0  # relevance/3


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


def test_context_mode_budget_respected_and_dedup_suppresses() -> None:
    rr, _ = _rr()
    cands = [
        Candidate(id="gold", text="Refunds settle within 5 business days.", retrieval_rank=0),
        Candidate(id="dup", text="Refunds settle within five business days, confirmed today.", retrieval_rank=1),
        Candidate(id="other", text="The support team email is support@example.com for account help.", retrieval_rank=2),
    ]
    sel = rr.select("how long do refunds take to settle?", cands, budget_tokens=200)
    assert sel.total_tokens <= 200
    assert all(it.label is not Label.DROP for it in sel.selected)
    # The near-duplicate was suppressed by the embedding pass.
    assert [it.candidate.id for it in sel.suppressed_near_duplicates] == ["dup"]


def test_dedup_keeps_highest_value_member_of_cluster() -> None:
    """A stale duplicate ranked first must not win the cluster over the current fact."""
    from jev_reranker.models import HeadJudgments, RankedItem

    def item(cid: str, text: str, rank: int, value: float, label: Label) -> RankedItem:
        return RankedItem(
            candidate=Candidate(id=cid, text=text, retrieval_rank=rank),
            judgments=HeadJudgments(
                relevance=3.0, relevance_confidence=0.9, utility=3.0, utility_confidence=0.9,
                superseded=0.0, conflict=0.0,
            ),
            value=value,
            label=label,
            rank=rank,
        )

    items = [
        item("old", "Refunds settle within 5 business days.", 0, 0.30, Label.STALE),
        item("gold", "Refunds settle within five business days.", 1, 0.80, Label.USE),
    ]
    sel = select_for_budget(items, budget_tokens=100)
    assert [it.candidate.id for it in sel.selected] == ["gold"]
    assert [it.candidate.id for it in sel.suppressed_near_duplicates] == ["old"]


def test_select_for_budget_greedy_value_per_token() -> None:
    from jev_reranker.models import HeadJudgments, RankedItem

    def item(cid: str, value: float, words: int) -> RankedItem:
        text = ("w " * words).strip()
        return RankedItem(
            candidate=Candidate(id=cid, text=text, retrieval_rank=0),
            judgments=HeadJudgments(
                relevance=3.0,
                relevance_confidence=0.9,
                utility=3.0,
                utility_confidence=0.9,
                superseded=0.0,
                conflict=0.0,
            ),
            value=value,
            label=Label.KEEP,
            rank=0,
        )

    sel = select_for_budget([item("big", 0.9, 400), item("small", 0.5, 10)], budget_tokens=100, dedup_threshold=-1.0)
    assert [it.candidate.id for it in sel.selected] == ["small"]


def test_result_object_rich_fields() -> None:
    rr, _ = _rr()
    cands = [
        Candidate(id="a", text="deploy payments service to prod tonight", retrieval_rank=0),
        Candidate(id="b", text="OUTDATED deprecated old version: deploy payments at noon", retrieval_rank=1),
        Candidate(id="c", text="lunch menu pasta", retrieval_rank=2),
    ]
    res = rr.rerank("how do I deploy payments?", cands, mode="memory")
    assert isinstance(res, RerankResult)
    assert res.latency_ms is not None and res.latency_ms >= 0
    assert res.usage.get("input_tokens", 0) > 0
    # Disjoint splits covering all items.
    n = len(res.items)
    assert len(res.selected) + len(res.rejected) + len(res.uncertain) + len(res.conflicts) == n
    # decisions alias + labels list + documents
    assert res.items[0].decisions is res.items[0].judgments
    assert len(res.labels) == n
    docs = res.documents
    assert docs[0].id == res.items[0].candidate.id
    assert docs[0].decisions.relevance == res.items[0].judgments.relevance


def test_memory_item_metadata_first_class() -> None:
    rr, _ = _rr()
    mem = MemoryItem(
        id="m1",
        text="Refunds settle within 5 business days.",
        retrieval_rank=0,
        source="zendesk",
        created_at="2026-09-01T10:00:00Z",
        importance=0.9,
        entity_ids=["refunds", "policy"],
        session_id="s-123",
    )
    res = rr.rerank("refunds?", [mem], mode="memory")
    assert res.items[0].candidate.source == "zendesk"
    doc = res.documents[0]
    assert doc.metadata["source"] == "zendesk"


def test_source_priority_changes_order_without_changing_values() -> None:
    from jev_reranker.models import PolicyConfig

    judge = OfflineJudge()
    rr = JevReranker(judge=judge, policy=PolicyConfig(source_priority={"runbook": 0.99}, default_source_priority=0.1))
    cands = [
        Candidate(id="a", text="deploy payments with kubectl rollout", retrieval_rank=1, source="chatter"),
        Candidate(id="b", text="deploy payments with kubectl rollout", retrieval_rank=1, source="runbook"),
    ]
    res = rr.rerank("how do I deploy payments?", cands, mode="memory")
    assert res.items[0].candidate.id == "b"
    assert res.items[0].value == res.items[1].value


def test_async_parity_sync_judge() -> None:
    rr, _ = _rr()
    cands = [Candidate(id="a", text="refunds settle within 5 business days", retrieval_rank=0)]
    res = asyncio.run(rr.arerank("refunds?", cands, mode="relevance"))
    assert res.items[0].candidate.id == "a"


def test_async_native_path_with_async_judge() -> None:
    calls = 0

    class AsyncFake:
        model_name = "async-fake"

        async def judge(
            self,
            state: dict[str, Any],
            questions: dict[str, dict[str, object]],
            query: str,
            candidates: list[Candidate],
        ) -> tuple[JudgeResult, dict[str, int]]:
            nonlocal calls
            calls += 1
            answers: JudgeResult = JudgeResult()
            for key in questions:
                kind = key.split("__")[0]
                if kind in ("rel", "util"):
                    answers[key] = {"score": 3.0, "confidence": 0.95}
                else:
                    answers[key] = {"noul": 0.0}
            return answers, {"input_tokens": 10, "output_tokens": 2}

    rr = JevReranker(judge=AsyncFake())
    cands = [Candidate(id=f"c{i}", text=f"deploy payments step {i}", retrieval_rank=i) for i in range(3)]
    res = asyncio.run(rr.arerank("deploy payments?", cands, mode="memory"))
    assert calls == 1
    assert res.latency_ms is not None and res.latency_ms >= 0
    assert res.items[0].value > 0.5  # 0.55*1 + 0.45*1 = 1.0 with no penalties
