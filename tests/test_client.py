"""Client tests: ONE-call rule, modes/head subsets, caching, limits, telemetry, fallback paths."""

from typing import Any

from jev_reranker.cache import JudgmentCache, cache_key
from jev_reranker.client import JevClient
from jev_reranker.judges import JudgeResult, OfflineJudge
from jev_reranker.models import Candidate, Label, RankedItem


class RecordingJudge:
    """OfflineJudge wrapper that records the question map it is handed."""

    model_name = "offline-judge-v2"

    def __init__(self) -> None:
        self.inner = OfflineJudge()
        self.calls = 0
        self.seen_questions: list[dict[str, dict[str, object]]] = []

    def judge(
        self,
        state: dict[str, Any],
        questions: dict[str, dict[str, object]],
        query: str,
        candidates: list[Candidate],
    ) -> tuple[JudgeResult, dict[str, int]]:
        self.calls += 1
        self.seen_questions.append(dict(questions))
        return self.inner.judge(state, questions, query, candidates)


def _cands(n: int = 4) -> list[Candidate]:
    topics = ["deploy payments service prod", "oncall database cluster", "rate limit search api", "refund settle days"]
    return [
        Candidate(id=f"c{i}", text=f"note {topics[i % len(topics)]} detail {i}", retrieval_rank=i) for i in range(n)
    ]


def test_one_call_per_rerank() -> None:
    judge = OfflineJudge()
    client = JevClient(judge=judge)
    items, meta = client.judge_once("how do refunds settle?", _cands(8), mode="memory")
    assert judge.calls == 1
    assert client.calls_made == 1
    assert len(items) == 8
    assert meta["fallback_used"] is False


def test_relevance_mode_judges_relevance_head_only() -> None:
    judge = RecordingJudge()
    client = JevClient(judge=judge)
    items, _ = client.judge_once("q", _cands(3), mode="relevance")
    assert judge.calls == 1
    keys = set(judge.seen_questions[0])
    assert keys == {"rel__c0", "rel__c1", "rel__c2"}
    assert all(it.judgments.superseded == 0.0 and it.judgments.conflict == 0.0 for it in items)
    assert all(it.label not in (Label.STALE, Label.CONFLICT) for it in items)


def test_memory_mode_judges_all_four_heads() -> None:
    judge = RecordingJudge()
    client = JevClient(judge=judge)
    client.judge_once("q", _cands(2), mode="memory")
    keys = set(judge.seen_questions[0])
    assert keys == {"rel__c0", "util__c0", "sup__c0", "con__c0", "rel__c1", "util__c1", "sup__c1", "con__c1"}


def test_context_mode_judges_all_four_heads() -> None:
    judge = RecordingJudge()
    client = JevClient(judge=judge)
    client.judge_once("q", _cands(2), mode="context")
    keys = set(judge.seen_questions[0])
    assert "sup__c0" in keys and "con__c1" in keys and "rel__c0" in keys and "util__c1" in keys


def test_cache_hit_makes_no_second_call() -> None:
    judge = OfflineJudge()
    cache = JudgmentCache()
    client = JevClient(judge=judge, cache=cache)
    cands = _cands()
    client.judge_once("query cache me", cands, mode="memory")
    items, meta = client.judge_once("query cache me", cands, mode="memory")
    assert meta["cached"] is True
    assert judge.calls == 1
    assert client.cache_hits == 1
    assert len(items) == 4


def test_cache_key_changes_with_policy_version_and_heads() -> None:
    k1 = cache_key("q", [("a", "t")], "m", "v1", "v1")
    k2 = cache_key("q", [("a", "t")], "m", "v2", "v1")
    k3 = cache_key("q", [("a", "t")], "m", "v1", "v1", heads=("rel",))
    assert k1 != k2
    assert k1 != k3


def test_candidate_limit_enforced() -> None:
    judge = OfflineJudge()
    client = JevClient(judge=judge, max_candidates=5)
    items, _ = client.judge_once(
        "q", _cands(0) + [Candidate(id=f"x{i}", text="t", retrieval_rank=i) for i in range(20)], mode="memory"
    )
    assert len(items) == 5
    assert judge.calls == 1


def test_token_based_input_limit_truncates_and_telemetrizes() -> None:
    events: list[dict[str, Any]] = []
    judge = OfflineJudge()
    client = JevClient(judge=judge, input_token_limit=400, on_event=events.append)
    cands = [Candidate(id=f"x{i}", text="word " * 600, retrieval_rank=i) for i in range(6)]
    items, meta = client.judge_once("q", cands, mode="memory")
    assert len(items) < 6  # token budget dropped some candidates
    assert "dropped_by_token_budget" in meta
    assert any(e["event"] == "input_truncated" for e in events)
    assert judge.calls == 1  # still ONE call


def test_telemetry_events_fire() -> None:
    events: list[dict[str, Any]] = []
    judge = OfflineJudge()
    client = JevClient(judge=judge, cache=JudgmentCache(), on_event=events.append)
    cands = _cands(2)
    client.judge_once("telemetry q", cands, mode="memory")
    client.judge_once("telemetry q", cands, mode="memory")
    kinds = [e["event"] for e in events]
    assert "judge_call" in kinds and "cache_hit" in kinds
    call_event = next(e for e in events if e["event"] == "judge_call")
    assert call_event["n_questions"] == 8 and call_event["mode"] == "memory"


def test_fallback_retrieval_order_on_judge_failure() -> None:
    class Boom:
        model_name = "boom"

        def judge(
            self,
            state: dict[str, Any],
            questions: dict[str, dict[str, object]],
            query: str,
            candidates: list[Candidate],
        ) -> tuple[JudgeResult, dict[str, int]]:
            raise ConnectionError("typesafe overloaded")

    events: list[dict[str, Any]] = []
    client = JevClient(judge=Boom(), on_event=events.append)
    cands = _cands(3)
    items, meta = client.judge_once("q", cands, mode="memory")
    assert meta["fallback_used"] is True
    assert [it.candidate.id for it in items] == ["c0", "c1", "c2"]
    assert all(it.label is Label.UNCERTAIN for it in items)
    assert any(e["event"] == "fallback" for e in events)


def test_custom_callable_fallback() -> None:
    def fb(query: str, cands: list[Candidate]) -> list[RankedItem]:
        from jev_reranker.client import retrieval_order_fallback

        return list(reversed(retrieval_order_fallback(query, cands)))

    class Boom:
        model_name = "boom"

        def judge(
            self,
            state: dict[str, Any],
            questions: dict[str, dict[str, object]],
            query: str,
            candidates: list[Candidate],
        ) -> tuple[JudgeResult, dict[str, int]]:
            raise TimeoutError("t/o")

    client = JevClient(judge=Boom(), fallback=fb)
    items, meta = client.judge_once("q", _cands(3), mode="memory")
    assert meta["fallback_used"] is True
    assert [it.candidate.id for it in items] == ["c2", "c1", "c0"]


def test_empty_candidates_no_call() -> None:
    judge = OfflineJudge()
    client = JevClient(judge=judge)
    items, meta = client.judge_once("q", [], mode="memory")
    assert items == []
    assert judge.calls == 0
