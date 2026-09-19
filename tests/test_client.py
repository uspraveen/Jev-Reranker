"""Client tests: ONE-call rule, caching, limits, fallback paths."""

from jev_reranker.cache import JudgmentCache, cache_key
from jev_reranker.client import JevClient
from jev_reranker.judges import OfflineJudge
from jev_reranker.models import Candidate, Label


def _cands(n: int = 4) -> list[Candidate]:
    topics = ["deploy payments service prod", "oncall database cluster", "rate limit search api", "refund settle days"]
    return [
        Candidate(id=f"c{i}", text=f"note {topics[i % len(topics)]} detail {i}", retrieval_rank=i) for i in range(n)
    ]


def test_one_call_per_rerank() -> None:
    judge = OfflineJudge()
    client = JevClient(judge=judge)
    items, meta = client.judge_once("how do refunds settle?", _cands(8))
    assert judge.calls == 1
    assert client.calls_made == 1
    assert len(items) == 8
    assert meta["fallback_used"] is False


def test_cache_hit_makes_no_second_call() -> None:
    judge = OfflineJudge()
    cache = JudgmentCache()
    client = JevClient(judge=judge, cache=cache)
    cands = _cands()
    client.judge_once("query cache me", cands)
    items, meta = client.judge_once("query cache me", cands)
    assert meta["cached"] is True
    assert judge.calls == 1
    assert len(items) == 4


def test_cache_key_changes_with_policy_version() -> None:
    k1 = cache_key("q", [("a", "t")], "m", "v1", "v1")
    k2 = cache_key("q", [("a", "t")], "m", "v2", "v1")
    assert k1 != k2


def test_candidate_limit_enforced() -> None:
    judge = OfflineJudge()
    client = JevClient(judge=judge, max_candidates=5)
    items, _ = client.judge_once(
        "q", _cands(0) + [Candidate(id=f"x{i}", text="t", retrieval_rank=i) for i in range(20)]
    )
    assert len(items) == 5
    assert judge.calls == 1


def test_fallback_retrieval_order_on_judge_failure() -> None:
    class Boom:
        model_name = "boom"

        def judge(self, state: object, questions: object, query: str, candidates: list[Candidate]) -> object:
            raise ConnectionError("typesafe overloaded")

    client = JevClient(judge=Boom())  # type: ignore[arg-type]
    cands = _cands(3)
    items, meta = client.judge_once("q", cands)
    assert meta["fallback_used"] is True
    assert [it.candidate.id for it in items] == ["c0", "c1", "c2"]
    assert all(it.label is Label.UNCERTAIN for it in items)


def test_custom_callable_fallback() -> None:
    def fb(query: str, cands: list[Candidate]) -> list:
        from jev_reranker.client import retrieval_order_fallback

        return list(reversed(retrieval_order_fallback(query, cands)))

    class Boom:
        model_name = "boom"

        def judge(self, state: object, questions: object, query: str, candidates: list[Candidate]) -> object:
            raise TimeoutError("t/o")

    client = JevClient(judge=Boom(), fallback=fb)  # type: ignore[arg-type]
    items, meta = client.judge_once("q", _cands(3))
    assert meta["fallback_used"] is True
    assert [it.candidate.id for it in items] == ["c2", "c1", "c0"]


def test_empty_candidates_no_call() -> None:
    judge = OfflineJudge()
    client = JevClient(judge=judge)
    items, meta = client.judge_once("q", [])
    assert items == []
    assert judge.calls == 0
