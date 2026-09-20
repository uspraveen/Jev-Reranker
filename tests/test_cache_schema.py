"""Cache + question-schema tests."""

from pathlib import Path
from typing import cast

from jev_reranker.cache import JudgmentCache, cache_key
from jev_reranker.policy import QUESTION_SCHEMA_VERSION, build_questions


def test_build_questions_four_heads_per_candidate() -> None:
    q = build_questions(["a", "b"])
    assert len(q) == 8
    assert q["rel__a"]["type"] == "score"
    assert q["util__a"]["type"] == "score"
    assert q["sup__a"]["type"] == "noul"
    assert q["con__b"]["type"] == "noul"
    assert len(cast("list[str]", q["rel__a"]["criteria"]) or []) == 4  # 0-3 rubric: four levels


def test_cache_roundtrip_file(tmp_path: Path) -> None:
    p = tmp_path / "cache.json"
    c = JudgmentCache(p)
    c.set("k", {"a": 1})
    c2 = JudgmentCache(p)
    assert c2.get("k") == {"a": 1}


def test_cache_key_stable_and_sensitive() -> None:
    base = cache_key("q", [("a", "t1")], "m", "v1", QUESTION_SCHEMA_VERSION)
    assert cache_key("q", [("a", "t1")], "m", "v1", QUESTION_SCHEMA_VERSION) == base
    assert cache_key("q2", [("a", "t1")], "m", "v1", QUESTION_SCHEMA_VERSION) != base
    assert cache_key("q", [("a", "t2")], "m", "v1", QUESTION_SCHEMA_VERSION) != base
    assert cache_key("q", [("a", "t1")], "m", "v1", QUESTION_SCHEMA_VERSION, heads=("rel",)) != base
