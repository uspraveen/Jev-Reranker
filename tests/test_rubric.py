"""Rubric tests: byte-identical default preset, resolution, custom flow, cache-key identity."""

import pytest

from jev_reranker.cache import cache_key
from jev_reranker.policy import QUESTION_SCHEMA_VERSION, build_questions
from jev_reranker.rubric import (
    BUILTIN_RUBRICS,
    NoulHeadRubric,
    Rubric,
    ScoreHeadRubric,
    resolve_rubric,
)


def test_default_rubric_matches_original_hardcoded_questions() -> None:
    """agent_memory must reproduce the original hard-coded questions verbatim."""
    q = build_questions(["a"])
    assert q["rel__a"] == {
        "type": "score",
        "instructions": "How relevant is candidate [a] to the query?",
        "criteria": [
            "irrelevant to the query",
            "partially relevant background",
            "mostly relevant, on-topic",
            "directly answers the query",
        ],
    }
    assert q["util__a"] == {
        "type": "score",
        "instructions": "How actionable/useful is candidate [a] for acting on the query?",
        "criteria": [
            "no actionable content",
            "useful background only",
            "mostly actionable",
            "directly usable to act",
        ],
    }
    assert q["sup__a"] == {
        "type": "noul",
        "instructions": (
            "Is the fact in candidate [a] superseded — that is, replaced or invalidated "
            "by newer information about the same fact (in the other candidates or the query)?"
        ),
        "criteria": {
            "true": "A newer piece of information replaces or invalidates this same fact",
            "false": "This fact still stands on its own (irrelevance alone does not make it superseded)",
        },
    }
    assert q["con__a"] == {
        "type": "noul",
        "instructions": "Does candidate [a] conflict with the query or the other candidates?",
        "criteria": {
            "true": "It asserts something contradicted elsewhere",
            "false": "It is consistent with the rest",
        },
    }


def test_resolve_rubric_preset_custom_and_error() -> None:
    assert resolve_rubric(None).name == "agent_memory"
    assert resolve_rubric("code_search").name == "code_search"
    assert resolve_rubric(BUILTIN_RUBRICS["generic_retrieval"]).name == "generic_retrieval"
    with pytest.raises(ValueError, match="unknown rubric"):
        resolve_rubric("does-not-exist")


def test_builtin_rubric_ids() -> None:
    assert {r.id for r in BUILTIN_RUBRICS.values()} == {
        "agent_memory@v1",
        "generic_retrieval@v1",
        "code_search@v1",
    }


def test_custom_rubric_flows_into_questions() -> None:
    custom = Rubric(
        name="legal",
        rel=ScoreHeadRubric(
            instructions="Is candidate [cid] controlling authority for the query?",
            criteria=["off point", "background", "persuasive", "controlling"],
        ),
        util=ScoreHeadRubric(
            instructions="How actionable is candidate [cid]?",
            criteria=["none", "some", "most", "fully"],
        ),
        sup=NoulHeadRubric(instructions="Is candidate [cid] superseded?", criteria={"true": "yes", "false": "no"}),
        con=NoulHeadRubric(instructions="Does candidate [cid] conflict?", criteria={"true": "yes", "false": "no"}),
    )
    q = build_questions(["x"], rubric=custom)
    assert q["rel__x"]["instructions"] == "Is candidate [x] controlling authority for the query?"
    assert q["rel__x"]["criteria"] == ["off point", "background", "persuasive", "controlling"]
    assert q["util__x"]["instructions"] == "How actionable is candidate [x]?"


def test_rubric_id_in_cache_key() -> None:
    base = cache_key("q", [("a", "t")], "m", "v1", QUESTION_SCHEMA_VERSION)
    assert cache_key("q", [("a", "t")], "m", "v1", QUESTION_SCHEMA_VERSION, rubric_id="agent_memory@v1") == base
    assert cache_key("q", [("a", "t")], "m", "v1", QUESTION_SCHEMA_VERSION, rubric_id="code_search@v1") != base
