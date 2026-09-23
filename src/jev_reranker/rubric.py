"""Question rubrics: the text Jev is actually asked, per head.

The policy math is domain-generic, but the *questions* encode what a domain
means by "relevant", "usable", "superseded", "conflicting" — so they are
configurable. A :class:`Rubric` carries one question per head (``rel``/``util``
Score heads with 4-level 0-3 criteria, ``sup``/``con`` Noul heads with
true/false criteria). Instructions may contain the literal token ``[cid]``,
replaced with each candidate's id at question-build time.

Built-in presets: ``agent_memory`` (the default — byte-identical to the
originally hard-coded questions), ``generic_retrieval`` (documents/RAG), and
``code_search``. A fully custom rubric is one ``Rubric`` instance away:
``JevReranker(rubric=Rubric(...))``.

Rubric identity participates in the judgment cache key (``Rubric.id``):
swapping rubrics never serves stale judgments.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ScoreHeadRubric(BaseModel):
    """A Score-head question: instructions template + 4-level 0-3 rubric."""

    kind: Literal["score"] = "score"
    instructions: str = Field(description="Question text; may contain the literal token [cid]")
    criteria: list[str] = Field(min_length=2, description="Ordered levels, index == score level")


class NoulHeadRubric(BaseModel):
    """A Noul-head question: instructions template + true/false criteria."""

    kind: Literal["noul"] = "noul"
    instructions: str = Field(description="Question text; may contain the literal token [cid]")
    criteria: dict[str, str] = Field(description="Keys 'true' and 'false'")


class Rubric(BaseModel):
    """One domain's four question texts; ``id`` keys the judgment cache."""

    name: str
    version: str = "v1"
    rel: ScoreHeadRubric
    util: ScoreHeadRubric
    sup: NoulHeadRubric
    con: NoulHeadRubric

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"

    def head(self, head: str) -> ScoreHeadRubric | NoulHeadRubric:
        heads: dict[str, ScoreHeadRubric | NoulHeadRubric] = {
            "rel": self.rel,
            "util": self.util,
            "sup": self.sup,
            "con": self.con,
        }
        return heads[head]


AGENT_MEMORY = Rubric(
    name="agent_memory",
    rel=ScoreHeadRubric(
        instructions="How relevant is candidate [cid] to the query?",
        criteria=[
            "irrelevant to the query",
            "partially relevant background",
            "mostly relevant, on-topic",
            "directly answers the query",
        ],
    ),
    util=ScoreHeadRubric(
        instructions="How actionable/useful is candidate [cid] for acting on the query?",
        criteria=[
            "no actionable content",
            "useful background only",
            "mostly actionable",
            "directly usable to act",
        ],
    ),
    sup=NoulHeadRubric(
        instructions=(
            "Is the fact in candidate [cid] superseded — that is, replaced or invalidated "
            "by newer information about the same fact (in the other candidates or the query)?"
        ),
        criteria={
            "true": "A newer piece of information replaces or invalidates this same fact",
            "false": "This fact still stands on its own (irrelevance alone does not make it superseded)",
        },
    ),
    con=NoulHeadRubric(
        instructions="Does candidate [cid] conflict with the query or the other candidates?",
        criteria={
            "true": "It asserts something contradicted elsewhere",
            "false": "It is consistent with the rest",
        },
    ),
)

GENERIC_RETRIEVAL = Rubric(
    name="generic_retrieval",
    rel=ScoreHeadRubric(
        instructions="How relevant is candidate [cid] to the query?",
        criteria=[
            "irrelevant to the query",
            "tangentially related background",
            "mostly on-topic for the query",
            "directly answers the query",
        ],
    ),
    util=ScoreHeadRubric(
        instructions="How usable is the content of candidate [cid] for answering the query?",
        criteria=[
            "no usable content for the query",
            "background context only",
            "substantially usable content",
            "directly usable answer material",
        ],
    ),
    sup=NoulHeadRubric(
        instructions=(
            "Is the content in candidate [cid] superseded — replaced or invalidated by newer "
            "information about the same subject (in the other candidates or the query)?"
        ),
        criteria={
            "true": "Newer information replaces or invalidates this content",
            "false": "This content still stands on its own (irrelevance alone does not make it superseded)",
        },
    ),
    con=NoulHeadRubric(
        instructions="Does candidate [cid] contradict the query or the other candidates?",
        criteria={
            "true": "It asserts something contradicted elsewhere",
            "false": "It is consistent with the rest",
        },
    ),
)

CODE_SEARCH = Rubric(
    name="code_search",
    rel=ScoreHeadRubric(
        instructions="How relevant is candidate [cid] to the code search query?",
        criteria=[
            "unrelated to the query",
            "touches related APIs but not the ask",
            "relevant code, partially matching",
            "exactly the code the query asks for",
        ],
    ),
    util=ScoreHeadRubric(
        instructions="How usable is candidate [cid] as-is for the task in the query?",
        criteria=[
            "not usable for the task",
            "needs substantial rework",
            "needs minor adaptation",
            "drop-in usable",
        ],
    ),
    sup=NoulHeadRubric(
        instructions=(
            "Is the code in candidate [cid] superseded — a deprecated or replaced API/version "
            "made obsolete by newer information (in the other candidates or the query)?"
        ),
        criteria={
            "true": "A newer or non-deprecated alternative replaces this code",
            "false": "This code still stands (irrelevance alone does not make it superseded)",
        },
    ),
    con=NoulHeadRubric(
        instructions=(
            "Does candidate [cid] conflict with the query requirements or the other candidates "
            "(e.g., incompatible assumptions, contradicting behavior)?"
        ),
        criteria={
            "true": "It asserts or assumes something contradicted elsewhere",
            "false": "It is consistent with the rest",
        },
    ),
)

BUILTIN_RUBRICS: dict[str, Rubric] = {
    AGENT_MEMORY.name: AGENT_MEMORY,
    GENERIC_RETRIEVAL.name: GENERIC_RETRIEVAL,
    CODE_SEARCH.name: CODE_SEARCH,
}

DEFAULT_RUBRIC = AGENT_MEMORY
DEFAULT_RUBRIC_ID = AGENT_MEMORY.id


def resolve_rubric(rubric: str | Rubric | None) -> Rubric:
    """``None``/preset name -> built-in (default ``agent_memory``); ``Rubric`` -> itself."""
    if rubric is None:
        return DEFAULT_RUBRIC
    if isinstance(rubric, Rubric):
        return rubric
    try:
        return BUILTIN_RUBRICS[rubric]
    except KeyError:
        raise ValueError(f"unknown rubric {rubric!r}; built-ins: {sorted(BUILTIN_RUBRICS)}") from None
