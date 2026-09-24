"""Judge backends: the thing that turns (query, candidates) into head judgments.

- ``LiveJevJudge``: the real path - one ``TypeSafeClient.system_one()`` call
  per rerank with all active heads batched. Requires ``TYPESAFE_API_KEY``.
- ``AsyncLiveJevJudge``: identical but on the SDK's native async client
  (``AsyncTypeSafeClient``) - one async call per rerank, no executor.
- ``OfflineJudge``: deterministic lexical judge with identical answer shapes,
  for tests / synthetic eval / demos without a key. Clearly labeled everywhere
  it is used; never presented as Jev output.

All live backends retry with exponential backoff + jitter on
408/429/5xx/529 (honoring Retry-After) via the SDK ``RetryPolicy``.
"""

from __future__ import annotations

import inspect
import os
import re
from typing import Any, Protocol

from jev_reranker.models import Candidate

_WORD = re.compile(r"[a-z0-9]+")

RETRY_HTTP_STATUSES = frozenset({408, 429, 529, 500, 502, 503, 504})


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class JudgeResult(dict[str, Any]):
    """Mapping question-key -> raw answer dict (score/confidence or noul)."""


class Judge(Protocol):
    """Any backend that answers a batched question map in ONE call."""

    @property
    def model_name(self) -> str:
        """Backend model identifier (used in traces)."""
        ...

    def judge(
        self,
        state: dict[str, Any],
        questions: dict[str, dict[str, object]],
        query: str,
        candidates: list[Candidate],
    ) -> tuple[JudgeResult, dict[str, int]]:
        """Return (answers, usage). Must issue at most one remote call."""
        ...


class AsyncJudge(Protocol):
    """Async twin of :class:`Judge` - one awaitable remote call per rerank."""

    @property
    def model_name(self) -> str: ...

    async def judge(
        self,
        state: dict[str, Any],
        questions: dict[str, dict[str, object]],
        query: str,
        candidates: list[Candidate],
    ) -> tuple[JudgeResult, dict[str, int]]: ...


def is_async_judge(judge: Any) -> bool:
    return inspect.iscoroutinefunction(getattr(judge, "judge", None))


class OfflineJudge:
    """Deterministic lexical judge. Test/eval/demo use ONLY.

    relevance/utility from token overlap between query and candidate;
    superseded from explicit supersession cues + recency order;
    conflict from contradiction cues vs. the query/other candidates.
    Score heads answer on the same 0-3 rubric as live Jev.
    """

    model_name = "offline-judge-v2"

    SUPERSEDE_CUES = ("outdated", "deprecated", "superseded", "old version", "no longer", "replaced", "we plan to")
    CONFLICT_CUES = ("however", "but actually", "contradicts", "wrong", "incorrect", "never", "not true")

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.calls = 0  # counts judge invocations (assert == 1 per rerank in tests)

    def judge(
        self,
        state: dict[str, Any],
        questions: dict[str, dict[str, object]],
        query: str,
        candidates: list[Candidate],
    ) -> tuple[JudgeResult, dict[str, int]]:
        self.calls += 1
        q_toks = _tokens(query)
        answers = JudgeResult()
        # Recency order for superseded reasoning: later retrieval_rank == newer.
        order = {c.id: (c.retrieval_rank if c.retrieval_rank is not None else 0) for c in candidates}
        newest = max(order.values()) if order else 0
        by_id = {c.id: c for c in candidates}
        for key in questions:
            kind, _, cid = key.partition("__")
            cand = by_id[cid]
            c_toks = _tokens(cand.text)
            ov = _overlap(q_toks, c_toks)
            low = cand.text.lower()
            if kind == "rel":
                score = min(3.0, ov * 9.0)
                conf = 0.55 + min(0.4, ov * 2.0)
                answers[key] = {"score": score, "confidence": conf}
            elif kind == "util":
                score = min(3.0, ov * 7.5 + (0.6 if len(cand.text) > 80 else 0.0))
                conf = 0.55 + min(0.4, ov * 2.0)
                answers[key] = {"score": score, "confidence": conf}
            elif kind == "sup":
                cue = 0.75 if any(cue in low for cue in self.SUPERSEDE_CUES) else 0.0
                age = 0.0 if order[cid] >= newest else 0.25
                answers[key] = {"noul": min(1.0, cue + age)}
            elif kind == "con":
                cue = 0.8 if any(cue in low for cue in self.CONFLICT_CUES) else 0.05
                answers[key] = {"noul": cue}
        usage = {"input_tokens": sum(len(c.text) // 4 for c in candidates) + len(query) // 4, "output_tokens": 0}
        return answers, usage


def make_sdk_retry(max_retries: int = 4) -> Any:
    """SDK RetryPolicy: exponential backoff + jitter on 429/5xx, honor Retry-After."""
    from typesafe_sdk import RetryPolicy

    return RetryPolicy(
        max_retries=max_retries,
        backoff_initial=0.5,
        backoff_max=8.0,
        backoff_jitter=0.25,
        http_statuses=set(RETRY_HTTP_STATUSES),
        respect_retry_after=True,
    )


def to_sdk_questions(
    questions: dict[str, dict[str, object]],
    score_cls: Any,
    noul_cls: Any,
    noul_criteria_cls: Any,
) -> dict[str, Any]:
    """Translate policy question dicts into SDK Score/Noul objects."""
    sdk_q: dict[str, Any] = {}
    for key, q in questions.items():
        if q["type"] == "score":
            raw_levels = q.get("criteria")
            levels = [str(x) for x in raw_levels] if isinstance(raw_levels, list) else []
            sdk_q[key] = score_cls(instructions=str(q["instructions"]), criteria=levels)
        else:
            raw_crit = q.get("criteria")
            crit = dict(raw_crit) if isinstance(raw_crit, dict) else {}
            sdk_q[key] = noul_cls(
                instructions=str(q["instructions"]),
                criteria=noul_criteria_cls(true=crit.get("true"), false=crit.get("false")),
            )
    return sdk_q


def parse_sdk_answers(
    questions: dict[str, dict[str, object]],
    response: Any,
) -> tuple[JudgeResult, dict[str, int]]:
    answers = JudgeResult()
    for key, q in questions.items():
        ans: Any = response.answers[key]
        if q["type"] == "noul":
            answers[key] = {"noul": float(ans.noul)}
        else:
            answers[key] = {"score": float(ans.score), "confidence": float(ans.confidence)}
    usage = {
        "input_tokens": int(response.usage.input_tokens or 0),
        "output_tokens": int(response.usage.output_tokens or 0),
    }
    return answers, usage


class LiveJevJudge:
    """Real Jev backend via the official ``typesafe-sdk`` (one call per rerank)."""

    def __init__(
        self,
        model: str = "jev-latest",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 4,
    ) -> None:
        try:
            from typesafe_sdk import Noul, NoulCriteria, Score, TypeSafeClient
        except ImportError as exc:
            raise RuntimeError("typesafe-sdk is required for LiveJevJudge: pip install typesafe-sdk") from exc
        key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise RuntimeError("TYPESAFE_API_KEY is not set; cannot create LiveJevJudge")
        # Mandatory backoff: exponential + jitter on 429/5xx (incl. 529 overload),
        # honoring Retry-After. TypeSafe is under heavy load; never hammer it.
        self._client = TypeSafeClient(
            api_key=key,
            base_url=base_url,
            timeout=timeout,
            retry=make_sdk_retry(max_retries),
        )
        self._score_cls = Score
        self._noul_cls = Noul
        self._noul_criteria_cls = NoulCriteria
        self._model = model
        self.calls = 0

    @property
    def model_name(self) -> str:
        return self._model

    def judge(
        self,
        state: dict[str, Any],
        questions: dict[str, dict[str, object]],
        query: str,
        candidates: list[Candidate],
    ) -> tuple[JudgeResult, dict[str, int]]:
        self.calls += 1
        response = self._client.system_one(
            state=state,
            questions=to_sdk_questions(questions, self._score_cls, self._noul_cls, self._noul_criteria_cls),
            model=self._model,
        )
        return parse_sdk_answers(questions, response)


class AsyncLiveJevJudge:
    """Native-async Jev backend on the SDK's ``AsyncTypeSafeClient``.

    One awaited ``system_one`` call per rerank - no thread executor.
    """

    def __init__(
        self,
        model: str = "jev-latest",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 4,
    ) -> None:
        try:
            from typesafe_sdk import AsyncTypeSafeClient, Noul, NoulCriteria, Score
        except ImportError as exc:
            raise RuntimeError("typesafe-sdk is required for AsyncLiveJevJudge: pip install typesafe-sdk") from exc
        key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise RuntimeError("TYPESAFE_API_KEY is not set; cannot create AsyncLiveJevJudge")
        self._client = AsyncTypeSafeClient(
            api_key=key,
            base_url=base_url,
            timeout=timeout,
            retry=make_sdk_retry(max_retries),
        )
        self._score_cls = Score
        self._noul_cls = Noul
        self._noul_criteria_cls = NoulCriteria
        self._model = model
        self.calls = 0

    @property
    def model_name(self) -> str:
        return self._model

    async def aclose(self) -> None:
        await self._client.aclose()

    async def judge(
        self,
        state: dict[str, Any],
        questions: dict[str, dict[str, object]],
        query: str,
        candidates: list[Candidate],
    ) -> tuple[JudgeResult, dict[str, int]]:
        self.calls += 1
        response = await self._client.system_one(
            state=state,
            questions=to_sdk_questions(questions, self._score_cls, self._noul_cls, self._noul_criteria_cls),
            model=self._model,
        )
        return parse_sdk_answers(questions, response)
