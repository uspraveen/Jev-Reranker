"""Baseline systems for MemoryBench-JR (no Jev calls; locally measured).

- ``retrieval_order``: rank candidates by their retrieval rank (the floor).
- ``embedding``: rank by cosine(hashed-embedding(query), embedding(candidate)).

Every number in the comparison table is produced by running these systems on
the same generated cases - nothing is copied from prior work.
"""

from __future__ import annotations

from benchmarks.memorybench_jr.generate import BenchCase
from jev_reranker.dedup import similarity


def rank_retrieval_order(case: BenchCase) -> list[str]:
    cands = sorted(case.candidates, key=lambda c: (c.retrieval_rank if c.retrieval_rank is not None else 10**9, c.id))
    return [c.id for c in cands]


def rank_embedding(case: BenchCase) -> list[str]:
    scored = [(similarity(case.query, c.text), -(c.retrieval_rank or 0), c.id) for c in case.candidates]
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [cid for _, _, cid in scored]


def recall_at_k(ranked: list[str], gold_id: str, k: int) -> bool:
    if not gold_id:
        return False
    return gold_id in ranked[:k]


def top1_is_use(labels: dict[str, str]) -> bool:
    """For answer_not_present: correct rejection = nothing fed as usable."""
    return all(v not in ("USE", "KEEP") for v in labels.values())


def candidate_texts(case: BenchCase) -> dict[str, str]:
    return {c.id: c.text for c in case.candidates}
