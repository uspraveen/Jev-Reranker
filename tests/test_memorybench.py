"""MemoryBench-JR unit tests: determinism, coverage, calibration math, oracle knapsack."""

from typing import cast

from benchmarks.memorybench_jr.calibration import brier_score, ece, threshold_table
from benchmarks.memorybench_jr.generate import CATEGORIES, generate_cases
from jev_reranker.models import RankedItem


def test_generator_deterministic_and_covered() -> None:
    a = generate_cases(500, seed=7)
    b = generate_cases(500, seed=7)
    assert [c.id for c in a] == [c.id for c in b]
    assert [(c.query, [x.text for x in c.candidates]) for c in a] == [
        (c.query, [x.text for x in c.candidates]) for c in b
    ]
    cats = {c.category for c in a}
    assert cats == set(CATEGORIES)
    assert len(a) == 500
    assert all(len(c.candidates) >= 4 for c in a)
    # answer_not_present cases have empty gold and empty relevant set
    anp = [c for c in a if c.category == "answer_not_present"]
    assert anp and all(c.gold_id == "" and not c.relevant_ids for c in anp)


def test_generator_varies_with_seed() -> None:
    a = generate_cases(50, seed=7)
    b = generate_cases(50, seed=8)
    assert any(x.query != y.query or [t.text for t in x.candidates] != [t.text for t in y.candidates]
               for x, y in zip(a, b, strict=True))


def test_brier_perfect_and_worst() -> None:
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0
    assert brier_score([0.0, 1.0], [1, 0]) == 1.0
    assert abs(brier_score([0.5, 0.5], [1, 0]) - 0.25) < 1e-9


def test_ece_perfectly_calibrated_zero() -> None:
    # Each bin's mean outcome exactly equals its mean predicted probability.
    probs = [0.2] * 10 + [0.5] * 10 + [0.8] * 10
    outcomes = [1] * 2 + [0] * 8 + [1] * 5 + [0] * 5 + [1] * 8 + [0] * 2
    assert ece(probs, outcomes) < 1e-9


def test_threshold_table_counts_sum() -> None:
    rows = threshold_table([0.2, 0.6, 0.8], [0, 1, 1])
    top = rows[-1]
    assert top["threshold"] == 0.9
    assert top["predicted_positive"] == 0
    row5 = next(r for r in rows if r["threshold"] == 0.5)
    assert row5["predicted_positive"] == 2 and row5["recall"] == 1.0


def test_oracle_knapsack_beats_or_equals_greedy() -> None:
    from benchmarks.memorybench_jr.context_metrics import oracle_value  # noqa: F811

    class FakeCand:
        def __init__(self, cid: str, tokens: int) -> None:
            self.id = cid
            self.text = "w " * tokens
            self.token_estimate = tokens

    class FakeItem:
        def __init__(self, cid: str, tokens: int, value: float) -> None:
            self.candidate = FakeCand(cid, tokens)
            self.value = value

    items: list[RankedItem] = cast(
        "list[RankedItem]", [FakeItem("a", 6, 0.9), FakeItem("b", 4, 0.7), FakeItem("c", 4, 0.65)]
    )
    assert oracle_value(items, 8) == 1.35  # b + c beats a alone (0.9)


def test_context_efficiency_bounds() -> None:
    from benchmarks.memorybench_jr.context_metrics import context_efficiency  # noqa: F811
    from jev_reranker.models import Candidate, ContextSelection, HeadJudgments, Label, RankedItem

    def item(cid: str, tokens: int, value: float) -> RankedItem:
        return RankedItem(
            candidate=Candidate(id=cid, text="w " * tokens, token_estimate=tokens, retrieval_rank=0),
            judgments=HeadJudgments(
                relevance=3.0, relevance_confidence=0.9, utility=3.0, utility_confidence=0.9,
                superseded=0.0, conflict=0.0,
            ),
            value=value,
            label=Label.USE,
            rank=0,
        )

    items = [item("a", 5, 0.8), item("b", 5, 0.6)]
    # Budget fits only one item; the oracle also picks the better one.
    sel = ContextSelection(selected=[items[0]], total_tokens=5, budget_tokens=5, excluded=[])
    assert context_efficiency(sel, items) == 1.0
    sel2 = ContextSelection(selected=[items[1]], total_tokens=5, budget_tokens=5, excluded=[])
    assert context_efficiency(sel2, items) == 0.75
