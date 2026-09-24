# MemoryBench-JR

Hard agent-memory benchmark for Jev-Reranker. Deterministic, seed-fixed,
**500 cases across 10 categories** (50 each). Every number reported by the
runners is measured by executing the real pipeline - nothing is simulated,
extrapolated, or copied.

## Categories

| category | what it tests |
|---|---|
| `superseded_fact` | old oncall assignment vs newer calendar entry (gold STALE on the old) |
| `contradiction` | policy doc vs contradicting chatter (gold CONFLICT on the chatter) |
| `near_duplicate` | reworded duplicate of the same fact (dedup must suppress one) |
| `wrong_entity_decoy` | same keywords, wrong service/entity in the answer |
| `old_plan_vs_final` | early plan vs recorded final decision (plan is STALE) |
| `changed_preference` | outdated user preference vs explicit update (old is STALE) |
| `temporal_query` | "as of DATE" - pick the value valid at that date |
| `irrelevant_lexical` | keyword overlap with wrong intent ("rotate the API key") |
| `multi_hop` | answer requires chaining two memories (owner + escalation policy) |
| `answer_not_present` | no candidate answers; correct behavior is to USE nothing |

Each case carries a query, 4-6 `MemoryItem` candidates (with timestamps,
sources, importance, entity_ids, session_id), a gold answer id, gold
STALE/CONFLICT labels, relevant ids, and near-duplicate ids.

## Reproducing

```bash
pip install -e ".[dev]"        # offline baseline (CI-fast)
python -m benchmarks.memorybench_jr.run_eval --judge offline

export TYPESAFE_API_KEY=...    # live run (~2 API calls per case: memory + relevance)
python -m benchmarks.memorybench_jr.run_eval --judge live --n 500 --seed 7
```

The live runner is sequential with a configurable inter-call sleep
(`--sleep`, default 0.05 s) and relies on the SDK `RetryPolicy` (exponential
backoff + jitter, honors Retry-After) for 429/5xx. The context pass and all
baselines are local: the context pass reuses cached judgments (same
question payload as memory mode), and the embedding/retrieval-order
baselines never call Jev.

## Metrics

- **recall@1 / recall@3** - gold answer ranked first / in top 3 (memory and
  relevance modes), excluding `answer_not_present` cases.
- **rejection_accuracy** - on `answer_not_present` cases, fraction where no
  candidate was labeled USE/KEEP.
- **label precision/recall/F1** - per STALE and CONFLICT gold labels.
- **dedup precision/recall/F1** - context-pass near-duplicate suppression vs
  gold `near_dup_ids`.
- **Context Precision / Recall / Efficiency** (owner section 21):
  - Precision = |selected ∩ relevant| / |selected|
  - Recall = |selected ∩ relevant| / |relevant|
  - Efficiency = sum(value of selected) / sum(value of the ORACLE selection
    under the same budget), oracle = exact DP knapsack over the same items.
  Budget = 45% of the case's total tokens per case.
- **Calibration** (live run only; from `predictions_memorybench_jr_live.jsonl`):
  - Binary outcomes per candidate: `useful` (is gold; p = utility/3),
    `relevant` (is gold; p = relevance/3), `superseded` (gold STALE; p =
    Noul), `conflict` (gold CONFLICT; p = Noul).
  - Brier score, ECE (10 equal-width bins), reliability curves
    (`calibration_reliability_curves.png`), accuracy-vs-threshold /
    coverage tables (`calibration_accuracy_vs_threshold.png`).

## Files

- `generate.py` - the deterministic generator (`generate_cases(n, seed)`).
- `run_eval.py` - full suite runner (writes results JSON + predictions JSONL + PNGs).
- `baselines.py` - retrieval-order and hashed-bag-of-words embedding baselines.
- `calibration.py` - Brier/ECE/curves/threshold tables.
- `context_metrics.py` - context precision/recall/efficiency + oracle knapsack.
- `plots.py` - matplotlib renderers (Agg backend).

Results land in `benchmarks/results/` (committed with full provenance:
date, host, model, API call counts, seeds, git commit).
