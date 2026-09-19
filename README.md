# Jev-Reranker

Decision-aware, calibrated context selection for AI agents, powered by TypeSafe Jev.
Jev judges; deterministic Python policy ranks. **One Jev call per rerank** — never one call per memory.

Instead of embedding similarity, Jev-Reranker asks a System-1 decision model
(Jev `/v1/systemone`) four calibrated questions per candidate — relevance
(Score 0–2), utility (Score 0–2), superseded (Noul 0–1), conflict (Noul 0–1) —
**all batched into a single request**, then applies a transparent,
versioned policy (`POLICY_VERSION=v1`, `QUESTION_SCHEMA_VERSION=v1`) to rank,
label, and select.

## Three modes

| Mode | What it does |
|---|---|
| `relevance` | Classic query/passage rerank on policy value. |
| `memory` | Agent-memory triage: `STALE` / `CONFLICT` / `UNCERTAIN` labels surfaced, superseded + conflict heads fully active. |
| `context` | Token-budget selection: greedy value-per-token knapsack over non-`DROP` items so the host agent always fits its window. |

Labels: `USE` (feed the agent) · `KEEP` (background, budget permitting) ·
`DROP` (below value threshold) · `STALE` (superseded) · `CONFLICT`
(contradicted elsewhere) · `UNCERTAIN` (judge confidence below gate).

## Design: one batched Jev call

`policy.build_questions` builds the 4N question map (`rel__<id>`,
`util__<id>`, `sup__<id>`, `con__<id>`); `LiveJevJudge` translates it to SDK
`Score`/`Noul` questions and issues **exactly one** `TypeSafeClient.system_one()`
call per `rerank`/`select` (asserted in tests: `judge.calls == 1`).
Per-case `judge_calls_per_rerank = 1.0` is measured in the synthetic eval.

Two hard rules, enforced by the policy and covered by tests:

1. **Confidence never scales the value.** `policy_value` uses only the two
   Score heads (normalized 0–2 → 0–1) minus superseded/conflict penalties.
   Confidence only *gates*: below `confidence_gate` (0.45) an item becomes
   `UNCERTAIN` instead of acting on its value.
2. **Never one Jev call per memory.** Candidates × heads are one payload;
   resilience (candidate caps, truncation, cache, fallback) keeps the host
   agent non-brittle when Jev is unreachable (`fallback='retrieval_order'`
   preserves retrieval order with zero-confidence `UNCERTAIN` judgments).

## Setup

```bash
pip install -e ".[dev]"          # tests, lint, types (+numpy for eval extras)
pip install -e ".[integrations]" # langchain / llama-index / qdrant adapters
export TYPESAFE_API_KEY=...      # live Jev; without it, OfflineJudge is used
```

`JevReranker()` auto-picks `LiveJevJudge` iff `TYPESAFE_API_KEY` is set,
else the deterministic lexical `OfflineJudge` (test/eval/demo only —
always labeled `offline-judge-v1`, never presented as Jev output).

## Usage

```python
from jev_reranker import Candidate, JevReranker

rr = JevReranker()
res = rr.rerank(
    "how long do refunds take to settle?",
    [Candidate(id="a", text="Refunds settle within 5 business days.", retrieval_rank=0),
     Candidate(id="b", text="OUTDATED deprecated old version: refunds settle in 30 days.",
               retrieval_rank=1)],
    mode="memory",
)
for it in res.items:
    print(it.rank, it.label.value, round(it.value, 3), it.candidate.text)

sel = rr.select("summarize the refund policy", cands, budget_tokens=150)
print(sel.total_tokens, "of", sel.budget_tokens)
```

More: `examples/quickstart.py`, `examples/memory_triage.py`,
`examples/context_budget.py`, `demo/arena.py --cli`
(or `streamlit run demo/arena.py`). Framework adapters
(LangChain compressor, LangGraph `memory_triage_node`, LlamaIndex
postprocessor, Qdrant helper) live in `jev_reranker.integrations`.

## Measured results (synthetic hard-memory eval)

`python -m jev_reranker.eval_synthetic --n 300 --seed 7` —
300 generated cases (6 categories × 50: stale, conflict, distractor,
paraphrase, multi_hop, recency), real `JevReranker` pipeline, measured
2026-09-19 (trace: `benchmarks/results/synthetic_eval.json`):

| metric | value |
|---|---|
| judge | `offline-judge-v1` (lexical stand-in — see caveat) |
| recall@1 | 0.1667 |
| recall@3 | 0.8333 |
| STALE/CONFLICT label precision / recall / F1 | 1.0 / 1.0 / 1.0 |
| judge calls per rerank | 1.0 |
| fallback rate | 0.0 |
| p50 latency | ~1.3 ms |
| per-category recall@1 (stale / conflict / distractor / paraphrase / multi_hop / recency) | 0.0 / 0.0 / 0.0 / 0.0 / 1.0 / 0.0 |

Caveat (read before quoting): the recall numbers are bounded by the
*lexical* offline judge — distractors repeat the query verbatim, so token
overlap favors them over the gold answer. What this eval genuinely verifies
is pipeline mechanics: exactly one judge call per rerank, perfect
STALE/CONFLICT flagging on cue-bearing memories, zero fallbacks. Ranking
quality against **live Jev is not yet measured** (`TYPESAFE_API_KEY` absent).

Sandbox rerun: the same eval (`--n 300 --seed 7`) was executed inside a
Blaxel sandbox (`learnchain` workspace, 2026-09-19, trace:
`benchmarks/results/synthetic_eval_blaxel.json`) with identical results
(r@1 0.1667, r@3 0.8333, F1 1.0, 1.0 calls/rerank, 0 fallback; p50 0.68 ms).

Confidence snapshot (same 300 cases, top-1 bucketed by `min_confidence`):
all 300 land in `[0.75, 1.00]` at accuracy 0.1667 — the lexical judge is
uniformly overconfident, so its confidence carries no discrimination.
Live-Jev calibration (reliability curve, threshold tuning) is open work
pending an API key; thresholds were deliberately **not** tuned to the
offline stand-in.

Other checks (2026-09-19, local VM): `pytest` 22 passed; `ruff check`
clean; `mypy --strict` clean (10 files); arena CLI verified on both
samples (STALE + CONFLICT correctly flagged); all three examples run.
LongMemEval slice runner exists (`longmemeval_slice.py`, `--n 50`) but the
`xiaojiu-z/LongMemEvalS` dataset was unreachable from this VM, so no
numbers are reported for it.

## Layout

```
src/jev_reranker/  policy.py client.py judges.py models.py reranker.py
                   cache.py integrations.py eval_synthetic.py longmemeval_slice.py
tests/             policy / client / reranker / cache-schema (22 tests)
examples/ demo/ benchmarks/results/ scripts/run_benchmarks.py
```

