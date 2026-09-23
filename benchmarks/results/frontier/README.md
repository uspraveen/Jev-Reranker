# Frontier comparison — BEIR mini-study (2026-09-23)

Jev-Reranker (live `jev-latest`, relevance mode) vs Cohere's flagship reranker
(`rerank-v4.0-pro`, trial key) vs a BM25 retrieval floor, on three BEIR
datasets. Every system re-scored the **identical** BM25 top-30 candidate
lists; NDCG@10 via `pytrec_eval` (trec gains). Full per-query rows are
committed next to this file (`<dataset>__<system>.jsonl`).

## Protocol

- Datasets: `scifact`, `nfcorpus`, `fiqa` (BeIR via HuggingFace parquet).
- 200 seeded queries per dataset (seed 42, `random.Random(42).sample` over
  sorted qids with ≥1 relevant doc) — a subsample, not the full test sets.
- Candidates: bm25s, lowercase + English stopwords, no stemmer, top-30.
  Published BEIR BM25 numbers use heavier pipelines and full query sets;
  our floor (e.g. scifact 0.679 vs published ~0.665) is close but not
  identical — all systems here share it exactly, so comparisons are
  apples-to-apples internally.
- Jev lane: product path unchanged — `JevReranker(rubric="generic_retrieval")`
  relevance mode, one batched `/v1/systemone` call per query (30 rel
  questions), score = relevance head / 3. Unpaced live calls, p50 ≈ 0.2 s.
- Cohere lane: `/v2/rerank`, `rerank-v4.0-pro`, top_n=30. Trial-key pacing
  (≥7 s between requests) — see latency note below.

## NDCG@10 (n=200 per dataset)

| dataset | BM25 (floor) | **Jev-Reranker** | Cohere rerank-v4.0-pro |
|---|---|---|---|
| scifact | 0.6788 | **0.7841** | 0.7728 |
| nfcorpus | 0.3019 | **0.3426** | 0.3346 |
| fiqa | 0.2485 | 0.4069 | **0.4198** |
| **3-set average** | 0.4097 | **0.5112** | 0.5091 |

Jev wins 2 of 3 datasets and the average; Cohere's flagship wins fiqa.
With n=200 per dataset, single-digit-millisecond-scale differences are
within subsample noise — the honest reading is **parity with the flagship
on relevance ranking, from a general-purpose decision model asked explicit
questions, zero-shot** (no relevance-training exposure), at p50 ≈ 0.2 s per
query (30 candidates, one API call).

A score-tie diagnostic (ties broken by policy-value order and by BM25 order)
changed nothing (±0.0000 NDCG on all datasets): Jev's Score answers are
quasi-continuous (e.g. 2.8/3), so tie-handling is not a factor here.

## Latency & cost (context, not a head-to-head)

- Jev: p50 186–208 ms/query across datasets; ~7.34M input tokens over 600
  queries (~12.2k/query: 30 candidates × ~400 tokens + query).
- Cohere: 600 billed search units. Measured wall time per query (~7 s)
  reflects OUR trial-key throttle (7 s pacing), not Cohere's service
  latency — do not quote it as a Cohere latency number.

## Lanes attempted but not reported (no numbers claimed)

- **Voyage `rerank-2.5` / `rerank-2.5-lite`**: trial tier (no payment
  method) — 3 RPM. Single-request smokes succeeded, but the free quota
  exhausted during setup; every subsequent request 429s ("add a payment
  method"), including isolated single calls from separate IPs. Rows from
  the attempt are committed as error JSONLs. Runnable after adding billing.
- **Open-weights `bge-reranker-v2-m3` (GGUF via llama.cpp)**: llama.cpp was
  built from source on the Alpine/musl sandbox (no torch wheels there),
  the Q8_0 GGUF served via `llama-server --reranking`, and the endpoint
  smoke-validated (correct ordering on a toy pair). Measured cost on the
  3-vCPU CPU-only sandbox: **~434 s per query** (one full row, committed)
  → the 600-query pass would take ~70 h. Reported as measured-infeasible
  on this hardware, not skipped silently.

## Provenance & reproduction

- Harness: `benchmarks/frontier/` (data loader, adapters, runner,
  laptop-side sandbox orchestrator). Resumable per-query JSONLs; error rows
  retry on re-run.
- `frontier_results.json` / `provenance.json`: machine-readable outputs.
- `sandbox_*.log`: build/server/run logs from the benchmark sandbox
  (Blaxel, Alpine, 4 GB), torn down after collection.
- Reproduce: `python -m benchmarks.frontier.run_frontier` with
  `TYPESAFE_API_KEY` / `COHERE_API_KEY` set (see `--help`; the sandbox
  path is scripted in `orchestrate_local.py`).

Generated 2026-09-23 from commits at `1c2b4d8`-family harness; runner seed
42; k=30. Author-run, single-seed, subsampled — treat as a mini-study, not
a leaderboard claim.
