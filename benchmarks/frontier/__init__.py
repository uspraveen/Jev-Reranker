"""Frontier head-to-head: Jev-Reranker vs hosted and open-weights rerankers on BEIR.

Protocol: BM25 (lean tokenizer) retrieves top-k candidates per query; every
reranker re-scores the IDENTICAL candidate lists; NDCG@10 / Recall@10 via
pytrec_eval (trec convention). Query subsampling is seeded. Everything is
resumable: per-(system, dataset) JSONL of per-query scores + latencies.
"""
