"""MemoryBench-JR: hard agent-memory benchmark for Jev-Reranker.

Deterministic, seed-fixed generator producing 500 difficult memory-triage
cases across 10 categories, plus runners, calibration, baselines and
context-selection metrics. See README.md in this directory.
"""

from __future__ import annotations

from benchmarks.memorybench_jr.generate import BENCH_VERSION, CATEGORIES, BenchCase, generate_cases

__all__ = ["BENCH_VERSION", "CATEGORIES", "BenchCase", "generate_cases"]
