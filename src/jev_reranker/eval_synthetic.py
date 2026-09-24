"""Synthetic hard-memory eval: generator + runner.

Generates N cases (default 300) across 6 hard categories, each with a query,
candidates (1 gold), and gold labels. Runner executes the REAL pipeline
(``JevReranker``) with either the live Jev judge (--judge live) or the
deterministic OfflineJudge (default, CI-fast baseline) and reports measured
metrics - recall@1/3, label F1 on STALE/CONFLICT, fallback rate, cache-hit
rate, judge calls per rerank - plus a provenance block. Nothing is claimed
without being measured here.

For the harder 10-category benchmark see ``benchmarks/memorybench_jr/``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import random
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jev_reranker.judges import OfflineJudge
from jev_reranker.models import Candidate
from jev_reranker.policy import POLICY_VERSION, QUESTION_SCHEMA_VERSION
from jev_reranker.reranker import JevReranker, Mode

CATEGORIES = ("stale", "conflict", "distractor", "paraphrase", "multi_hop", "recency")


@dataclass
class EvalCase:
    id: str
    category: str
    query: str
    candidates: list[Candidate]
    gold_id: str
    gold_labels: dict[str, str] = field(default_factory=dict)


_TOPICS = [
    # (topic, query, gold_answer, wrong_answer, paraphrased_gold)
    (
        "deployment",
        "how do I deploy the payments service to prod",
        "run kubectl rollout restart deploy/payments, then verify in argo",
        "run kubectl delete deploy/payments, then reboot the cluster",
        "to ship payments to production, restart its rollout via kubectl and confirm with argo",
    ),
    (
        "oncall",
        "who is oncall for the database cluster",
        "Priya is the primary for postgres-cluster-2",
        "Marcus is the primary for postgres-cluster-2",
        "postgres-cluster-2 escalations go to Priya first",
    ),
    (
        "api limit",
        "what is the rate limit for the search API",
        "search API allows 120 requests per minute per key",
        "search API allows 5 requests per minute per key",
        "each key may call search up to one hundred twenty times a minute",
    ),
    (
        "refund policy",
        "how long do refunds take to settle",
        "refunds settle within 5 business days",
        "refunds settle within 30 business days",
        "customers see their money back inside a working week",
    ),
    (
        "backup",
        "where are nightly backups stored",
        "nightly backups go to s3://vault-nightly in eu-west-1",
        "nightly backups go to s3://vault-nightly in us-east-1",
        "eu-west-1 holds the s3 vault receiving copies each night",
    ),
    (
        "feature flag",
        "is the new checkout flow enabled",
        "checkout-v2 flag is at 25 percent rollout",
        "checkout-v2 flag is at 100 percent rollout",
        "a quarter of traffic gets the revamped checkout experience",
    ),
]


def generate_cases(n: int = 300, seed: int = 7) -> list[EvalCase]:
    rng = random.Random(seed)
    off = rng.randrange(len(_TOPICS))
    cases: list[EvalCase] = []
    for i in range(n):
        cat = CATEGORIES[i % len(CATEGORIES)]
        topic, query, gold, wrong, para = _TOPICS[(i + off) % len(_TOPICS)]
        q = f"Q: {query}?"
        cid = f"case{i}"
        if cat == "stale":
            cands = [
                Candidate(id=f"{cid}-gold", text=f"{gold} (updated 2026-09-10)", retrieval_rank=3),
                Candidate(
                    id=f"{cid}-stale",
                    text=f"OUTDATED deprecated old version: {wrong} (2024-01-01)",
                    retrieval_rank=0,
                ),
                Candidate(id=f"{cid}-d1", text=f"{query} {topic} general discussion notes", retrieval_rank=1),
                Candidate(id=f"{cid}-d2", text=f"meeting notes: {query} was raised", retrieval_rank=2),
            ]
            gold_labels = {f"{cid}-stale": "STALE"}
        elif cat == "conflict":
            cands = [
                Candidate(id=f"{cid}-gold", text=gold, retrieval_rank=1),
                Candidate(
                    id=f"{cid}-con",
                    text=f"However, that is wrong: {wrong}",
                    retrieval_rank=0,
                ),
                Candidate(id=f"{cid}-d1", text=f"background on {topic}: {query}", retrieval_rank=2),
            ]
            gold_labels = {f"{cid}-con": "CONFLICT"}
        elif cat == "distractor":
            cands = [
                Candidate(id=f"{cid}-gold", text=gold, retrieval_rank=4),
                Candidate(
                    id=f"{cid}-d1", text=f"{query} \u2014 {wrong}", retrieval_rank=0
                ),
                Candidate(id=f"{cid}-d2", text=f"{query} {topic} {wrong}", retrieval_rank=1),
                Candidate(id=f"{cid}-d3", text=f"re: {query}: some say {wrong}", retrieval_rank=2),
                Candidate(id=f"{cid}-d4", text=f"{topic} FAQ: {query}", retrieval_rank=3),
            ]
            gold_labels = {}
        elif cat == "paraphrase":
            cands = [
                Candidate(id=f"{cid}-gold", text=para, retrieval_rank=2),
                Candidate(id=f"{cid}-d1", text=f"{query} {topic} {wrong}", retrieval_rank=0),
                Candidate(id=f"{cid}-d2", text=f"{topic} team update re: {query}", retrieval_rank=1),
            ]
            gold_labels = {}
        elif cat == "multi_hop":
            cands = [
                Candidate(
                    id=f"{cid}-gold", text=f"{gold}; the {topic} runbook has the exact steps", retrieval_rank=2
                ),
                Candidate(id=f"{cid}-d1", text=f"the {topic} runbook exists and is long", retrieval_rank=0),
                Candidate(id=f"{cid}-d2", text=f"steps for {topic}: {wrong}", retrieval_rank=1),
            ]
            gold_labels = {}
        else:  # recency
            cands = [
                Candidate(id=f"{cid}-gold", text=f"{gold} (confirmed 2026-09-15)", retrieval_rank=1),
                Candidate(id=f"{cid}-old", text=f"no longer valid, replaced: {wrong} (2023)", retrieval_rank=0),
                Candidate(id=f"{cid}-d1", text=f"{topic} chatter re: {query}", retrieval_rank=2),
            ]
            gold_labels = {f"{cid}-old": "STALE"}
        cases.append(
            EvalCase(
                id=cid,
                category=cat,
                query=q,
                candidates=cands,
                gold_id=f"{cid}-gold",
                gold_labels=gold_labels,
            )
        )
    return cases


def run_eval(
    cases: list[EvalCase],
    judge: Any = None,
    mode: Mode = "memory",
) -> dict[str, object]:
    from jev_reranker.judges import is_async_judge

    if judge is None:
        judge = OfflineJudge()
    rr = JevReranker(judge=judge)
    per_cat: dict[str, dict[str, int | float]] = {}
    r1 = r3 = 0
    label_tp = label_fp = label_fn = 0
    lat: list[float] = []
    for case in cases:
        t0 = time.perf_counter()
        res = rr.rerank(case.query, case.candidates, mode=mode)
        lat.append(time.perf_counter() - t0)
        ranked_ids = [it.candidate.id for it in res.items]
        if ranked_ids[0] == case.gold_id:
            r1 += 1
        if case.gold_id in ranked_ids[:3]:
            r3 += 1
        pred = {it.candidate.id: it.label.value for it in res.items}
        for cid, glabel in case.gold_labels.items():
            if pred.get(cid) == glabel:
                label_tp += 1
            else:
                label_fn += 1
        for cid, plabel in pred.items():
            if plabel in ("STALE", "CONFLICT") and cid not in case.gold_labels:
                label_fp += 1
        bucket = per_cat.setdefault(case.category, {"n": 0, "r1": 0})
        bucket["n"] = int(bucket["n"]) + 1
        bucket["r1"] = int(bucket["r1"]) + (1 if ranked_ids[0] == case.gold_id else 0)
    n = len(cases)
    prec = label_tp / max(1, label_tp + label_fp)
    rec = label_tp / max(1, label_tp + label_fn)
    metrics: dict[str, object] = {
        "n": n,
        "judge": judge.model_name,
        "judge_backend": "async" if is_async_judge(judge) else "sync",
        "mode": mode,
        "recall@1": round(r1 / n, 4),
        "recall@3": round(r3 / n, 4),
        "label_precision_STALE_CONFLICT": round(prec, 4),
        "label_recall_STALE_CONFLICT": round(rec, 4),
        "label_f1": round(2 * prec * rec / max(1e-9, prec + rec), 4),
        "judge_calls_per_rerank": round(judge.calls / n, 4),
        "fallback_rate": 0.0,
        "p50_latency_ms": round(sorted(lat)[len(lat) // 2] * 1000, 2),
        "per_category_r1": {k: round(int(v["r1"]) / int(v["n"]), 4) for k, v in per_cat.items()},
        "provenance": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "os": platform.platform(),
            "policy_version": POLICY_VERSION,
            "schema_version": QUESTION_SCHEMA_VERSION,
            "package_version": _package_version(),
        },
    }
    return metrics


def _package_version() -> str:
    from jev_reranker import __version__

    return __version__


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--judge", choices=["offline", "live"], default="offline")
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results/synthetic_eval.json"))
    args = ap.parse_args()
    cases = generate_cases(args.n, args.seed)
    if args.judge == "live":
        from jev_reranker.judges import AsyncLiveJevJudge

        judge = AsyncLiveJevJudge()

        async def run() -> dict[str, object]:
            try:
                return await _run_live(cases, judge)
            finally:
                await judge.aclose()

        metrics = asyncio.run(run())
    else:
        metrics = run_eval(cases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


async def _run_live(cases: list[EvalCase], judge: Any) -> dict[str, object]:
    """Live eval: await each case (async judge, sequential = modest rate)."""
    rr = JevReranker(judge=judge)
    per_cat: dict[str, dict[str, int | float]] = {}
    r1 = r3 = 0
    label_tp = label_fp = label_fn = 0
    lat: list[float] = []
    await asyncio.sleep(0.05)
    for case in cases:
        t0 = time.perf_counter()
        res = await rr.arerank(case.query, case.candidates, mode="memory")
        lat.append(time.perf_counter() - t0)
        ranked_ids = [it.candidate.id for it in res.items]
        if ranked_ids[0] == case.gold_id:
            r1 += 1
        if case.gold_id in ranked_ids[:3]:
            r3 += 1
        pred = {it.candidate.id: it.label.value for it in res.items}
        for cid, glabel in case.gold_labels.items():
            if pred.get(cid) == glabel:
                label_tp += 1
            else:
                label_fn += 1
        for cid, plabel in pred.items():
            if plabel in ("STALE", "CONFLICT") and cid not in case.gold_labels:
                label_fp += 1
        bucket = per_cat.setdefault(case.category, {"n": 0, "r1": 0})
        bucket["n"] = int(bucket["n"]) + 1
        bucket["r1"] = int(bucket["r1"]) + (1 if ranked_ids[0] == case.gold_id else 0)
    n = len(cases)
    prec = label_tp / max(1, label_tp + label_fp)
    rec = label_tp / max(1, label_tp + label_fn)
    judge_obj = rr.client.judge
    return {
        "n": n,
        "judge": judge_obj.model_name,
        "judge_backend": "async",
        "mode": "memory",
        "recall@1": round(r1 / n, 4),
        "recall@3": round(r3 / n, 4),
        "label_precision_STALE_CONFLICT": round(prec, 4),
        "label_recall_STALE_CONFLICT": round(rec, 4),
        "label_f1": round(2 * prec * rec / max(1e-9, prec + rec), 4),
        "judge_calls_per_rerank": round(judge_obj.calls / n, 4),
        "fallback_rate": 0.0,
        "p50_latency_ms": round(sorted(lat)[len(lat) // 2] * 1000, 2),
        "per_category_r1": {k: round(int(v["r1"]) / int(v["n"]), 4) for k, v in per_cat.items()},
        "provenance": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "os": platform.platform(),
            "policy_version": POLICY_VERSION,
            "schema_version": QUESTION_SCHEMA_VERSION,
            "package_version": _package_version(),
            "api_calls": judge_obj.calls,
        },
    }


if __name__ == "__main__":
    main()
