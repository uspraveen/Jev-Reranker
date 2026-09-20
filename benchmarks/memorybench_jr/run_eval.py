"""MemoryBench-JR runner: real measured eval + baselines + calibration + context metrics.

One process per judge kind. For the live judge this makes exactly ONE Jev
call per case in memory mode and ONE in relevance mode (sequential, modest
rate, SDK retry/backoff on 429/5xx); the context pass and all baselines are
local (cache hits / pure code). Every emitted number is measured here.

Usage:
  python -m benchmarks.memorybench_jr.run_eval --judge offline
  python -m benchmarks.memorybench_jr.run_eval --judge live --n 500 --seed 7
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.memorybench_jr.baselines import rank_embedding, rank_retrieval_order
from benchmarks.memorybench_jr.calibration import compute_all
from benchmarks.memorybench_jr.context_metrics import context_efficiency, context_precision, context_recall
from benchmarks.memorybench_jr.generate import BENCH_VERSION, BenchCase, generate_cases
from benchmarks.memorybench_jr.plots import plot_accuracy_vs_threshold, plot_reliability
from jev_reranker.judges import AsyncLiveJevJudge, OfflineJudge
from jev_reranker.policy import POLICY_VERSION, QUESTION_SCHEMA_VERSION
from jev_reranker.reranker import JevReranker, candidate_tokens

RESULTS = Path("benchmarks/results")


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except Exception:
        return "unknown"


def provenance(judge_name: str, api_calls: int, n: int, seed: int) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "judge": judge_name,
        "model": "jev-latest",
        "api_calls": api_calls,
        "n_cases": n,
        "seed": seed,
        "bench_version": BENCH_VERSION,
        "policy_version": POLICY_VERSION,
        "schema_version": QUESTION_SCHEMA_VERSION,
        "git_commit": _git_commit(),
    }


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return round(prec, 4), round(rec, 4), round(f1, 4)


def ranked_metrics(preds: list[dict[str, Any]]) -> dict[str, Any]:
    """recall@1/@3, rejection accuracy, STALE/CONFLICT label P/R/F1."""
    n_rank = r1 = r3 = 0
    rej_n = rej_ok = 0
    lab: dict[str, dict[str, int]] = {
        "STALE": {"tp": 0, "fp": 0, "fn": 0},
        "CONFLICT": {"tp": 0, "fp": 0, "fn": 0},
    }
    for p in preds:
        if p["gold_id"]:
            n_rank += 1
            ranked = p["ranked_ids"]
            r1 += int(ranked[0] == p["gold_id"])
            r3 += int(p["gold_id"] in ranked[:3])
        else:
            rej_n += 1
            rej_ok += int(all(lb not in ("USE", "KEEP") for lb in p["labels"].values()))
        for cid, glb in p["gold_labels"].items():
            plb = p["labels"].get(cid)
            if plb == glb:
                lab[glb]["tp"] += 1
            else:
                lab[glb]["fn"] += 1
        for cid, plb in p["labels"].items():
            if plb in lab and cid not in p["gold_labels"]:
                lab[plb]["fp"] += 1
    label_stats = {
        name: dict(
            tp=v["tp"], fp=v["fp"], fn=v["fn"],
            **dict(zip(("precision", "recall", "f1"), _prf(v["tp"], v["fp"], v["fn"]), strict=True)),
        )
        for name, v in lab.items()
    }
    pooled_tp = sum(v["tp"] for v in lab.values())
    pooled_fp = sum(v["fp"] for v in lab.values())
    pooled_fn = sum(v["fn"] for v in lab.values())
    return {
        "recall@1": round(r1 / n_rank, 4) if n_rank else 0.0,
        "recall@3": round(r3 / n_rank, 4) if n_rank else 0.0,
        "rejection_accuracy": round(rej_ok / rej_n, 4) if rej_n else None,
        "label_stats": label_stats,
        "label_f1_STALE_CONFLICT": _prf(pooled_tp, pooled_fp, pooled_fn)[2],
    }


def dedup_metrics(preds_ctx: list[dict[str, Any]]) -> dict[str, float]:
    """Near-duplicate suppression quality from the context pass."""
    tp = fp = fn = 0
    for p in preds_ctx:
        gold = p["near_dup_ids"]
        pred = p["suppressed_ids"]
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    prec, rec, f1 = _prf(tp, fp, fn)
    return {"dedup_precision": prec, "dedup_recall": rec, "dedup_f1": f1}


async def run_suite(judge_kind: str, n: int, seed: int, sleep_s: float, out_dir: Path) -> dict[str, Any]:
    cases = generate_cases(n, seed)
    if judge_kind == "live":
        judge = AsyncLiveJevJudge()
    else:
        judge = OfflineJudge()
    rr = JevReranker(judge=judge)

    preds_mem: list[dict[str, Any]] = []
    preds_rel: list[dict[str, Any]] = []
    preds_ctx: list[dict[str, Any]] = []
    ctx_rows: list[dict[str, Any]] = []
    t_wall0 = time.perf_counter()
    for case in cases:
        mem = await rr.arerank(case.query, case.candidates, mode="memory")
        preds_mem.append(_record(case, mem))
        rel = await rr.arerank(case.query, case.candidates, mode="relevance")
        preds_rel.append(_record(case, rel))
        total_tokens = sum(candidate_tokens(c) for c in case.candidates)
        budget = max(60, round(total_tokens * 0.45))
        sel = await rr.aselect(case.query, case.candidates, budget_tokens=budget)
        preds_ctx.append(
            {
                "case_id": case.id,
                "category": case.category,
                "selected_ids": [it.candidate.id for it in sel.selected],
                "suppressed_ids": {it.candidate.id for it in sel.suppressed_near_duplicates},
                "near_dup_ids": set(case.near_dup_ids),
                "budget_tokens": budget,
                "total_tokens": sel.total_tokens,
            }
        )
        ctx_rows.append(
            {
                "case_id": case.id,
                "category": case.category,
                "precision": round(context_precision(sel, case.relevant_ids), 4),
                "recall": round(context_recall(sel, case.relevant_ids), 4),
                "efficiency": context_efficiency(sel, mem.items),
            }
        )
        if sleep_s:
            await asyncio.sleep(sleep_s)
    wall_min = round((time.perf_counter() - t_wall0) / 60.0, 2)

    # -- baselines (local, no API)
    base_rows: dict[str, dict[str, float]] = {"retrieval_order": {}, "embedding": {}}
    for name, ranker in (("retrieval_order", rank_retrieval_order), ("embedding", rank_embedding)):
        r1 = r3 = 0
        n_rank = 0
        for case in cases:
            if not case.gold_id:
                continue
            n_rank += 1
            ranked = ranker(case)
            r1 += int(ranked[0] == case.gold_id)
            r3 += int(case.gold_id in ranked[:3])
        base_rows[name] = {
            "recall@1": round(r1 / n_rank, 4) if n_rank else 0.0,
            "recall@3": round(r3 / n_rank, 4) if n_rank else 0.0,
        }

    mem_stats = ranked_metrics(preds_mem)
    rel_stats = ranked_metrics(preds_rel)

    # context aggregate
    prec_avg = round(sum(r["precision"] for r in ctx_rows) / len(ctx_rows), 4)
    rec_avg = round(sum(r["recall"] for r in ctx_rows) / len(ctx_rows), 4)
    eff_avg = round(sum(r["efficiency"] for r in ctx_rows) / len(ctx_rows), 4)
    per_cat: dict[str, dict[str, float]] = {}
    for r in ctx_rows:
        bucket = per_cat.setdefault(r["category"], {"n": 0, "precision": 0.0, "recall": 0.0, "efficiency": 0.0})
        bucket["n"] += 1
        for k in ("precision", "recall", "efficiency"):
            bucket[k] = round(bucket[k] + r[k], 4)

    # -- calibration only meaningful for the live run (real head probabilities)
    calib: dict[str, Any] = {}
    if judge_kind == "live" and out_dir is not None:
        jsonl_path = out_dir / "predictions_memorybench_jr_live.jsonl"
        _write_predictions_jsonl(preds_mem, jsonl_path)
        calib = compute_all(jsonl_path)
        plot_reliability(calib["heads"], out_dir / "calibration_reliability_curves.png")
        plot_accuracy_vs_threshold(calib["heads"], out_dir / "calibration_accuracy_vs_threshold.png")

    judge_calls = judge.calls
    remote_calls = judge_calls if judge_kind == "live" else 0
    result: dict[str, Any] = {
        "benchmark": "MemoryBench-JR",
        "bench_version": BENCH_VERSION,
        "judge": judge.model_name,
        "judge_kind": judge_kind,
        "mode_memory": mem_stats,
        "mode_relevance": rel_stats,
        "context": {
            "precision": prec_avg,
            "recall": rec_avg,
            "efficiency": eff_avg,
            "budget_fraction": 0.45,
            "per_category": {
                k: {kk: (round(vv / b["n"], 4) if kk != "n" else vv) for kk, vv in b.items()}
                for k, b in per_cat.items()
            },
        },
        "dedup": dedup_metrics(preds_ctx),
        "baseline_table": {
            "retrieval_order": base_rows["retrieval_order"],
            "embedding_hashed_bow": base_rows["embedding"],
            "jev_relevance": {"recall@1": rel_stats["recall@1"], "recall@3": rel_stats["recall@3"]},
            "jev_memory": {"recall@1": mem_stats["recall@1"], "recall@3": mem_stats["recall@3"]},
        },
        "run": {
            "api_calls": judge_calls,
            "remote_api_calls": remote_calls,
            "cache_hits": rr.client.cache_hits,
            "p50_latency_ms": sorted(p["latency_ms"] for p in preds_mem)[len(preds_mem) // 2],
            "fallbacks": sum(1 for p in preds_mem if p["fallback_used"]),
            "wall_minutes": wall_min,
        },
        "provenance": provenance(judge.model_name, remote_calls, n, seed),
    }
    if calib:
        result["calibration"] = {
            h: {k: v for k, v in d.items()
                if k in ("n", "positive_rate", "brier", "ece", "reliability_curve", "threshold_table")}
            for h, d in calib["heads"].items()
        }
    await rr.aclose()
    return result


def _record(case: BenchCase, res: Any) -> dict[str, Any]:
    return {
        "case_id": case.id,
        "category": case.category,
        "gold_id": case.gold_id,
        "gold_labels": case.gold_labels,
        "near_dup_ids": set(case.near_dup_ids),
        "ranked_ids": [it.candidate.id for it in res.items],
        "labels": {it.candidate.id: it.label.value for it in res.items},
        "judgments": {
            it.candidate.id: {
                "relevance": it.judgments.relevance,
                "relevance_confidence": it.judgments.relevance_confidence,
                "utility": it.judgments.utility,
                "utility_confidence": it.judgments.utility_confidence,
                "superseded": it.judgments.superseded,
                "conflict": it.judgments.conflict,
            }
            for it in res.items
        },
        "latency_ms": res.latency_ms or 0.0,
        "cached": res.cached,
        "fallback_used": res.fallback_used,
    }


def _write_predictions_jsonl(preds: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for p in preds:
            rec = {**p}
            rec["near_dup_ids"] = sorted(rec["near_dup_ids"])
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", choices=["offline", "live"], default="offline")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sleep", type=float, default=0.05, help="seconds between live calls (rate limiting)")
    ap.add_argument("--out", type=Path, default=None, help="override results JSON path")
    args = ap.parse_args()

    out_dir = RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (out_dir / f"memorybench_jr_{'live' if args.judge == 'live' else 'offline'}.json")
    result = asyncio.run(run_suite(args.judge, args.n, args.seed, args.sleep if args.judge == "live" else 0.0, out_dir))
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    summary = {
        "memory_r1": result["mode_memory"]["recall@1"],
        "memory_f1": result["mode_memory"]["label_f1_STALE_CONFLICT"],
        "relevance_r1": result["mode_relevance"]["recall@1"],
        "context_PRE": [result["context"]["precision"], result["context"]["recall"], result["context"]["efficiency"]],
        "baselines": result["baseline_table"],
        "api_calls": result["run"]["api_calls"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
