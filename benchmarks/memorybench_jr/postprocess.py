"""Recompute policy-dependent MemoryBench-JR metrics from a predictions JSONL.

Raw Jev judgments (the per-candidate head values in the JSONL) are
policy-independent facts from the live run; this script re-applies the
current deterministic policy to them and recomputes labels, rankings,
context selection and dedup metrics without any API calls. Calibration
outputs are untouched (they depend only on raw judgments).

Usage:
  python -m benchmarks.memorybench_jr.postprocess \
      --predictions benchmarks/results/predictions_memorybench_jr_live.jsonl \
      --in-place benchmarks/results/memorybench_jr_live.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.memorybench_jr.context_metrics import context_efficiency, context_precision, context_recall
from benchmarks.memorybench_jr.generate import generate_cases
from benchmarks.memorybench_jr.run_eval import dedup_metrics, ranked_metrics
from jev_reranker.models import HeadJudgments, PolicyConfig
from jev_reranker.policy import POLICY_VERSION, apply_policy, heads_for_mode
from jev_reranker.reranker import candidate_tokens, select_for_budget


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--result", type=Path, required=True, help="results JSON to update in place")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg = PolicyConfig()
    cases = generate_cases(args.n, args.seed)
    case_by_id = {c.id: c for c in cases}

    preds_mem: list[dict[str, Any]] = []
    preds_rel: list[dict[str, Any]] = []
    preds_ctx: list[dict[str, Any]] = []
    ctx_rows: list[dict[str, Any]] = []

    with args.predictions.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            case = case_by_id[rec["case_id"]]
            for mode, store in (("memory", preds_mem), ("relevance", preds_rel)):
                heads = heads_for_mode(mode)
                judgments = {cid: HeadJudgments(**jd) for cid, jd in rec["judgments"].items()}
                items = apply_policy(case.candidates, judgments, cfg, heads)
                store.append(
                    {
                        "case_id": rec["case_id"],
                        "category": rec["category"],
                        "gold_id": rec["gold_id"],
                        "gold_labels": rec["gold_labels"],
                        "ranked_ids": [it.candidate.id for it in items],
                        "labels": {it.candidate.id: it.label.value for it in items},
                    }
                )
                if mode == "memory":
                    mem_items = items
            total_tokens = sum(candidate_tokens(c) for c in case.candidates)
            budget = max(60, round(total_tokens * 0.45))
            sel = select_for_budget(mem_items, budget)
            preds_ctx.append(
                {
                    "case_id": rec["case_id"],
                    "category": rec["category"],
                    "suppressed_ids": {it.candidate.id for it in sel.suppressed_near_duplicates},
                    "near_dup_ids": set(case.near_dup_ids),
                }
            )
            ctx_rows.append(
                {
                    "case_id": rec["case_id"],
                    "category": rec["category"],
                    "precision": round(context_precision(sel, case.relevant_ids), 4),
                    "recall": round(context_recall(sel, case.relevant_ids), 4),
                    "efficiency": context_efficiency(sel, mem_items),
                }
            )

    result = json.loads(args.result.read_text(encoding="utf-8"))
    result["mode_memory"] = ranked_metrics(preds_mem)
    result["mode_relevance"] = ranked_metrics(preds_rel)
    prec_avg = round(sum(r["precision"] for r in ctx_rows) / len(ctx_rows), 4)
    rec_avg = round(sum(r["recall"] for r in ctx_rows) / len(ctx_rows), 4)
    eff_avg = round(sum(r["efficiency"] for r in ctx_rows) / len(ctx_rows), 4)
    per_cat: dict[str, dict[str, float]] = {}
    for r in ctx_rows:
        bucket = per_cat.setdefault(r["category"], {"n": 0, "precision": 0.0, "recall": 0.0, "efficiency": 0.0})
        bucket["n"] += 1
        for k in ("precision", "recall", "efficiency"):
            bucket[k] = round(bucket[k] + r[k], 4)
    result["context"] = {
        "precision": prec_avg,
        "recall": rec_avg,
        "efficiency": eff_avg,
        "budget_fraction": 0.45,
        "per_category": {
            k: {kk: (round(vv / b["n"], 4) if kk != "n" else vv) for kk, vv in b.items()}
            for k, b in per_cat.items()
        },
    }
    result["dedup"] = dedup_metrics(preds_ctx)
    result["baseline_table"]["jev_relevance"] = {
        "recall@1": result["mode_relevance"]["recall@1"],
        "recall@3": result["mode_relevance"]["recall@3"],
    }
    result["baseline_table"]["jev_memory"] = {
        "recall@1": result["mode_memory"]["recall@1"],
        "recall@3": result["mode_memory"]["recall@3"],
    }
    result["recomputed"] = {
        "policy_version": POLICY_VERSION,
        "note": (
            "labels/rankings/context/dedup recomputed offline from the committed predictions JSONL "
            "after a policy-layer fix (head flags gate on their own Noul probability; dominant head "
            "wins; score-confidence gates value labels only). Raw judgments and calibration are "
            "unchanged from the live run."
        ),
    }
    args.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"updated {args.result}")
    print(json.dumps({
        "memory": result["mode_memory"],
        "dedup": result["dedup"],
        "context": {k: result["context"][k] for k in ("precision", "recall", "efficiency")},
    }, indent=2)[:2000])


if __name__ == "__main__":
    main()
