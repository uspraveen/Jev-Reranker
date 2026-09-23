"""Frontier run CLI.

    python -m benchmarks.frontier.run_frontier --smoke          # 5 queries, scifact only
    python -m benchmarks.frontier.run_frontier                   # full run

Resumable: per-(dataset, system) JSONL in --out; completed queries are
skipped on re-run. Writes metrics_partial.json after every (dataset, system)
and frontier_results.json at the end.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.frontier.adapters import AdapterError, BaseAdapter, build_adapters
from benchmarks.frontier.data import DATASETS, bm25_candidates, load_beir, sample_queries

ALL_SYSTEMS = ["bm25", "jev-latest", "voyage-2.5", "voyage-2.5-lite", "cohere", "bge-reranker-v2-m3-gguf"]


def pctl(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return round(s[idx], 2)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def evaluate(qrels: dict[str, dict[str, int]], runs: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict]:
    import pytrec_eval

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.10"})
    out: dict[str, dict] = {}
    for system, run in runs.items():
        per_query = evaluator.evaluate({q: v for q, v in run.items() if q in qrels})
        ndcg = [m["ndcg_cut_10"] for m in per_query.values()]
        recall = [m["recall_10"] for m in per_query.values()]
        out[system] = {
            "n": len(per_query),
            "ndcg_cut_10_mean": round(statistics.fmean(ndcg), 4) if ndcg else None,
            "ndcg_cut_10_std": round(statistics.pstdev(ndcg), 4) if len(ndcg) > 1 else 0.0,
            "recall_10_mean": round(statistics.fmean(recall), 4) if recall else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--systems", nargs="*", default=ALL_SYSTEMS)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="5 queries on scifact only")
    ap.add_argument("--out", default="benchmarks/results/frontier")
    ap.add_argument("--cohere-min-interval", type=float, default=7.0)
    args = ap.parse_args()

    datasets = ["scifact"] if args.smoke else args.datasets
    n_queries = 5 if args.smoke else args.queries
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"

    import os

    keys = {k: os.environ.get(k, "") for k in ("TYPESAFE_API_KEY", "VOYAGE_API_KEY", "COHERE_API_KEY")}
    needed = {k for k in keys if k == "TYPESAFE_API_KEY" and "jev-latest" in args.systems}
    needed |= {k for k in keys if k == "COHERE_API_KEY" and "cohere" in args.systems}
    needed |= {k for k in keys if k == "VOYAGE_API_KEY" and any(s.startswith("voyage") for s in args.systems)}
    missing = [k for k in sorted(needed) if not keys[k]]
    if missing:
        print(f"FATAL missing env keys: {missing}", flush=True)
        return 2

    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "smoke": args.smoke,
        "seed": args.seed,
        "candidate_depth_k": args.k,
        "queries_per_dataset": n_queries,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": (data_dir / "git_commit.txt").read_text().strip() if (data_dir / "git_commit.txt").exists() else None,
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"=== frontier run (smoke={args.smoke}, datasets={datasets}, n={n_queries}, k={args.k}) ===", flush=True)

    all_metrics: dict[str, dict] = {}
    for dname in datasets:
        corpus, queries, qrels = load_beir(dname, data_dir)
        qids = sample_queries(qrels, n_queries, args.seed)
        cands = bm25_candidates(corpus, {q: queries[q] for q in qids}, args.k, data_dir / f"{dname}_bm25top{args.k}.json")
        print(f"[{dname}] corpus={len(corpus)} sampled_queries={len(qids)}", flush=True)

        for system in args.systems:
            jsonl = out_dir / f"{dname}__{system}.jsonl"
            done = {r["qid"] for r in load_jsonl(jsonl) if "qid" in r and "error" not in r}
            todo = [q for q in qids if q not in done]
            adapter: BaseAdapter | None = None
            t_sys = time.perf_counter()
            for i, qid in enumerate(todo):
                if adapter is None:
                    adapter = build_adapters(
                        [system], keys, cohere_min_interval_s=args.cohere_min_interval
                    )[0]
                try:
                    scores, meta = adapter.rerank(queries[qid], cands[qid], [corpus[d] for d in cands[qid]])
                    row = {"qid": qid, "scores": scores, **{k: v for k, v in meta.items()}}
                except AdapterError as exc:
                    row = {"qid": qid, "error": str(exc)[:300]}
                except Exception as exc:  # keep the run alive; record and continue
                    row = {"qid": qid, "error": f"{type(exc).__name__}: {exc}"[:300]}
                with jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if (i + 1) % 25 == 0 or i + 1 == len(todo):
                    print(f"[{dname}] {system}: {i + 1}/{len(todo)} ({time.perf_counter() - t_sys:.0f}s)", flush=True)
            if adapter is not None:
                adapter.close()

            rows = load_jsonl(jsonl)
            runs: dict[str, dict[str, float]] = {}
            lat: list[float] = []
            errors = 0
            usage: dict[str, float] = {}
            for r in rows:
                if "error" in r:
                    errors += 1
                    continue
                runs[r["qid"]] = r["scores"]
                if isinstance(r.get("latency_ms"), (int, float)):
                    lat.append(float(r["latency_ms"]))
                for k, v in (r.get("usage") or {}).items():
                    usage[k] = usage.get(k, 0) + float(v)
                for extra in ("total_tokens", "search_units", "pairs"):
                    if r.get(extra) is not None:
                        usage[extra] = usage.get(extra, 0) + float(r[extra])
            try:
                m = evaluate(qrels, {system: runs})
                sysm = m.get(system, {})
            except Exception as exc:  # eval deps missing on this host: rows still collected
                print(f"[{dname}] {system}: eval unavailable here ({type(exc).__name__})", flush=True)
                sysm = {}
            lat_list = [float(r["latency_ms"]) for r in rows if isinstance(r.get("latency_ms"), (int, float))]
            summary = {
                **sysm,
                "errors": errors,
                "latency_ms_p50": pctl(lat_list, 50),
                "latency_ms_p95": pctl(lat_list, 95),
                "usage_totals": usage,
            }
            all_metrics.setdefault(dname, {})[system] = summary
            (out_dir / "metrics_partial.json").write_text(
                json.dumps(all_metrics, indent=2), encoding="utf-8"
            )
            print(f"[{dname}] {system}: ndcg@10={summary.get('ndcg_cut_10_mean')} errors={errors}", flush=True)

    results = {"provenance": provenance, "metrics": all_metrics}
    (out_dir / "frontier_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
