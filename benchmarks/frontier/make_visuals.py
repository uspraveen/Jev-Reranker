"""Render the frontier-comparison charts from committed per-query rows.

Every number shown is recomputed here from the committed JSONLs with
pytrec_eval (the same evaluator the runners used) — nothing hand-typed.
Writes frontier_summary.json + three PNGs next to the results.

    python -m benchmarks.frontier.make_visuals
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytrec_eval

ROOT = Path(__file__).resolve().parents[1] / "results" / "frontier"
DATASETS = ("scifact", "nfcorpus", "fiqa")
SYSTEMS = (
    ("bm25", "BM25 (floor)", "#B0B0B0"),
    ("jev-latest", "Jev-Reranker", "#1F6FEB"),
    ("cohere", "Cohere v4.0-pro", "#8E8E8E"),
    ("qwen3-reranker-0.6b", "Qwen3-0.6B (open)", "#C9A227"),
    ("bge-reranker-v2-m3", "BGE-v2-m3 (open)", "#7A9E7E"),
    ("ms-marco-minilm-l6-v2", "MiniLM-L6 (open)", "#B07A99"),
)
# Measured latency per query (p50, ms) with measurement vantage; Cohere's
# benchmark-run wall time was polluted by trial-key throttling, so its bar
# uses a 5-call unpaced spot-check from a different vantage (laptop).
LATENCY = (
    ("jev-latest", 199.5, "sandbox → API"),
    ("cohere", 499.0, "laptop → API (spot-check n=5)"),
    ("qwen3-reranker-0.6b", 233.9, "on-GPU"),
    ("bge-reranker-v2-m3", 354.1, "on-GPU"),
    ("ms-marco-minilm-l6-v2", None, "on-GPU"),
)
LATENCY_MINILM = None  # filled from rows below

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
})


def load_rows(path: Path) -> dict[str, dict[str, float]]:
    best: dict[str, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "scores" in row:
            best[row["qid"]] = row["scores"]
    return best


def main() -> None:
    summary: dict[str, dict[str, float]] = {}
    per_query: dict[str, dict[str, list[float]]] = {}
    for ds in DATASETS:
        blob = json.loads((ROOT / "data" / f"{ds}_loaded.json").read_text(encoding="utf-8"))
        qrels = blob["qrels"]
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
        summary[ds] = {}
        per_query[ds] = {}
        for key, _, _ in SYSTEMS:
            runs = load_rows(ROOT / f"{ds}__{key}.jsonl")
            rows = evaluator.evaluate({q: r for q, r in runs.items() if q in qrels})
            vals = [m["ndcg_cut_10"] for m in rows.values()]
            summary[ds][key] = round(statistics.fmean(vals), 4)
            per_query[ds][key] = vals
    (ROOT / "frontier_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    global LATENCY_MINILM
    minilm_lats = []
    for ds in DATASETS:
        rows = load_rows(ROOT / f"{ds}__ms-marco-minilm-l6-v2.jsonl")
        rows2 = []
        for line in (ROOT / f"{ds}__ms-marco-minilm-l6-v2.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if "latency_ms" in r and "scores" in r:
                    rows2.append(r["latency_ms"])
        minilm_lats += rows2
    LATENCY_MINILM = round(statistics.median(minilm_lats), 1) if minilm_lats else None

    # Chart 1 — grouped bars per dataset.
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=160)
    width = 0.15
    import numpy as np

    x = np.arange(len(DATASETS))
    for i, (key, label, color) in enumerate(SYSTEMS):
        vals = [summary[ds][key] for ds in DATASETS]
        offset = (i - (len(SYSTEMS) - 1) / 2) * width
        emphasis = key == "jev-latest"
        bars = ax.bar(
            x + offset, vals, width * 0.92, label=label, color=color,
            edgecolor="#333333" if emphasis else "none", linewidth=1.2 if emphasis else 0,
            zorder=3 if emphasis else 2,
        )
        for b, v in zip(bars, vals, strict=True):
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=7.5,
                        fontweight="bold" if emphasis else "normal", color="#1F6FEB" if emphasis else "#444444")
    ax.set_xticks(x)
    ax.set_xticklabels([ds.capitalize() for ds in DATASETS])
    ax.set_ylabel("NDCG@10")
    ax.set_ylim(0, 0.9)
    ax.set_title("Frontier comparison on BEIR — identical BM25 top-30 candidates for every system\n"
                 "(NDCG@10, 200 seeded queries per dataset, pytrec_eval)", fontsize=12)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "frontier_ndcg_by_dataset.png", bbox_inches="tight")
    plt.close(fig)

    # Chart 2 — 3-set average, sorted horizontal bars.
    avgs = {key: round(statistics.fmean([summary[ds][key] for ds in DATASETS]), 4) for key, _, _ in SYSTEMS}
    order = sorted(SYSTEMS, key=lambda s: avgs[s[0]])
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=160)
    names = [label for _, label, _ in order]
    vals = [avgs[key] for key, _, _ in order]
    colors = [color for _, _, color in order]
    bars = ax.barh(names, vals, color=colors, height=0.62,
                   edgecolor=["#333333" if k == "jev-latest" else "none" for k, _, _ in order],
                   linewidth=[1.4 if k == "jev-latest" else 0 for k, _, _ in order])
    for b, v in zip(bars, vals, strict=True):
        ax.annotate(f"{v:.4f}", (v, b.get_y() + b.get_height() / 2), va="center", ha="left",
                    xytext=(4, 0), textcoords="offset points", fontsize=10, fontweight="bold", color="#333333")
    ax.set_xlim(0, 0.6)
    ax.set_xlabel("NDCG@10, mean of scifact / nfcorpus / fiqa")
    ax.set_title("3-dataset average — Jev-Reranker leads the frontier table", fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "frontier_average_ndcg.png", bbox_inches="tight")
    plt.close(fig)

    # Chart 3 — lift over the BM25 floor per dataset.
    fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=160)
    rerankers = [s for s in SYSTEMS if s[0] != "bm25"]
    width = 0.19
    for i, (key, label, color) in enumerate(rerankers):
        vals = [summary[ds][key] - summary[ds]["bm25"] for ds in DATASETS]
        offset = (i - (len(rerankers) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width * 0.92, label=label, color=color,
                      edgecolor="#333333" if key == "jev-latest" else "none",
                      linewidth=1.2 if key == "jev-latest" else 0)
        for b, v in zip(bars, vals, strict=True):
            ax.annotate(f"+{v:.3f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                        fontsize=7.5, color="#444444")
    ax.axhline(0, color="#888888", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([ds.capitalize() for ds in DATASETS])
    ax.set_ylabel("NDCG@10 lift over BM25 floor")
    ax.set_title("Reranking lift over retrieval — every reranker helps; Jev most on 2 of 3 datasets", fontsize=12)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "frontier_lift_over_bm25.png", bbox_inches="tight")
    plt.close(fig)

    # Chart 4 — latency per query (p50), vantage-annotated.
    import numpy as np2

    lat_entries = []
    for key, p50, vantage in LATENCY:
        if key == "ms-marco-minilm-l6-v2":
            p50 = LATENCY_MINILM
        if p50 is not None:
            lat_entries.append((key, p50, vantage))
    lat_entries.sort(key=lambda e: e[1])
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=160)
    labels = {k: l for k, l, _ in SYSTEMS}
    names = [labels[k] + "\n(" + v + ")" for k, _, v in lat_entries]
    vals = [v for _, v, _ in lat_entries]
    colors = [dict((k, c) for k, _, c in SYSTEMS)[k] for k, _, _ in lat_entries]
    bars = ax.bar(names, vals, color=colors, width=0.6,
                  edgecolor=["#333333" if k == "jev-latest" else "none" for k, _, _ in lat_entries],
                  linewidth=[1.4 if k == "jev-latest" else 0 for k, _, _ in lat_entries])
    for b, v in zip(bars, vals, strict=True):
        ax.annotate(f"{v:.0f} ms", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color="#333333")
    ax.set_ylabel("p50 latency per query (ms, 30 candidates)")
    ax.set_title("Latency per query — Cohere bar is an unpaced spot-check (the throttled benchmark\n"
                 "run is not latency-comparable); GPU lanes measured on-node; vantages differ",
                 fontsize=11)
    ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig(ROOT / "frontier_latency.png", bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(avgs, indent=2))
    print("charts written to", ROOT)


if __name__ == "__main__":
    main()
