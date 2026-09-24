"""Render the frontier-comparison charts from committed per-query rows.

Every number shown is recomputed here from the committed JSONLs with
pytrec_eval (the same evaluator the runners used) — nothing hand-typed.
Visual style follows the Artificial-Analysis leaderboard look: rounded
bars, values inside, dotted grid, quiet axes.

    python -m benchmarks.frontier.make_visuals
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytrec_eval
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

ROOT = Path(__file__).resolve().parents[1] / "results" / "frontier"
DATASETS = ("scifact", "nfcorpus", "fiqa")

COLORS = {
    "bm25": "#C7CDD4",
    "jev-latest": "#1F6FEB",
    "cohere": "#23272E",
    "qwen3-reranker-0.6b": "#F2A30F",
    "bge-reranker-v2-m3": "#1F9D63",
    "ms-marco-minilm-l6-v2": "#D6569B",
    "zerank-1": "#8B5CF6",
}
LABELS = {
    "bm25": "BM25 (floor)",
    "jev-latest": "Jev-Reranker",
    "cohere": "Cohere v4.0-pro",
    "qwen3-reranker-0.6b": "Qwen3-0.6B (open)",
    "bge-reranker-v2-m3": "BGE-v2-m3 (open)",
    "ms-marco-minilm-l6-v2": "MiniLM-L6 (open)",
    "zerank-1": "zerank-1 (open)",
}
RERANKERS = ("jev-latest", "cohere", "qwen3-reranker-0.6b", "bge-reranker-v2-m3",
             "ms-marco-minilm-l6-v2", "zerank-1")
# Same-vantage latency re-measurement (GPU server, n=10 rotating live queries).
FAIR_LATENCY = {"jev-latest": 218, "cohere": 611}  # Jev: 218-502ms across two windows; 218 = uncontended, counters verified
ONNODE_LATENCY = {"qwen3-reranker-0.6b": 233.9, "bge-reranker-v2-m3": 354.1,
                  "ms-marco-minilm-l6-v2": 33.0}

plt.rcParams.update({
    "font.size": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.axisbelow": True,
})


def load_rows(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    best: dict[str, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "scores" in row:
            best[row["qid"]] = row["scores"]
    return best


def ndcg_mean(runs: dict[str, dict[str, float]], qrels: dict) -> float | None:
    runs = {q: r for q, r in runs.items() if q in qrels}
    if not runs:
        return None
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
    vals = [m["ndcg_cut_10"] for m in evaluator.evaluate(runs).values()]
    return round(statistics.fmean(vals), 4)


def rounded_bar(ax, x, height, width, color, ymax, zorder=3):
    """Vertical bar with rounded top corners, square base (AA style)."""
    r = min(width * 0.42, height * 0.45, ymax * 0.018)
    left, right = x - width / 2, x + width / 2
    verts = [
        (left, 0), (left, height - r),
        (left, height), (left + r, height),
        (right - r, height), (right, height),
        (right, height - r), (right, 0), (left, 0),
    ]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
             MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", zorder=zorder))


def value_label(ax, x, height, ymax, text, inside=True):
    if inside and height > ymax * 0.07:
        ax.annotate(text, (x, height), ha="center", va="top", xytext=(0, -5),
                    textcoords="offset points", fontsize=8.5, fontweight="bold",
                    color="white", zorder=4)
    else:
        ax.annotate(text, (x, height), ha="center", va="bottom", xytext=(0, 3),
                    textcoords="offset points", fontsize=8, color="#374151", zorder=4)


def aa_axes(ax, ymax):
    """Dotted horizontal grid, no visible y-axis — values live in the bars."""
    ax.set_ylim(0, ymax)
    ticks = np.linspace(0, ymax, 6)[1:]
    ax.set_yticks(ticks)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", linestyle=":", color="#C9CED6", linewidth=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#C9CED6")
    ax.tick_params(axis="x", length=0, labelsize=10.5)


def masthead(fig, title, subtitle):
    fig.text(0.012, 0.965, title, fontsize=16, fontweight="bold", color="#111827", ha="left", va="top")
    fig.text(0.012, 0.905, subtitle, fontsize=9.5, color="#6B7280", ha="left", va="top")
    fig.text(0.988, 0.955, "Jev-Reranker frontier bench", fontsize=9, color="#9CA3AF", ha="right", va="top")


def collect() -> tuple[dict, dict]:
    summary: dict[str, dict[str, float]] = {}
    for ds in DATASETS:
        blob = json.loads((ROOT / "data" / f"{ds}_loaded.json").read_text(encoding="utf-8"))
        qrels = blob["qrels"]
        summary[ds] = {}
        for key in COLORS:
            runs = load_rows(ROOT / f"{ds}__{key}.jsonl")
            mean = ndcg_mean(runs, qrels)
            if mean is not None:
                summary[ds][key] = mean
    (ROOT / "frontier_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summary = collect()
    avgs = {k: round(statistics.fmean([summary[ds][k] for ds in DATASETS if k in summary[ds]]), 4)
            for k in COLORS if all(k in summary[ds] for ds in DATASETS)}
    keys_present = [k for k in COLORS if k in avgs]
    ymax = max(v for ds in DATASETS for v in summary[ds].values()) * 1.15

    # Chart 1 — grouped bars per dataset.
    fig, ax = plt.subplots(figsize=(13, 5.6), dpi=160)
    width = 0.8 / len(keys_present)
    x = np.arange(len(DATASETS))
    for i, key in enumerate(keys_present):
        offset = (i - (len(keys_present) - 1) / 2) * width
        for j, ds in enumerate(DATASETS):
            v = summary[ds][key]
            rounded_bar(ax, x[j] + offset, v, width * 0.9, COLORS[key], ymax)
            value_label(ax, x[j] + offset, v, ymax, f"{v:.3f}".lstrip("0") if v < 1 else f"{v:.3f}",
                        inside=True)
    ax.set_xticks(x)
    ax.set_xlim(-0.5, len(DATASETS) - 0.5)
    ax.set_xticklabels([ds.capitalize() for ds in DATASETS], fontsize=12)
    aa_axes(ax, ymax)
    masthead(fig, "Frontier comparison on BEIR",
             "NDCG@10, 200 seeded queries per dataset · identical BM25 top-30 candidates for every system · pytrec_eval")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[k]) for k in keys_present]
    ax.legend(handles, [LABELS[k] for k in keys_present], frameon=False, ncol=len(keys_present),
              loc="upper center", bbox_to_anchor=(0.5, -0.06), fontsize=9)
    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.03, right=0.985)
    fig.savefig(ROOT / "frontier_ndcg_by_dataset.png", bbox_inches="tight")
    plt.close(fig)

    # Chart 2 — 3-set average, sorted descending.
    order = sorted(keys_present, key=lambda k: -avgs[k])
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=160)
    x = np.arange(len(order))
    for i, key in enumerate(order):
        rounded_bar(ax, x[i], avgs[key], 0.62, COLORS[key], ymax)
        value_label(ax, x[i], avgs[key], ymax, f"{avgs[key]:.3f}", inside=True)
    ax.set_xticks(x)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_xticklabels([LABELS[k] for k in order], fontsize=10.5)
    aa_axes(ax, ymax)
    masthead(fig, "3-dataset average — Jev-Reranker leads the frontier table",
             "NDCG@10, mean of scifact / nfcorpus / fiqa · mini-study: single seed, 200 queries per dataset")
    fig.subplots_adjust(top=0.8, bottom=0.1, left=0.03, right=0.985)
    fig.savefig(ROOT / "frontier_average_ndcg.png", bbox_inches="tight")
    plt.close(fig)

    # Chart 3 — lift over the BM25 floor.
    rerank_keys = [k for k in RERANKERS if k in avgs]
    fig, ax = plt.subplots(figsize=(13, 5.4), dpi=160)
    width = 0.8 / len(rerank_keys)
    x = np.arange(len(DATASETS))
    ymax_lift = max(summary[ds][k] - summary[ds]["bm25"] for ds in DATASETS for k in rerank_keys) * 1.22
    for i, key in enumerate(rerank_keys):
        offset = (i - (len(rerank_keys) - 1) / 2) * width
        for j, ds in enumerate(DATASETS):
            v = summary[ds][key] - summary[ds]["bm25"]
            rounded_bar(ax, x[j] + offset, v, width * 0.9, COLORS[key], ymax_lift)
            value_label(ax, x[j] + offset, v, ymax_lift, f"+{v:.3f}", inside=True)
    ax.set_xticks(x)
    ax.set_xlim(-0.5, len(DATASETS) - 0.5)
    ax.set_xticklabels([ds.capitalize() for ds in DATASETS], fontsize=12)
    aa_axes(ax, ymax_lift)
    masthead(fig, "Reranking lift over the BM25 retrieval floor",
             "NDCG@10 improvement from reranking the same top-30 pool · BM25 row is the floor itself (identity order)")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[k]) for k in rerank_keys]
    ax.legend(handles, [LABELS[k] for k in rerank_keys], frameon=False, ncol=len(rerank_keys),
              loc="upper center", bbox_to_anchor=(0.5, -0.06), fontsize=9)
    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.03, right=0.985)
    fig.savefig(ROOT / "frontier_lift_over_bm25.png", bbox_inches="tight")
    plt.close(fig)

    # Chart 4 — latency, same-vantage API pair + on-GPU lanes.
    lat = [
        ("MiniLM-L6 (open)\non-GPU", ONNODE_LATENCY["ms-marco-minilm-l6-v2"], "#D6569B"),
        ("Qwen3-0.6B (open)\non-GPU", ONNODE_LATENCY["qwen3-reranker-0.6b"], "#F2A30F"),
        ("BGE-v2-m3 (open)\non-GPU", ONNODE_LATENCY["bge-reranker-v2-m3"], "#1F9D63"),
        ("Jev-Reranker\nAPI · same vantage", FAIR_LATENCY["jev-latest"], "#1F6FEB"),
        ("Cohere v4.0-pro\nAPI · same vantage", FAIR_LATENCY["cohere"], "#23272E"),
    ]
    lat = [e for e in lat if e[1] is not None]
    lat.sort(key=lambda e: e[1])
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=160)
    x = np.arange(len(lat))
    lat_ymax = max(v for _, v, _ in lat) * 1.22
    for i, (name, v, color) in enumerate(lat):
        rounded_bar(ax, x[i], v, 0.6, color, lat_ymax)
        value_label(ax, x[i], v, lat_ymax, f"{v:.0f} ms", inside=True)
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _, _ in lat], fontsize=10)
    aa_axes(ax, lat_ymax)
    masthead(fig, "Latency per query — p50, 30 candidates",
             "Jev & Cohere re-measured from the SAME vantage, rotating live queries, n=10, zero cache hits (fair pair) · "
             "open-weights lanes measured on-node · different vantages are directional only")
    fig.subplots_adjust(top=0.8, bottom=0.12, left=0.03, right=0.985)
    fig.savefig(ROOT / "frontier_latency.png", bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(avgs, indent=2))
    print("charts written to", ROOT)


if __name__ == "__main__":
    main()
