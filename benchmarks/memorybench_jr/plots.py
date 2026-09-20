"""Matplotlib plots for the calibration suite (committed PNGs under benchmarks/results/)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_reliability(heads: dict[str, dict[str, Any]], out_path: Path) -> None:
    """Reliability curves (observed frequency vs mean predicted p) per head."""
    fig, axes = plt.subplots(1, len(heads), figsize=(4.2 * len(heads), 3.8), sharey=True)
    if len(heads) == 1:
        axes = [axes]
    for ax, (name, data) in zip(axes, heads.items(), strict=True):
        xs, ys = [], []
        for binrow in data["reliability_curve"]:
            if binrow["mean_predicted"] is not None and binrow["observed_frequency"] is not None:
                xs.append(binrow["mean_predicted"])
                ys.append(binrow["observed_frequency"])
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
        ax.plot(xs, ys, "o-", color="#0a6cff", label=f"Jev (ECE={data['ece']:.3f})")
        ax.set_title(f"{name} (n={data['n']})")
        ax.set_xlabel("mean predicted probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("observed frequency")
    fig.suptitle("MemoryBench-JR reliability curves (live Jev run)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_accuracy_vs_threshold(heads: dict[str, dict[str, Any]], out_path: Path) -> None:
    """Accuracy and coverage vs decision threshold for the binary heads."""
    fig, axes = plt.subplots(1, len(heads), figsize=(4.2 * len(heads), 3.8), sharey=True)
    if len(heads) == 1:
        axes = [axes]
    for ax, (name, data) in zip(axes, heads.items(), strict=True):
        rows = data["threshold_table"]
        ts = [r["threshold"] for r in rows]
        acc = [r["accuracy"] for r in rows]
        cov = [r["coverage"] for r in rows]
        ax.plot(ts, acc, "o-", color="#0a6cff", label="accuracy")
        ax.plot(ts, cov, "s--", color="#e07b00", label="coverage")
        ax.set_title(name)
        ax.set_xlabel("decision threshold t (predict positive if p >= t)")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("fraction")
    fig.suptitle("Accuracy vs confidence threshold (live Jev run)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
