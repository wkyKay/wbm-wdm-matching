"""Plot summary figures for the three Mixed38K retrieval experiments.

Run from the repository root with::

    python match/experiments/analysis/plot_retrieval_metrics.py

Figures are written to ``match/experiments/analysis/figures`` by default.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "match" / "experiments" / "artifacts"
DEFAULT_OUT = Path(__file__).resolve().parent / "figures"

EXPERIMENTS = {
    "same_label": {
        "label": "Same-label",
        "metrics": ARTIFACTS / "mixed38k_same_label_retrieval_100_geometry" / "metrics.json",
        "trials": ARTIFACTS / "mixed38k_same_label_retrieval_100_geometry" / "trials.csv",
        "rank_field": "best_target_rank",
        "class_metric": "top5_target_hit_rate",
    },
    "single_to_multi": {
        "label": "Single-to-multi",
        "metrics": ARTIFACTS / "mixed38k_single_to_multi_retrieval_100" / "metrics.json",
        "trials": ARTIFACTS / "mixed38k_single_to_multi_retrieval_100" / "trials.csv",
        "rank_field": "first_positive_rank",
        "class_metric": "hit_at_5",
    },
    "transform": {
        "label": "Transform",
        "metrics": ARTIFACTS / "mixed38k_transform_retrieval_100" / "metrics.json",
        "trials": ARTIFACTS / "mixed38k_transform_retrieval_100" / "trials.csv",
        "rank_field": "best_target_rank",
        "class_metric": "top5_target_hit_rate",
    },
}

CLASSES = ("center", "donut", "edge-loc", "edge-ring", "loc", "random", "scratch", "near-full")
COLORS = {"same_label": "#2563eb", "single_to_multi": "#dc2626", "transform": "#16a34a"}


def _load() -> dict[str, dict[str, Any]]:
    loaded = {}
    for key, spec in EXPERIMENTS.items():
        if not spec["metrics"].is_file():
            raise FileNotFoundError(f"Missing metrics file: {spec['metrics']}")
        with spec["metrics"].open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        loaded[key] = {**spec, "data": metrics}
    return loaded


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig: plt.Figure, output: Path, title: str) -> None:
    fig.suptitle(title, y=0.99, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_overall(data: dict[str, dict[str, Any]], out: Path) -> None:
    metrics = ("hit_at_1", "hit_at_5", "hit_at_10", "mrr_target")
    labels = ("Hit@1", "Hit@5", "Hit@10", "MRR")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (key, spec) in zip(axes, data.items()):
        overall = spec["data"]["overall"]
        values = [overall.get(metric, overall.get(metric.replace("_target", ""), 0.0)) for metric in metrics]
        bars = ax.bar(labels, values, color=COLORS[key], alpha=0.88, width=0.68)
        ax.set_title(spec["label"])
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Rate / reciprocal rank" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=30)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.1%}", ha="center", va="bottom", fontsize=9)
    _save(fig, out / "overall_metrics.png", "Overall retrieval performance")


def plot_class_heatmap(data: dict[str, dict[str, Any]], out: Path) -> None:
    matrix = np.full((len(CLASSES), len(data)), np.nan, dtype=float)
    for col, (key, spec) in enumerate(data.items()):
        for row, class_name in enumerate(CLASSES):
            entry = spec["data"].get("per_class", {}).get(class_name)
            if entry is not None:
                matrix[row, col] = float(entry[spec["class_metric"]])
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(data)), [spec["label"] for spec in data.values()])
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Query class")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.0%}", ha="center", va="center", color="black" if value < 0.68 else "white", fontweight="bold")
            else:
                ax.text(col, row, "N/A", ha="center", va="center", color="#777777")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    _save(fig, out / "per_class_hit5_heatmap.png", "Per-class Hit@5")


def plot_transform_breakdown(data: dict[str, dict[str, Any]], out: Path) -> None:
    transform = data["transform"]["data"].get("per_transform", {})
    names = list(transform)
    hit5 = [float(transform[name]["hit_at_5"]) for name in names]
    hit10 = [float(transform[name]["hit_at_10"]) for name in names]
    mean_rank = [float(transform[name]["mean_rank"]) for name in names]
    order = np.argsort(hit5)
    names = [names[i] for i in order]
    hit5 = [hit5[i] for i in order]
    hit10 = [hit10[i] for i in order]
    mean_rank = [mean_rank[i] for i in order]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    y = np.arange(len(names))
    axes[0].barh(y + 0.18, hit5, height=0.34, label="Hit@5", color="#16a34a")
    axes[0].barh(y - 0.18, hit10, height=0.34, label="Hit@10", color="#86efac")
    axes[0].set_yticks(y, names)
    axes[0].set_xlim(0, 1.05)
    axes[0].set_xlabel("Target-level hit rate")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="x", alpha=0.25)
    axes[1].barh(y, mean_rank, color="#f59e0b", alpha=0.9)
    axes[1].set_yticks(y, names)
    axes[1].set_xlabel("Mean target rank (lower is better)")
    axes[1].grid(axis="x", alpha=0.25)
    for i, value in enumerate(hit5):
        axes[0].text(value + 0.02, i + 0.18, f"{value:.0%}", va="center", fontsize=8)
    _save(fig, out / "transform_breakdown.png", "Transform robustness")


def _read_ranks(spec: dict[str, Any]) -> tuple[np.ndarray, dict[str, list[float]]]:
    rows = []
    by_class: dict[str, list[float]] = {}
    with spec["trials"].open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = float(row[spec["rank_field"]])
            rows.append(value)
            by_class.setdefault(row["query_class"], []).append(value)
    return np.asarray(rows, dtype=float), by_class


def plot_rank_cdf(data: dict[str, dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for key, spec in data.items():
        ranks, _ = _read_ranks(spec)
        x = np.sort(ranks)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", label=spec["label"], color=COLORS[key], linewidth=2)
    ax.set_xlim(1, 30)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Rank of first positive target")
    ax.set_ylabel("Fraction of queries")
    ax.set_xticks([1, 3, 5, 10, 20, 30])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, out / "first_positive_rank_cdf.png", "First-positive rank distribution")


def plot_class_rank_boxplot(data: dict[str, dict[str, Any]], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    for ax, (key, spec) in zip(axes, data.items()):
        _, by_class = _read_ranks(spec)
        names = [name for name in CLASSES if name in by_class]
        values = [by_class[name] for name in names]
        positions = [CLASSES.index(name) + 1 for name in names]
        ax.boxplot(
            values,
            positions=positions,
            vert=False,
            patch_artist=True,
            boxprops={"facecolor": COLORS[key], "alpha": 0.55},
        )
        ax.set_title(spec["label"])
        ax.set_xlabel("First positive rank")
        ax.set_xlim(0, 100)
        ax.set_ylim(0.5, len(CLASSES) + 0.5)
        ax.set_yticks(range(1, len(CLASSES) + 1), CLASSES)
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="y", labelsize=9)
    axes[0].set_ylabel("Query class")
    _save(fig, out / "per_class_rank_boxplot.png", "Per-class first-positive rank")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Directory for generated PNG figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = _load()
    plot_overall(data, args.out_dir)
    plot_class_heatmap(data, args.out_dir)
    plot_transform_breakdown(data, args.out_dir)
    plot_rank_cdf(data, args.out_dir)
    plot_class_rank_boxplot(data, args.out_dir)
    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
