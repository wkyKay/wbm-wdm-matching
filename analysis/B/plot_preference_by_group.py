"""
实验B：Preference Accuracy by Transformation Group 分组柱状图。
展示三种方法在 8 种变换类别上的偏好一致性。
"""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METHOD_FILES = {
    "Proposed (Local SSL)": PROJECT_ROOT / "artifacts/proposed/metrics/7.31/b_proposed_preference_metrics.json",
    "Partial Match (Hand-crafted)": PROJECT_ROOT / "artifacts/partial_match/metrics/7.31_1/b_partialmatch_preference_metrics_1.json",
    "DenseIR (Global SSL)": PROJECT_ROOT / "artifacts/wafer-denseir/metrics/7.31/b_waferdesnir_preference_metrics.json",
}

COLORS = ["#55A868", "#4C72B0", "#DD8452"]
GROUP_LABELS = {
    "noise": "Noise", "dropout": "Dropout", "rotation": "Rotation",
    "shift": "Shift", "scale": "Scale", "cluster_dropout": "Cluster\nDropout",
    "cluster_extra": "Cluster\nExtra", "hard_negative": "Hard Neg.",
}


def load_data():
    data = {}
    for name, path in METHOD_FILES.items():
        with open(path) as f:
            raw = json.load(f)
        data[name] = raw["by_group"]
    return data


def plot(data: dict, out_path: Path):
    groups = list(next(iter(data.values())).keys())
    methods = list(data.keys())
    n_groups = len(groups)
    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=(14, 5.5))

    x = np.arange(n_groups)
    bar_width = 0.25
    offsets = np.linspace(-bar_width, bar_width, n_methods)

    for i, method in enumerate(methods):
        values = [data[method][g]["preference_accuracy"] for g in groups]
        bars = ax.bar(x + offsets[i], values, bar_width, color=COLORS[i],
                      label=method, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABELS.get(g, g) for g in groups], fontsize=9)
    ax.set_ylabel("Preference Accuracy", fontsize=11)
    ax.set_ylim(0.50, 1.08)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="y", alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    data = load_data()
    out_path = Path(__file__).resolve().parent / "preference_by_group.png"
    plot(data, out_path)


if __name__ == "__main__":
    main()
