"""
跨方法检索性能对比柱状图 (1×3)。
对比 Hand-crafted Desc. / Global SSL / Local SSL 在
LabelNDCG / ExactHitRate / MeanJaccard 上的表现 (@1, @5, @10)。
"""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METHOD_FILES = {
    "Proposed (Local SSL)": PROJECT_ROOT / "artifacts/proposed/metrics/7.31/label_metrics_proposed.json",
    "Partial Match (Hand-crafted)": PROJECT_ROOT / "artifacts/partial_match/metrics/7.31_1/label_metrics_partialmatch.json",
    "DenseIR (Global SSL)": PROJECT_ROOT / "artifacts/wafer-denseir/metrics/7.31/label_metrics_denseir.json",
}

METRICS = ["LabelNDCG", "ExactHitRate", "MeanJaccard"]
K_VALUES = [1, 5, 10]
COLORS = ["#4C72B0", "#DD8452", "#55A868"]

# 统一 Y 轴范围：起点 50%，上保留充足空间标注数值
Y_LIM = (0.50, 1.08)

# retrieval key 映射: ExactHitRate 读取 ExactRate 字段
METRIC_KEY_MAP = {
    "LabelNDCG": "LabelNDCG",
    "ExactHitRate": "ExactRate",
    "MeanJaccard": "MeanJaccard",
}


def load_data() -> dict[str, dict]:
    data = {}
    for name, path in METHOD_FILES.items():
        with open(path) as f:
            raw = json.load(f)
        retrieval = raw["retrieval"]
        entry = {}
        for metric in METRICS:
            key = METRIC_KEY_MAP[metric]
            for k in K_VALUES:
                entry[f"{metric}@{k}"] = retrieval[f"{key}@{k}"]
        data[name] = entry
    return data


def plot_comparison(data: dict[str, dict], out_path: Path):
    methods = list(data.keys())
    n_methods = len(methods)
    n_k = len(K_VALUES)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Cross-Method Retrieval Performance Comparison", fontsize=14, fontweight="bold", y=1.01)

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        x = np.arange(n_k)
        bar_width = 0.25
        offsets = np.linspace(-bar_width, bar_width, n_methods)

        for i, method in enumerate(methods):
            values = [data[method][f"{metric}@{k}"] for k in K_VALUES]
            bars = ax.bar(x + offsets[i], values, bar_width, color=COLORS[i],
                          label=method, edgecolor="white", linewidth=0.5)

            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_title(f"{metric} @ K", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"@{k}" for k in K_VALUES], fontsize=10)
        ax.set_ylim(*Y_LIM)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.grid(axis="y", alpha=0.3)

    # 共用图例
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    data = load_data()
    out_path = Path(__file__).resolve().parent / "retrieval_comparison.png"
    plot_comparison(data, out_path)


if __name__ == "__main__":
    main()
