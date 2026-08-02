"""
Per-Class HitRate 热力图。
展示各方法在不同 wafer 类别上的 HitRate@1, @5, @10。
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METHOD_FILES = {
    "Proposed (Local SSL)": PROJECT_ROOT / "artifacts/proposed/metrics/7.31/label_metrics_proposed.json",
    "Partial Match (Hand-crafted)": PROJECT_ROOT / "artifacts/partial_match/metrics/7.31_1/label_metrics_partialmatch.json",
    "DenseIR (Global SSL)": PROJECT_ROOT / "artifacts/wafer-denseir/metrics/7.31/label_metrics_denseir.json",
}

CLASS_ORDER = ["center", "donut", "edge-loc", "edge-ring", "loc", "scratch", "near-full", "random"]
K_VALUES = [1, 5, 10]


def load_per_class_data() -> dict[int, np.ndarray]:
    """返回 {K: matrix} 其中 matrix[class_idx, method_idx] = HitRate@K"""
    methods = list(METHOD_FILES.keys())
    result = {}
    for k in K_VALUES:
        result[k] = np.zeros((len(CLASS_ORDER), len(methods)))

    for j, (name, path) in enumerate(METHOD_FILES.items()):
        with open(path) as f:
            raw = json.load(f)
        per_class = raw["per_class"]
        for i, cls in enumerate(CLASS_ORDER):
            for k in K_VALUES:
                result[k][i, j] = per_class[cls][f"HitRate@{k}"]

    return result


def plot_one_heatmap(ax, matrix: np.ndarray, row_labels: list[str],
                     col_labels: list[str], title: str, global_vmin: float):
    cmap = LinearSegmentedColormap.from_list("blue_grad", [
        "#deebf7", "#9ecae1", "#4292c6", "#08519c",
    ])

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=global_vmin, vmax=1.0)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            norm_val = (val - global_vmin) / (1.0 - global_vmin) if global_vmin < 1.0 else 0
            text_color = "white" if norm_val > 0.55 else "#333"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=text_color)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    is_first = ax.get_subplotspec().is_first_col()
    ax.set_yticklabels(row_labels if is_first else [], fontsize=9)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

    return im


def plot_heatmaps(data: dict[int, np.ndarray], row_labels: list[str],
                  col_labels: list[str], out_path: Path):
    # 统一 vmin 取所有 K 中的全局最小值
    global_vmin = min(np.min(m) for m in data.values())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Per-Class HitRate Heatmap", fontsize=14, fontweight="bold", y=1.02)

    for idx, k in enumerate(K_VALUES):
        im = plot_one_heatmap(axes[idx], data[k], row_labels, col_labels,
                              f"HitRate@{k}", global_vmin)

    # 共用 colorbar，用专用 axes 放在右侧，避免遮挡子图
    fig.subplots_adjust(left=0.06, right=0.88, top=0.92, bottom=0.10, wspace=0.22)
    cax = fig.add_axes([0.90, 0.10, 0.015, 0.82])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("HitRate", fontsize=10)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    data = load_per_class_data()
    methods = list(METHOD_FILES.keys())
    out_path = Path(__file__).resolve().parent / "per_class_heatmap.png"
    plot_heatmaps(data, CLASS_ORDER, methods, out_path)


if __name__ == "__main__":
    main()
