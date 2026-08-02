"""
实验B：Preference Accuracy by Rule 排序柱状图 (横向)。
展示三种方法在 16 条偏好规则上的表现，按 Local (Learned) accuracy 降序。
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

RULE_LABELS = {
    "identity_over_cluster_dropout": "identity > cluster_dropout",
    "cluster_extra_over_easy_negative": "cluster_extra > easy_neg",
    "identity_over_cluster_extra": "identity > cluster_extra",
    "mild_noise_over_easy_negative": "mild_noise > easy_neg",
    "mild_dropout_over_easy_negative": "mild_dropout > easy_neg",
    "rot_90_over_easy_negative": "rot_90 > easy_neg",
    "mild_dropout_over_strong_dropout": "mild_dropout > strong_dropout",
    "rot_90_over_same_position_wrong_shape": "rot_90 > wrong_shape",
    "mild_noise_over_strong_noise": "mild_noise > strong_noise",
    "mild_shift_over_same_area_wrong_shape": "mild_shift > wrong_shape",
    "rot_180_over_easy_negative": "rot_180 > easy_neg",
    "rot_180_over_same_position_wrong_shape": "rot_180 > wrong_shape",
    "mild_shift_over_strong_shift": "mild_shift > strong_shift",
    "same_label_hard_negative_over_diff_label_spatially_close": "same_label_hard > diff_label_close",
    "scale_mild_over_easy_negative": "scale_mild > easy_neg",
    "cluster_dropout_over_easy_negative": "cluster_dropout > easy_neg",
}


def load_data() -> dict:
    data = {}
    for name, path in METHOD_FILES.items():
        with open(path) as f:
            raw = json.load(f)
        data[name] = raw["by_rule"]
    return data


def plot(data: dict, out_path: Path):
    methods = list(data.keys())
    n_methods = len(methods)

    # 取第一条记录的 rules 列表，然后按 Local (Learned) 降序
    all_rules = list(data[methods[0]].keys())
    ref_method = methods[0]
    all_rules.sort(key=lambda r: data[ref_method][r]["preference_accuracy"], reverse=True)

    n_rules = len(all_rules)

    fig, ax = plt.subplots(figsize=(10, 7))

    y = np.arange(n_rules)
    bar_height = 0.22
    offsets = np.linspace(-bar_height, bar_height, n_methods)

    for i, method in enumerate(methods):
        values = [data[method][r]["preference_accuracy"] for r in all_rules]
        bars = ax.barh(y + offsets[i], values, bar_height, color=COLORS[i],
                       label=method, edgecolor="white", linewidth=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=6.2)

    ax.set_yticks(y)
    ax.set_yticklabels([RULE_LABELS.get(r, r) for r in all_rules], fontsize=7.5)
    ax.set_xlabel("Preference Accuracy", fontsize=11)
    ax.set_xlim(0.40, 1.08)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    data = load_data()
    out_path = Path(__file__).resolve().parent / "preference_by_rule.png"
    plot(data, out_path)


if __name__ == "__main__":
    main()
