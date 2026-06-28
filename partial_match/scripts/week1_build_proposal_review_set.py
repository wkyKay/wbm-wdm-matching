# -*- coding: utf-8 -*-
"""
Build a class-balanced manual review set for Week 1 proposal evaluation.

The script samples 10-20 WM38K maps per defect class, runs several proposal
methods, and writes:
  - selected_samples.csv: sampled maps and labels
  - proposal_metrics.csv: automatic proposal metrics
  - manual_review_template.csv: fields for human visual judgement
  - figures/*.png: class-wise visual comparison panels
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from partial_match.core.clustering import cluster
from partial_match.data.data_io import CLASS_NAMES, filter_valid_samples, load_wm38k


DEFAULT_METHODS = [
    "raw",
    "filtered",
    "adhesion",
    "dilated_group",
    "dilated_adhesion",
    "group_then_adhesion",
    "geometry_merge",
    "topk",
    "topk_dilated",
]

METHOD_TITLES = {
    "raw": "Raw CC",
    "filtered": "Filtered",
    "adhesion": "Adhesion",
    "dilated_group": "Dilated Group",
    "dilated_adhesion": "Dilated + Adhesion",
    "group_then_adhesion": "Filter + Group + Adhesion",
    "geometry_merge": "Geometry Merge",
    "topk": "TopK",
    "topk_dilated": "TopK Dilated",
}


def _label_names(label_vec: np.ndarray) -> str:
    return "|".join(name for name, flag in zip(CLASS_NAMES, label_vec) if int(flag) == 1)


def _cluster_pixels(clusters: Sequence[Dict]) -> set:
    if clusters is None:
        return set()
    pixels = set()
    for item in clusters:
        coords = item.get("pixels", item.get("pixel_coords", []))
        for coord in coords:
            if isinstance(coord, dict):
                pixels.add((int(coord["row"]), int(coord["col"])))
            else:
                pixels.add((int(coord[0]), int(coord[1])))
    return pixels


def _build_overlay(clusters: Sequence[Dict], raw_map: np.ndarray) -> np.ndarray:
    overlay = np.zeros(raw_map.shape, dtype=float)
    valid_mask = (raw_map == 1) | (raw_map == 2)
    overlay[valid_mask] = 0.05
    if clusters is None:
        return overlay
    for idx, item in enumerate(clusters):
        coords = item.get("pixels", item.get("pixel_coords", []))
        value = 0.18 + (idx % 18) * 0.045
        for coord in coords:
            if isinstance(coord, dict):
                r, c = int(coord["row"]), int(coord["col"])
            else:
                r, c = int(coord[0]), int(coord[1])
            overlay[r, c] = value
    return overlay


def _proposal_kwargs(method: str, args: argparse.Namespace) -> Dict:
    kwargs = {
        "min_area": args.min_area,
        "top_k": args.top_k,
        "dilation_radius": args.dilation_radius,
        "use_closing": args.use_closing_for_grouping,
        "suspicious_area": args.suspicious_area,
        "min_suspicious_cues": args.min_suspicious_cues,
        "max_split_count": args.max_split_count,
        "min_split_coverage": args.min_split_coverage,
        "skip_ring_like": not args.disable_ring_guard,
    }
    if method == "topk":
        kwargs["base_method"] = args.topk_base_method
    return kwargs


def _is_slow_adhesion_method(method: str, args: argparse.Namespace) -> bool:
    if method in (
        "adhesion",
        "dilated_adhesion",
        "topk_dilated",
        "group_then_adhesion",
        "geometry_merge",
        "dilated_group_then_adhesion",
        "topk_group_then_adhesion",
        "topk_geometry_merge",
        "topk_gta",
    ):
        return True
    if method == "topk" and args.topk_base_method in (
        "adhesion",
        "dilated_adhesion",
        "group_then_adhesion",
        "geometry_merge",
        "dilated_group_then_adhesion",
    ):
        return True
    return False


def select_class_balanced_samples(
    maps: np.ndarray,
    labels: np.ndarray,
    original_indices: np.ndarray,
    samples_per_class: int,
    seed: int,
    prefer_single_label: bool,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_positions = np.where(labels[:, class_idx] == 1)[0]
        if prefer_single_label:
            single = class_positions[labels[class_positions].sum(axis=1) == 1]
            if len(single) >= min(samples_per_class, len(class_positions)):
                class_positions = single

        n_pick = min(samples_per_class, len(class_positions))
        if n_pick == 0:
            continue

        selected = rng.choice(class_positions, size=n_pick, replace=False)
        selected = sorted(int(x) for x in selected)
        for valid_pos in selected:
            label_vec = labels[valid_pos]
            rows.append({
                "review_class": class_name,
                "review_class_idx": class_idx,
                "valid_pos": int(valid_pos),
                "orig_index": int(original_indices[valid_pos]),
                "label_names": _label_names(label_vec),
                "label_cardinality": int(label_vec.sum()),
                "defect_area": int((maps[valid_pos] == 2).sum()),
                "valid_area": int(((maps[valid_pos] == 1) | (maps[valid_pos] == 2)).sum()),
            })

    selected_df = pd.DataFrame(rows)
    if selected_df.empty:
        return selected_df

    selected_df["is_duplicate_orig_index"] = selected_df.duplicated("orig_index", keep=False)
    selected_df = selected_df.sort_values(
        ["review_class_idx", "orig_index", "review_class"]
    ).reset_index(drop=True)
    return selected_df


def compute_metrics_for_samples(
    maps: np.ndarray,
    selected_df: pd.DataFrame,
    methods: Iterable[str],
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[Tuple[int, str], List[Dict]]]:
    metrics = []
    cache = {}

    for _, row in selected_df.iterrows():
        valid_pos = int(row["valid_pos"])
        raw_map = maps[valid_pos]
        defect_mask = raw_map == 2
        valid_mask = (raw_map == 1) | (raw_map == 2)
        defect_pixels = set(zip(*np.where(defect_mask)))
        defect_area = max(len(defect_pixels), 1)

        print(
            f"Evaluating orig_index={int(row['orig_index'])} "
            f"class={row['review_class']} defects={defect_area}",
            flush=True,
        )

        for method in methods:
            skipped = (
                args.skip_slow_large_maps
                and defect_area > args.max_defect_area_for_slow_methods
                and _is_slow_adhesion_method(method, args)
            )
            if skipped:
                clusters = None
                status = "skipped_large_map"
            else:
                clusters = cluster(
                    defect_mask,
                    valid_mask,
                    method=method,
                    **_proposal_kwargs(method, args),
                )
                status = "ok"
            cache[(valid_pos, method)] = clusters
            covered_pixels = _cluster_pixels(clusters)
            covered_defects = covered_pixels & defect_pixels
            extra_pixels = covered_pixels - defect_pixels
            metric_clusters = clusters or []
            areas = [int(item.get("area", 0)) for item in metric_clusters]

            metrics.append({
                "review_class": row["review_class"],
                "valid_pos": valid_pos,
                "orig_index": int(row["orig_index"]),
                "label_names": row["label_names"],
                "method": method,
                "status": status,
                "num_tokens": len(metric_clusters),
                "coverage_ratio": len(covered_defects) / defect_area,
                "extra_pixel_ratio": len(extra_pixels) / defect_area,
                "largest_token_area": max(areas) if areas else 0,
                "top_token_area_sum": int(sum(areas[:args.top_k])),
                "mean_token_area": float(np.mean(areas)) if areas else 0.0,
                "median_token_area": float(np.median(areas)) if areas else 0.0,
                "small_token_count": int(sum(area < args.small_area for area in areas)),
            })

    return pd.DataFrame(metrics), cache


def write_manual_review_template(metrics_df: pd.DataFrame, out_path: Path) -> None:
    template = metrics_df[[
        "review_class",
        "orig_index",
        "valid_pos",
        "label_names",
        "method",
        "status",
        "num_tokens",
        "coverage_ratio",
        "extra_pixel_ratio",
    ]].copy()
    template["visual_score_1_to_5"] = ""
    template["major_pattern_kept_y_n"] = ""
    template["over_fragmented_y_n"] = ""
    template["over_merged_y_n"] = ""
    template["noise_kept_y_n"] = ""
    template["retrieval_usable_y_n"] = ""
    template["notes"] = ""
    template.to_csv(out_path, index=False)


def plot_review_panels(
    maps: np.ndarray,
    selected_df: pd.DataFrame,
    methods: Sequence[str],
    cluster_cache: Dict[Tuple[int, str], List[Dict]],
    out_dir: Path,
    rows_per_page: int,
) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for class_name in CLASS_NAMES:
        class_df = selected_df[selected_df["review_class"] == class_name]
        if class_df.empty:
            continue

        pages = [
            class_df.iloc[start:start + rows_per_page]
            for start in range(0, len(class_df), rows_per_page)
        ]
        for page_idx, page_df in enumerate(pages, start=1):
            n_rows = len(page_df)
            n_cols = len(methods) + 1
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(2.35 * n_cols, 2.25 * n_rows),
                squeeze=False,
            )

            axes[0, 0].set_title("Raw Map", fontsize=9, fontweight="bold")
            for col, method in enumerate(methods, start=1):
                axes[0, col].set_title(METHOD_TITLES.get(method, method), fontsize=9, fontweight="bold")

            for row_idx, (_, sample) in enumerate(page_df.iterrows()):
                valid_pos = int(sample["valid_pos"])
                raw_map = maps[valid_pos]
                axes[row_idx, 0].imshow(raw_map, cmap="viridis")
                axes[row_idx, 0].set_ylabel(
                    f"orig {int(sample['orig_index'])}\n{sample['label_names']}",
                    fontsize=7,
                )

                for col, method in enumerate(methods, start=1):
                    clusters = cluster_cache[(valid_pos, method)]
                    overlay = _build_overlay(clusters, raw_map)
                    ax = axes[row_idx, col]
                    ax.imshow(
                        overlay,
                        cmap="tab20",
                        vmin=0,
                        vmax=1,
                    )
                    if clusters is None:
                        ax.text(
                            0.5,
                            0.5,
                            "skip",
                            transform=ax.transAxes,
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="red",
                        )
                        ax.set_xlabel("skip", fontsize=7)
                    else:
                        ax.set_xlabel(f"{len(clusters)} tok", fontsize=7)

            for ax in axes.flat:
                ax.set_xticks([])
                ax.set_yticks([])

            fig.suptitle(
                f"Week1 Proposal Review - {class_name} - page {page_idx}",
                fontsize=12,
                y=1.01,
            )
            fig.tight_layout()
            fig.savefig(fig_dir / f"proposal_review_{class_name}_p{page_idx}.png", dpi=160, bbox_inches="tight")
            plt.close(fig)


def write_summary(metrics_df: pd.DataFrame, selected_df: pd.DataFrame, out_path: Path, args: argparse.Namespace) -> None:
    ok_df = metrics_df[metrics_df["status"] == "ok"].copy()
    method_summary = ok_df.groupby("method").agg(
        maps=("orig_index", "nunique"),
        mean_tokens=("num_tokens", "mean"),
        median_tokens=("num_tokens", "median"),
        mean_coverage=("coverage_ratio", "mean"),
        min_coverage=("coverage_ratio", "min"),
        mean_extra_pixels=("extra_pixel_ratio", "mean"),
        mean_small_tokens=("small_token_count", "mean"),
    ).reset_index()
    skipped_summary = metrics_df[metrics_df["status"] != "ok"].groupby("method").size().to_dict()

    class_counts = selected_df.groupby("review_class")["orig_index"].nunique().reindex(CLASS_NAMES, fill_value=0)

    lines = [
        "# Week 1 Proposal Manual Review Set",
        "",
        "## Sampling",
        "",
        f"- Source: `{args.npz}`",
        f"- Samples per class target: `{args.samples_per_class}`",
        f"- Random seed: `{args.seed}`",
        f"- Prefer single-label samples: `{args.prefer_single_label}`",
        f"- Skip slow methods on large maps: `{args.skip_slow_large_maps}`",
        f"- Slow-method defect-area threshold: `{args.max_defect_area_for_slow_methods}`",
        f"- Unique sampled maps: `{selected_df['orig_index'].nunique()}`",
        f"- Review rows including multi-label membership: `{len(selected_df)}`",
        "",
        "Class counts:",
        "",
        "| Class | Unique maps |",
        "|---|---:|",
    ]
    for class_name, count in class_counts.items():
        lines.append(f"| {class_name} | {int(count)} |")

    lines.extend([
        "",
        "## Automatic Metrics By Method",
        "",
        "| Method | Maps | Skipped | Mean Tokens | Median Tokens | Mean Coverage | Min Coverage | Mean Extra Pixels | Mean Small Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in method_summary.iterrows():
        skipped = int(skipped_summary.get(row["method"], 0))
        lines.append(
            f"| {row['method']} | {int(row['maps'])} | {skipped} | "
            f"{row['mean_tokens']:.2f} | {row['median_tokens']:.1f} | "
            f"{row['mean_coverage']:.3f} | {row['min_coverage']:.3f} | "
            f"{row['mean_extra_pixels']:.3f} | {row['mean_small_tokens']:.2f} |"
        )

    lines.extend([
        "",
        "## Human Review Fields",
        "",
        "Fill `manual_review_template.csv` after checking the figures:",
        "",
        "- `visual_score_1_to_5`: overall proposal quality.",
        "- `major_pattern_kept_y_n`: whether the main defect pattern is retained.",
        "- `over_fragmented_y_n`: whether one pattern is split into too many pieces.",
        "- `over_merged_y_n`: whether different patterns are incorrectly merged.",
        "- `noise_kept_y_n`: whether obvious noise is retained as tokens.",
        "- `retrieval_usable_y_n`: whether the result is acceptable for compact retrieval.",
        "",
        "The automatic metrics are supporting evidence only. The final conclusion should combine them with the human review columns.",
        "",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a WM38K class-balanced proposal review set.")
    parser.add_argument("--npz", type=str, default="/Users/kayw/Documents/trae_projects/match-test/data/wm38k/Wafer_Map_Datasets.npz")
    parser.add_argument("--out-dir", type=str, default="/Users/kayw/Documents/trae_projects/match-test/artifacts/week1/proposal_review")
    parser.add_argument("--samples-per-class", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--prefer-single-label", action="store_true", help="Prefer single-label maps when a class has enough samples.")
    parser.add_argument("--rows-per-page", type=int, default=10)
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--small-area", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--topk-base-method", type=str, default="geometry_merge")
    parser.add_argument("--dilation-radius", type=int, default=1)
    parser.add_argument("--use-closing-for-grouping", action="store_true")
    parser.add_argument("--suspicious-area", type=int, default=40)
    parser.add_argument("--min-suspicious-cues", type=int, default=1)
    parser.add_argument("--max-split-count", type=int, default=12)
    parser.add_argument("--min-split-coverage", type=float, default=0.5)
    parser.add_argument("--disable-ring-guard", action="store_true")
    parser.add_argument(
        "--skip-slow-large-maps",
        action="store_true",
        help="Skip adhesion-based methods on maps whose defect area exceeds the threshold.",
    )
    parser.add_argument(
        "--max-defect-area-for-slow-methods",
        type=int,
        default=900,
        help="Defect-area threshold used by --skip-slow-large-maps.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    maps, labels = load_wm38k(args.npz)
    valid_maps, valid_labels, original_indices = filter_valid_samples(maps, labels)

    selected_df = select_class_balanced_samples(
        valid_maps,
        valid_labels,
        original_indices,
        args.samples_per_class,
        args.seed,
        args.prefer_single_label,
    )
    if selected_df.empty:
        raise RuntimeError("No labeled WM38K samples were selected.")

    selected_df.to_csv(out_dir / "selected_samples.csv", index=False)

    metrics_df, cluster_cache = compute_metrics_for_samples(
        valid_maps,
        selected_df,
        args.methods,
        args,
    )
    metrics_df.to_csv(out_dir / "proposal_metrics.csv", index=False)
    write_manual_review_template(metrics_df, out_dir / "manual_review_template.csv")
    plot_review_panels(
        valid_maps,
        selected_df,
        args.methods,
        cluster_cache,
        out_dir,
        args.rows_per_page,
    )
    write_summary(metrics_df, selected_df, out_dir / "proposal_review_summary.md", args)

    print(f"Selected review rows: {len(selected_df)}")
    print(f"Unique maps: {selected_df['orig_index'].nunique()}")
    print(f"Output: {out_dir}")
    print("Files:")
    print(f"  {out_dir / 'selected_samples.csv'}")
    print(f"  {out_dir / 'proposal_metrics.csv'}")
    print(f"  {out_dir / 'manual_review_template.csv'}")
    print(f"  {out_dir / 'proposal_review_summary.md'}")
    print(f"  {out_dir / 'figures'}")


if __name__ == "__main__":
    main()
