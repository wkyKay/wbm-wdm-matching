# -*- coding: utf-8 -*-
"""
Visualize the current proposal flow on multi-label WM38K wafers.

Columns:
  Raw Map -> 1 Filtered -> 2 Dilated Groups -> Grouped Original Pixels -> 3 Adhesion Result
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from partial_match.core.clustering import (
    _build_se,
    _compute_cluster_stats,
    _connected_components,
    _custom_binary_closing,
    _custom_binary_dilation,
    cluster,
)
from partial_match.data.data_io import CLASS_NAMES, filter_valid_samples, load_wm38k


def _label_names(label_vec: np.ndarray) -> str:
    return "|".join(name for name, flag in zip(CLASS_NAMES, label_vec) if int(flag) == 1)


def _coords(cluster_item: Dict) -> List[tuple]:
    coords = cluster_item.get("pixels", cluster_item.get("pixel_coords", []))
    out = []
    for coord in coords:
        if isinstance(coord, dict):
            out.append((int(coord["row"]), int(coord["col"])))
        else:
            out.append((int(coord[0]), int(coord[1])))
    return out


def _clusters_to_mask(shape: tuple, clusters: Sequence[Dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for item in clusters:
        for r, c in _coords(item):
            mask[r, c] = True
    return mask


def _overlay_clusters(
    raw_map: np.ndarray,
    clusters: Sequence[Dict],
    valid_level: float = 0.04,
    base_value: float = 0.18,
    step: float = 0.055,
) -> np.ndarray:
    overlay = np.zeros(raw_map.shape, dtype=float)
    valid_mask = (raw_map == 1) | (raw_map == 2)
    overlay[valid_mask] = valid_level
    for idx, item in enumerate(clusters):
        value = base_value + (idx % 14) * step
        for r, c in _coords(item):
            overlay[r, c] = value
    return overlay


def _overlay_group_masks(raw_map: np.ndarray, groups: Sequence[np.ndarray]) -> np.ndarray:
    overlay = np.zeros(raw_map.shape, dtype=float)
    valid_mask = (raw_map == 1) | (raw_map == 2)
    overlay[valid_mask] = 0.04
    for idx, group in enumerate(groups):
        value = 0.16 + (idx % 14) * 0.055
        for r, c in group.astype(int):
            overlay[r, c] = value
    return overlay


def _build_grouping_steps(
    raw_map: np.ndarray,
    min_area: int,
    dilation_radius: int,
    use_closing: bool,
    structure: str,
) -> Dict:
    defect_mask = raw_map == 2
    valid_mask = (raw_map == 1) | (raw_map == 2)
    H, W = raw_map.shape

    filtered = cluster(defect_mask, valid_mask, method="filtered", min_area=min_area)
    filtered_mask = _clusters_to_mask(raw_map.shape, filtered)

    se = _build_se(dilation_radius, structure)
    try:
        from scipy.ndimage import binary_closing, binary_dilation
        if use_closing:
            grouping_mask = binary_closing(filtered_mask, structure=se)
        else:
            grouping_mask = binary_dilation(filtered_mask, structure=se)
    except ImportError:
        if use_closing:
            grouping_mask = _custom_binary_closing(filtered_mask, se)
        else:
            grouping_mask = _custom_binary_dilation(filtered_mask, se)
    grouping_mask = grouping_mask & valid_mask

    groups = _connected_components(grouping_mask)
    original = defect_mask & valid_mask
    grouped_original = []
    for group in groups:
        group_mask = np.zeros(raw_map.shape, dtype=bool)
        group_mask[group[:, 0].astype(int), group[:, 1].astype(int)] = True
        original_pixels = np.argwhere(original & group_mask).astype(np.float32)
        if len(original_pixels) >= min_area:
            grouped_original.append(_compute_cluster_stats(original_pixels, H, W))

    return {
        "filtered": filtered,
        "filtered_mask": filtered_mask,
        "groups": groups,
        "grouping_mask": grouping_mask,
        "grouped_original": grouped_original,
    }


def _select_combo_samples(
    maps: np.ndarray,
    labels: np.ndarray,
    original_indices: np.ndarray,
    n_samples: int,
    seed: int,
    min_label_cardinality: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cardinalities = labels.sum(axis=1)
    positions = np.where(cardinalities >= min_label_cardinality)[0]
    if len(positions) == 0:
        return pd.DataFrame()

    selected = rng.choice(positions, size=min(n_samples, len(positions)), replace=False)
    rows = []
    for valid_pos in sorted(int(x) for x in selected):
        raw_map = maps[valid_pos]
        rows.append({
            "valid_pos": valid_pos,
            "orig_index": int(original_indices[valid_pos]),
            "label_names": _label_names(labels[valid_pos]),
            "label_cardinality": int(labels[valid_pos].sum()),
            "defect_area": int((raw_map == 2).sum()),
        })
    return pd.DataFrame(rows).sort_values(
        ["label_cardinality", "orig_index"],
        ascending=[False, True],
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize filtered -> dilated group -> adhesion steps on multi-label wafers.")
    parser.add_argument("--npz", type=str, default="/Users/kayw/Documents/trae_projects/match-test/data/wm38k/Wafer_Map_Datasets.npz")
    parser.add_argument("--out-dir", type=str, default="/Users/kayw/Documents/trae_projects/match-test/artifacts/week1/group_adhesion_step_figures")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--rows-per-page", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-label-cardinality", type=int, default=2)
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--dilation-radius", type=int, default=1)
    parser.add_argument("--use-closing-for-grouping", action="store_true")
    parser.add_argument("--structure", choices=["cross", "square"], default="cross")
    parser.add_argument("--suspicious-area", type=int, default=40)
    parser.add_argument("--min-suspicious-cues", type=int, default=1)
    parser.add_argument("--max-split-count", type=int, default=12)
    parser.add_argument("--min-split-coverage", type=float, default=0.5)
    parser.add_argument("--disable-ring-guard", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    maps, labels = load_wm38k(args.npz)
    valid_maps, valid_labels, original_indices = filter_valid_samples(maps, labels)
    selected_df = _select_combo_samples(
        valid_maps,
        valid_labels,
        original_indices,
        args.samples,
        args.seed,
        args.min_label_cardinality,
    )
    if selected_df.empty:
        raise RuntimeError("No multi-label WM38K samples matched the requested label cardinality.")

    selected_df.to_csv(out_dir / "group_adhesion_step_samples.csv", index=False)

    metric_rows = []
    pages = [
        selected_df.iloc[start:start + args.rows_per_page]
        for start in range(0, len(selected_df), args.rows_per_page)
    ]
    for page_idx, page_df in enumerate(pages, start=1):
        n_rows = len(page_df)
        n_cols = 5
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(2.9 * n_cols, 2.6 * n_rows),
            squeeze=False,
        )
        titles = [
            "Raw Map",
            "1 Filtered",
            "2 Dilated Groups",
            "Grouped Original Pixels",
            "3 Adhesion Result",
        ]
        for col, title in enumerate(titles):
            axes[0, col].set_title(title, fontsize=9, fontweight="bold")

        for row_idx, (_, sample) in enumerate(page_df.iterrows()):
            valid_pos = int(sample["valid_pos"])
            raw_map = valid_maps[valid_pos]
            defect_mask = raw_map == 2
            valid_mask = (raw_map == 1) | (raw_map == 2)
            steps = _build_grouping_steps(
                raw_map,
                args.min_area,
                args.dilation_radius,
                args.use_closing_for_grouping,
                args.structure,
            )
            final_clusters = cluster(
                defect_mask,
                valid_mask,
                method="group_then_adhesion",
                min_area=args.min_area,
                dilation_radius=args.dilation_radius,
                use_closing=args.use_closing_for_grouping,
                structure=args.structure,
                suspicious_area=args.suspicious_area,
                min_suspicious_cues=args.min_suspicious_cues,
                max_split_count=args.max_split_count,
                min_split_coverage=args.min_split_coverage,
                skip_ring_like=not args.disable_ring_guard,
            )

            axes[row_idx, 0].imshow(raw_map, cmap="viridis")
            axes[row_idx, 0].set_ylabel(
                f"orig {int(sample['orig_index'])}\n{sample['label_names']}",
                fontsize=7,
            )
            axes[row_idx, 1].imshow(_overlay_clusters(raw_map, steps["filtered"]), cmap="tab20", vmin=0, vmax=1)
            axes[row_idx, 1].set_xlabel(f"{len(steps['filtered'])} comps", fontsize=7)
            axes[row_idx, 2].imshow(_overlay_group_masks(raw_map, steps["groups"]), cmap="tab20", vmin=0, vmax=1)
            axes[row_idx, 2].set_xlabel(f"{len(steps['groups'])} groups", fontsize=7)
            axes[row_idx, 3].imshow(_overlay_clusters(raw_map, steps["grouped_original"]), cmap="tab20", vmin=0, vmax=1)
            axes[row_idx, 3].set_xlabel(f"{len(steps['grouped_original'])} orig groups", fontsize=7)
            axes[row_idx, 4].imshow(_overlay_clusters(raw_map, final_clusters), cmap="tab20", vmin=0, vmax=1)
            axes[row_idx, 4].set_xlabel(f"{len(final_clusters)} tokens", fontsize=7)

            split_accepted = sum(1 for item in final_clusters if item.get("split_status") == "accepted")
            metric_rows.append({
                "orig_index": int(sample["orig_index"]),
                "label_names": sample["label_names"],
                "label_cardinality": int(sample["label_cardinality"]),
                "defect_area": int(sample["defect_area"]),
                "filtered_count": len(steps["filtered"]),
                "dilated_group_count": len(steps["groups"]),
                "grouped_original_count": len(steps["grouped_original"]),
                "final_token_count": len(final_clusters),
                "accepted_split_token_count": split_accepted,
            })

        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(
            f"Filtered -> Dilated Group -> Adhesion on Multi-label Wafers - page {page_idx}",
            fontsize=12,
            y=1.01,
        )
        fig.tight_layout()
        fig.savefig(fig_dir / f"group_adhesion_steps_p{page_idx}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    pd.DataFrame(metric_rows).to_csv(out_dir / "group_adhesion_step_metrics.csv", index=False)

    print(f"Selected multi-label rows: {len(selected_df)}")
    print(f"Output: {out_dir}")
    print(f"Figures: {fig_dir}")
    print(f"Metrics: {out_dir / 'group_adhesion_step_metrics.csv'}")


if __name__ == "__main__":
    main()
