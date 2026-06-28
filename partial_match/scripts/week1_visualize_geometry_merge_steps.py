# -*- coding: utf-8 -*-
"""
Visualize the geometry-aware proposal flow on multi-label WM38K wafers.

Columns:
  Raw Map -> 1 Filtered -> 2 Adhesion Candidates -> 3 Geometry Merge -> TopK
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

from partial_match.core.clustering import cluster
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
    parser = argparse.ArgumentParser(description="Visualize filtered -> adhesion -> geometry merge proposal steps.")
    parser.add_argument("--npz", type=str, default="/Users/kayw/Documents/trae_projects/match-test/data/wm38k/Wafer_Map_Datasets.npz")
    parser.add_argument("--out-dir", type=str, default="/Users/kayw/Documents/trae_projects/match-test/artifacts/week1/geometry_merge_step_figures")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--rows-per-page", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-label-cardinality", type=int, default=2)
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--suspicious-area", type=int, default=40)
    parser.add_argument("--min-suspicious-cues", type=int, default=1)
    parser.add_argument("--max-split-count", type=int, default=8)
    parser.add_argument("--min-split-coverage", type=float, default=0.6)
    parser.add_argument("--ring-radial-gap", type=float, default=0.14)
    parser.add_argument("--ring-theta-gap", type=float, default=55.0)
    parser.add_argument("--line-gap", type=float, default=11.0)
    parser.add_argument("--line-angle-gap", type=float, default=30.0)
    parser.add_argument("--line-perp-gap", type=float, default=5.0)
    parser.add_argument("--blob-gap", type=float, default=3.0)
    parser.add_argument("--blob-max-bbox-area", type=int, default=220)
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

    selected_df.to_csv(out_dir / "geometry_merge_step_samples.csv", index=False)
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
        titles = ["Raw Map", "1 Filtered", "2 Adhesion", "3 Geometry Merge", f"TopK (k={args.top_k})"]
        for col, title in enumerate(titles):
            axes[0, col].set_title(title, fontsize=9, fontweight="bold")

        for row_idx, (_, sample) in enumerate(page_df.iterrows()):
            valid_pos = int(sample["valid_pos"])
            raw_map = valid_maps[valid_pos]
            defect_mask = raw_map == 2
            valid_mask = (raw_map == 1) | (raw_map == 2)

            common = {
                "min_area": args.min_area,
                "suspicious_area": args.suspicious_area,
                "min_suspicious_cues": args.min_suspicious_cues,
                "max_split_count": args.max_split_count,
                "min_split_coverage": args.min_split_coverage,
            }
            geom_kwargs = {
                **common,
                "ring_radial_gap": args.ring_radial_gap,
                "ring_theta_gap": args.ring_theta_gap,
                "line_gap": args.line_gap,
                "line_angle_gap": args.line_angle_gap,
                "line_perp_gap": args.line_perp_gap,
                "blob_gap": args.blob_gap,
                "blob_max_bbox_area": args.blob_max_bbox_area,
            }

            filtered = cluster(defect_mask, valid_mask, method="filtered", min_area=args.min_area)
            adhesion = cluster(defect_mask, valid_mask, method="adhesion", **common)
            geometry = cluster(defect_mask, valid_mask, method="geometry_merge", **geom_kwargs)
            topk = cluster(
                defect_mask,
                valid_mask,
                method="topk",
                base_method="geometry_merge",
                top_k=args.top_k,
                **geom_kwargs,
            )

            axes[row_idx, 0].imshow(raw_map, cmap="viridis")
            axes[row_idx, 0].set_ylabel(
                f"orig {int(sample['orig_index'])}\n{sample['label_names']}",
                fontsize=7,
            )
            axes[row_idx, 1].imshow(_overlay_clusters(raw_map, filtered), cmap="tab20", vmin=0, vmax=1)
            axes[row_idx, 1].set_xlabel(f"{len(filtered)} comps", fontsize=7)
            axes[row_idx, 2].imshow(_overlay_clusters(raw_map, adhesion), cmap="tab20", vmin=0, vmax=1)
            axes[row_idx, 2].set_xlabel(f"{len(adhesion)} cand", fontsize=7)
            axes[row_idx, 3].imshow(_overlay_clusters(raw_map, geometry), cmap="tab20", vmin=0, vmax=1)
            axes[row_idx, 3].set_xlabel(f"{len(geometry)} merged", fontsize=7)
            axes[row_idx, 4].imshow(_overlay_clusters(raw_map, topk), cmap="tab20", vmin=0, vmax=1)
            axes[row_idx, 4].set_xlabel(f"{len(topk)} tok", fontsize=7)

            merged_count = sum(1 for item in geometry if item.get("merged_count", 1) > 1)
            metric_rows.append({
                "orig_index": int(sample["orig_index"]),
                "label_names": sample["label_names"],
                "label_cardinality": int(sample["label_cardinality"]),
                "defect_area": int(sample["defect_area"]),
                "filtered_count": len(filtered),
                "adhesion_count": len(adhesion),
                "geometry_merge_count": len(geometry),
                "topk_count": len(topk),
                "merged_token_count": merged_count,
                "merge_reasons": "|".join(sorted(set(item.get("merge_reason", "single") for item in geometry))),
            })

        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(
            f"Filtered -> Adhesion -> Geometry Merge on Multi-label Wafers - page {page_idx}",
            fontsize=12,
            y=1.01,
        )
        fig.tight_layout()
        fig.savefig(fig_dir / f"geometry_merge_steps_p{page_idx}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    pd.DataFrame(metric_rows).to_csv(out_dir / "geometry_merge_step_metrics.csv", index=False)

    print(f"Selected multi-label rows: {len(selected_df)}")
    print(f"Output: {out_dir}")
    print(f"Figures: {fig_dir}")
    print(f"Metrics: {out_dir / 'geometry_merge_step_metrics.csv'}")


if __name__ == "__main__":
    main()
