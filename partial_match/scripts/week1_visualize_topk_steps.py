# -*- coding: utf-8 -*-
"""
Visualize the step-by-step TopK proposal process.

Columns:
  Raw Map -> Defect Mask -> Base Candidates -> TopK Selected -> Selected vs Rejected
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
    names = [name for name, flag in zip(CLASS_NAMES, label_vec) if int(flag) == 1]
    return "|".join(names)


def _sample_by_class(
    maps: np.ndarray,
    labels: np.ndarray,
    original_indices: np.ndarray,
    samples_per_class: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        positions = np.where(labels[:, class_idx] == 1)[0]
        if len(positions) == 0:
            continue
        selected = rng.choice(
            positions,
            size=min(samples_per_class, len(positions)),
            replace=False,
        )
        for valid_pos in sorted(int(x) for x in selected):
            raw_map = maps[valid_pos]
            rows.append({
                "review_class": class_name,
                "review_class_idx": class_idx,
                "valid_pos": valid_pos,
                "orig_index": int(original_indices[valid_pos]),
                "label_names": _label_names(labels[valid_pos]),
                "defect_area": int((raw_map == 2).sum()),
            })
    return pd.DataFrame(rows).sort_values(
        ["review_class_idx", "orig_index"]
    ).reset_index(drop=True)


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


def _selected_vs_rejected(raw_map: np.ndarray, candidates: Sequence[Dict], selected: Sequence[Dict]) -> np.ndarray:
    overlay = np.zeros(raw_map.shape, dtype=float)
    valid_mask = (raw_map == 1) | (raw_map == 2)
    overlay[valid_mask] = 0.04

    selected_sets = [set(_coords(item)) for item in selected]
    selected_union = set().union(*selected_sets) if selected_sets else set()

    for item in candidates:
        for r, c in _coords(item):
            overlay[r, c] = 0.12

    for idx, pixels in enumerate(selected_sets):
        value = 0.32 + (idx % 5) * 0.12
        for r, c in pixels:
            overlay[r, c] = value

    # Keep selected pixels visually dominant even if candidate ordering overlaps.
    for idx, item in enumerate(selected):
        value = 0.32 + (idx % 5) * 0.12
        for r, c in _coords(item):
            if (r, c) in selected_union:
                overlay[r, c] = value
    return overlay


def _write_metrics(rows: List[Dict], out_path: Path) -> None:
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize TopK proposal steps.")
    parser.add_argument("--npz", type=str, default="/Users/kayw/Documents/trae_projects/match-test/data/wm38k/Wafer_Map_Datasets.npz")
    parser.add_argument("--out-dir", type=str, default="/Users/kayw/Documents/trae_projects/match-test/artifacts/week1/topk_step_figures")
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--base-method", type=str, default="geometry_merge")
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--dilation-radius", type=int, default=1)
    parser.add_argument("--suspicious-area", type=int, default=40)
    parser.add_argument("--min-suspicious-cues", type=int, default=1)
    parser.add_argument("--max-split-count", type=int, default=12)
    parser.add_argument("--min-split-coverage", type=float, default=0.5)
    parser.add_argument("--rows-per-page", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    maps, labels = load_wm38k(args.npz)
    valid_maps, valid_labels, original_indices = filter_valid_samples(maps, labels)
    selected_df = _sample_by_class(
        valid_maps,
        valid_labels,
        original_indices,
        args.samples_per_class,
        args.seed,
    )
    selected_df.to_csv(out_dir / "topk_step_samples.csv", index=False)

    metric_rows = []
    for class_name in CLASS_NAMES:
        class_df = selected_df[selected_df["review_class"] == class_name]
        if class_df.empty:
            continue

        pages = [
            class_df.iloc[start:start + args.rows_per_page]
            for start in range(0, len(class_df), args.rows_per_page)
        ]
        for page_idx, page_df in enumerate(pages, start=1):
            n_rows = len(page_df)
            n_cols = 5
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(2.8 * n_cols, 2.55 * n_rows),
                squeeze=False,
            )
            titles = [
                "Raw Map",
                "Defect Mask",
                f"{args.base_method} candidates",
                f"TopK selected\n(k={args.top_k})",
                "Selected vs rejected",
            ]
            for col, title in enumerate(titles):
                axes[0, col].set_title(title, fontsize=9, fontweight="bold")

            for row_idx, (_, sample) in enumerate(page_df.iterrows()):
                valid_pos = int(sample["valid_pos"])
                raw_map = valid_maps[valid_pos]
                defect_mask = raw_map == 2
                valid_mask = (raw_map == 1) | (raw_map == 2)

                candidates = cluster(
                    defect_mask,
                    valid_mask,
                    method=args.base_method,
                    min_area=args.min_area,
                    dilation_radius=args.dilation_radius,
                    suspicious_area=args.suspicious_area,
                    min_suspicious_cues=args.min_suspicious_cues,
                    max_split_count=args.max_split_count,
                    min_split_coverage=args.min_split_coverage,
                )
                selected = cluster(
                    defect_mask,
                    valid_mask,
                    method="topk",
                    top_k=args.top_k,
                    base_method=args.base_method,
                    min_area=args.min_area,
                    dilation_radius=args.dilation_radius,
                    suspicious_area=args.suspicious_area,
                    min_suspicious_cues=args.min_suspicious_cues,
                    max_split_count=args.max_split_count,
                    min_split_coverage=args.min_split_coverage,
                )

                axes[row_idx, 0].imshow(raw_map, cmap="viridis")
                axes[row_idx, 0].set_ylabel(
                    f"orig {int(sample['orig_index'])}\n{sample['label_names']}",
                    fontsize=7,
                )
                axes[row_idx, 1].imshow(defect_mask, cmap="gray")
                axes[row_idx, 2].imshow(_overlay_clusters(raw_map, candidates), cmap="tab20", vmin=0, vmax=1)
                axes[row_idx, 2].set_xlabel(f"{len(candidates)} cand", fontsize=7)
                axes[row_idx, 3].imshow(_overlay_clusters(raw_map, selected), cmap="tab20", vmin=0, vmax=1)
                axes[row_idx, 3].set_xlabel(f"{len(selected)} / {args.top_k} tok", fontsize=7)
                axes[row_idx, 4].imshow(_selected_vs_rejected(raw_map, candidates, selected), cmap="tab20", vmin=0, vmax=1)
                axes[row_idx, 4].set_xlabel("color=selected", fontsize=7)

                defect_area = max(int(defect_mask.sum()), 1)
                selected_pixels = set()
                for item in selected:
                    selected_pixels.update(_coords(item))
                metric_rows.append({
                    "review_class": sample["review_class"],
                    "orig_index": int(sample["orig_index"]),
                    "label_names": sample["label_names"],
                    "defect_area": defect_area,
                    "base_method": args.base_method,
                    "top_k": args.top_k,
                    "dilation_radius": args.dilation_radius,
                    "suspicious_area": args.suspicious_area,
                    "min_suspicious_cues": args.min_suspicious_cues,
                    "max_split_count": args.max_split_count,
                    "min_split_coverage": args.min_split_coverage,
                    "candidate_count": len(candidates),
                    "selected_count": len(selected),
                    "selected_area": len(selected_pixels),
                    "selected_coverage": len(selected_pixels) / defect_area,
                })

            for ax in axes.flat:
                ax.set_xticks([])
                ax.set_yticks([])

            fig.suptitle(
                f"TopK Proposal Steps - {class_name} - page {page_idx}",
                fontsize=12,
                y=1.01,
            )
            fig.tight_layout()
            fig.savefig(fig_dir / f"topk_steps_{class_name}_p{page_idx}.png", dpi=170, bbox_inches="tight")
            plt.close(fig)

    _write_metrics(metric_rows, out_dir / "topk_step_metrics.csv")

    print(f"Selected rows: {len(selected_df)}")
    print(f"Output: {out_dir}")
    print(f"Figures: {fig_dir}")
    print(f"Metrics: {out_dir / 'topk_step_metrics.csv'}")


if __name__ == "__main__":
    main()
