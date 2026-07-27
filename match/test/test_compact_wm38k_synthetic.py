"""Synthetic WM38K retrieval benchmark for the match ``compact`` proposal mode.

Each group contains exactly 20 maps: one single-class anchor A and nineteen
synthetic candidates. Positive candidates preserve A under rotation, shift,
scale, or union with independent single-class B/C patterns. Negative candidates
contain only B/C. The benchmark compares compact token matching with strict
pixel-aligned IoU, which intentionally cannot compensate for those transforms.

This is an executable experiment script, not a unittest module. Run it with:

  PYTHONPATH=. python3 match/test/test_compact_wm38k_synthetic.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from match.core.local_matching import explain_count_partial_match
from match.core.models import BACKGROUND, GridMaps, VALID_HAS_DEFECT, VALID_NO_DEFECT
from match.core.similarity import SimilarityResult, compute_similarity


CLASS_NAMES = ("center", "donut", "edge-loc", "edge-ring", "loc", "random", "scratch", "near-full")
LOCAL_DISTRACTOR_CLASSES = ("loc", "edge-loc", "scratch")
DEFAULT_DATA_FILE = ROOT.parent / "data" / "wm38k" / "Wafer_Map_Datasets.npz"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "artifacts" / "compact_wm38k_synthetic"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    relevant: bool
    build: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate compact proposals on 20-map WM38K synthetic groups.")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--groups-per-class", type=int, default=1, help="Independent 20-map groups for each selected A class.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--anchor-class", choices=("all", *CLASS_NAMES), default="all", help="A class, or all eight base classes.")
    parser.add_argument("--b-class", choices=CLASS_NAMES, default=None, help="Optional fixed B class; default selects a class different from A.")
    parser.add_argument("--c-class", choices=CLASS_NAMES, default=None, help="Optional fixed C class; default selects a class different from A/B.")
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--top-k-proposals", type=int, default=6)
    parser.add_argument(
        "--proposal-mode",
        choices=("cc", "compact", "arc", "tangential-ring", "sparse-density"),
        default="compact",
        help="Token proposal mode.",
    )
    parser.add_argument("--min-token-score", type=float, default=0.30)
    parser.add_argument("--min-shape-sim", type=float, default=0.30, help="Minimum shape similarity for a token pair to be scored (0-1).")
    parser.add_argument("--score-shape-weight", type=float, default=0.60, help="Weight for token shape similarity (0-1).")
    parser.add_argument("--score-position-weight", type=float, default=0.25, help="Weight for token position affinity (0-1).")
    parser.add_argument("--score-scale-weight", type=float, default=0.15, help="Weight for token scale affinity (0-1).")
    parser.add_argument("--zernike-degree", type=int, default=8, help="Zernike moment degree for shape descriptors (higher = more discriminative).")
    parser.add_argument("--shift-cells", type=int, default=10, help="Absolute row/column shift used for synthetic transforms.")
    parser.add_argument("--scale-down", type=float, default=0.55, help="Center scale factor for contracted synthetic patterns.")
    parser.add_argument("--scale-up", type=float, default=1.50, help="Center scale factor for expanded synthetic patterns.")
    parser.add_argument("--save-figures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--assert-improvement", action="store_true", help="Fail if compact macro AP does not exceed IoU macro AP.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_benchmark(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.assert_improvement and summary["compact_macro_ap"] <= summary["iou_macro_ap"]:
        raise SystemExit("compact did not exceed the IoU baseline on macro AP")


def run_benchmark(args: argparse.Namespace) -> dict:
    if args.groups_per_class <= 0:
        raise ValueError("--groups-per-class must be positive")
    if args.shift_cells <= 0:
        raise ValueError("--shift-cells must be positive")
    if not 0 < args.scale_down < 1 < args.scale_up:
        raise ValueError("Require 0 < --scale-down < 1 < --scale-up")

    maps, labels = _load_wm38k(args.data_file)
    rng = np.random.default_rng(args.seed)
    selected = _select_source_indices(labels, args, rng)

    candidate_rows: list[dict] = []
    group_rows: list[dict] = []
    for group_id, (anchor_class, b_class, c_class, a_idx, b_idx, c_idx) in enumerate(selected):
        group_candidates, group_metrics = _score_group(
            group_id,
            maps[a_idx],
            maps[b_idx],
            maps[c_idx],
            a_idx,
            b_idx,
            c_idx,
            anchor_class,
            b_class,
            c_class,
            rng,
            args,
        )
        candidate_rows.extend(group_candidates)
        group_rows.append(group_metrics)

    summary = _summary(group_rows, args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "candidates.csv", candidate_rows)
    _write_csv(args.out_dir / "group_metrics.csv", group_rows)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "config.json").write_text(json.dumps(_json_config(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _load_wm38k(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"WM38K dataset not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        maps = data["arr_0"]
        labels = data["arr_1"].astype(np.int32)
    if maps.ndim != 3 or labels.ndim != 2 or len(maps) != len(labels):
        raise ValueError("Expected arr_0=(N,H,W) and arr_1=(N,C) in the WM38K NPZ file")
    return maps, labels


def _select_source_indices(labels: np.ndarray, args: argparse.Namespace, rng: np.random.Generator) -> list[tuple[str, str, str, int, int, int]]:
    anchor_classes = CLASS_NAMES if args.anchor_class == "all" else (args.anchor_class,)
    selected = []
    for anchor_class in anchor_classes:
        b_class, c_class = _distractor_classes(anchor_class, args.b_class, args.c_class)
        pools = []
        for class_name in (anchor_class, b_class, c_class):
            class_idx = CLASS_NAMES.index(class_name)
            pool = np.flatnonzero((labels[:, class_idx] == 1) & (labels.sum(axis=1) == 1))
            if len(pool) < args.groups_per_class:
                raise ValueError(f"Class {class_name} has only {len(pool)} single-label maps; need {args.groups_per_class}")
            pools.append(rng.choice(pool, size=args.groups_per_class, replace=False))
        selected.extend(
            (anchor_class, b_class, c_class, int(a), int(b), int(c))
            for a, b, c in zip(*pools)
        )
    return selected


def _distractor_classes(anchor_class: str, requested_b: str | None, requested_c: str | None) -> tuple[str, str]:
    preference = LOCAL_DISTRACTOR_CLASSES
    available = [class_name for class_name in preference if class_name != anchor_class]
    b_class = requested_b or available[0]
    if b_class == anchor_class:
        raise ValueError("--b-class must differ from the current A class")
    c_class = requested_c or next(class_name for class_name in available if class_name != b_class)
    if c_class in {anchor_class, b_class}:
        raise ValueError("--c-class must differ from the current A and B classes")
    return b_class, c_class


def _score_group(
    group_id: int,
    raw_a: np.ndarray,
    raw_b: np.ndarray,
    raw_c: np.ndarray,
    a_idx: int,
    b_idx: int,
    c_idx: int,
    anchor_class: str,
    b_class: str,
    c_class: str,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    valid = (raw_a == VALID_NO_DEFECT) | (raw_a == VALID_HAS_DEFECT)
    anchor = (raw_a == VALID_HAS_DEFECT) & valid
    pattern_b = (raw_b == VALID_HAS_DEFECT) & valid
    pattern_c = (raw_c == VALID_HAS_DEFECT) & valid
    reference = _grid(anchor, valid, role="reference", source_id=a_idx)

    specs = list(_candidate_specs(args.shift_cells, args.scale_down, args.scale_up))
    rng.shuffle(specs)
    rows = []
    plot_masks: dict[int, np.ndarray] = {}
    for candidate_index, spec in enumerate(specs, start=1):
        defects = spec.build(anchor, pattern_b, pattern_c) & valid
        candidate = _grid(defects, valid, role="candidate", source_id=-candidate_index)
        compact = explain_count_partial_match(
            reference,
            candidate,
            min_area=args.min_area,
            top_k=args.top_k_proposals,
            proposal_mode=args.proposal_mode,
            rotation_tolerance=True,
            min_token_score=args.min_token_score,
            min_shape_sim=args.min_shape_sim,
            score_shape_weight=args.score_shape_weight,
            score_position_weight=args.score_position_weight,
            score_scale_weight=args.score_scale_weight,
            zernike_degree=args.zernike_degree,
        )["result"]
        iou_result = compute_similarity(
            reference.count_map,
            candidate.count_map,
            method="iou",
            reference_status=reference.status_map,
            candidate_status=candidate.status_map,
        )
        iou = float(iou_result.score if isinstance(iou_result, SimilarityResult) else iou_result)
        rows.append({
            "group_id": group_id,
            "candidate_index": candidate_index,
            "candidate_name": spec.name,
            "relevant_contains_a": int(spec.relevant),
            "anchor_class": anchor_class,
            "b_class": b_class,
            "c_class": c_class,
            "source_a_id": a_idx,
            "source_b_id": b_idx,
            "source_c_id": c_idx,
            "candidate_defect_cells": int(defects.sum()),
            "compact_score": float(compact.score),
            "compact_matched_tokens": int(compact.matched_tokens),
            "compact_wbm_tokens": int(compact.wbm_tokens),
            "compact_wdm_tokens": int(compact.wdm_tokens),
            "iou_score": iou,
        })
        plot_masks[candidate_index] = defects

    _assign_ranks(rows, "compact_score", "compact_rank")
    _assign_ranks(rows, "iou_score", "iou_rank")
    if args.save_figures:
        _save_ranking_figure(args.out_dir / "ranking_figures", group_id, anchor, valid, rows, plot_masks, anchor_class)
    return rows, {
        "group_id": group_id,
        "source_a_id": a_idx,
        "source_b_id": b_idx,
        "source_c_id": c_idx,
        "anchor_class": anchor_class,
        "b_class": b_class,
        "c_class": c_class,
        "images_per_group": 20,
        "candidates_per_group": len(rows),
        "relevant_candidates": int(sum(row["relevant_contains_a"] for row in rows)),
        "compact_ap": _average_precision(rows, "compact_rank"),
        "iou_ap": _average_precision(rows, "iou_rank"),
        "compact_recall_at_5": _recall_at_k(rows, "compact_rank", 5),
        "iou_recall_at_5": _recall_at_k(rows, "iou_rank", 5),
        "compact_positive_mean": _mean_score(rows, "compact_score", True),
        "compact_negative_mean": _mean_score(rows, "compact_score", False),
        "iou_positive_mean": _mean_score(rows, "iou_score", True),
        "iou_negative_mean": _mean_score(rows, "iou_score", False),
    }


def _save_ranking_figure(
    output_dir: Path,
    group_id: int,
    anchor: np.ndarray,
    valid: np.ndarray,
    rows: list[dict],
    plot_masks: dict[int, np.ndarray],
    anchor_class: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(13.5, 22.5))
    grid = figure.add_gridspec(
        9, 5, height_ratios=(1, 1, 1, 1, 0.20, 1, 1, 1, 1), hspace=0.32, wspace=0.08
    )
    compact_axes = [figure.add_subplot(grid[row, col]) for row in range(4) for col in range(5)]
    separator_axis = figure.add_subplot(grid[4, :])
    iou_axes = [figure.add_subplot(grid[row, col]) for row in range(5, 9) for col in range(5)]
    separator_axis.set_facecolor("#1f2937")
    separator_axis.text(
        0.5, 0.5, "B. STRICT PIXEL IoU BASELINE RANKING", color="white", fontsize=13,
        fontweight="bold", ha="center", va="center",
    )
    separator_axis.set_xticks([])
    separator_axis.set_yticks([])
    for spine in separator_axis.spines.values():
        spine.set_visible(False)

    panels = (
        (compact_axes, "compact_score", "compact_rank", "A. COMPACT PROPOSAL MATCHING RANKING"),
        (iou_axes, "iou_score", "iou_rank", "B. STRICT PIXEL IoU BASELINE RANKING"),
    )
    for panel_axes, score_name, rank_name, panel_title in panels:
        _draw_wafer(panel_axes[0], anchor, valid, f"Reference A{chr(10)}{anchor_class}")
        ranked = sorted(rows, key=lambda row: row[rank_name])
        for axis, row in zip(panel_axes[1:], ranked):
            relevance = "A+" if row["relevant_contains_a"] else "B/C"
            _draw_wafer(
                axis,
                plot_masks[row["candidate_index"]],
                valid,
                f"#{row[rank_name]} {relevance} {row['candidate_name']}{chr(10)}{score_name.replace('_score', '')}={row[score_name]:.3f}",
            )
        panel_axes[0].set_ylabel(panel_title, fontsize=11, fontweight="bold", labelpad=28)

    figure.suptitle(f"WM38K synthetic group {group_id:02d}: anchor A={anchor_class}", fontsize=15, y=0.995)
    figure.text(0.5, 0.968, "A. COMPACT PROPOSAL MATCHING RANKING", fontsize=13, fontweight="bold", ha="center")
    figure.subplots_adjust(left=0.04, right=0.99, bottom=0.015, top=0.955)
    figure.savefig(output_dir / f"group_{group_id:02d}_{anchor_class}.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def _draw_wafer(axis, defects: np.ndarray, valid: np.ndarray, title: str) -> None:
    image = np.zeros((*defects.shape, 3), dtype=np.float32)
    image[valid] = (0.78, 0.78, 0.78)
    image[defects & valid] = (0.82, 0.08, 0.10)
    axis.imshow(image, interpolation="nearest")
    axis.set_title(title, fontsize=6.5)
    axis.axis("off")


def _candidate_specs(shift_cells: int, scale_down: float, scale_up: float) -> Iterable[CandidateSpec]:
    down_label = _transform_label(scale_down)
    up_label = _transform_label(scale_up)
    return (
        CandidateSpec("A_identity", True, lambda a, b, c: a),
        CandidateSpec("A_rot90", True, lambda a, b, c: np.rot90(a, 1)),
        CandidateSpec("A_rot180", True, lambda a, b, c: np.rot90(a, 2)),
        CandidateSpec("A_rot270", True, lambda a, b, c: np.rot90(a, 3)),
        CandidateSpec(f"A_shift_row{shift_cells}_col{-shift_cells}", True, lambda a, b, c: _shift(a, shift_cells, -shift_cells)),
        CandidateSpec(f"A_scale_{down_label}", True, lambda a, b, c: _scale_about_center(a, scale_down)),
        CandidateSpec(f"A_scale_{up_label}", True, lambda a, b, c: _scale_about_center(a, scale_up)),
        CandidateSpec("A_rot90_plus_B", True, lambda a, b, c: np.rot90(a, 1) | b),
        CandidateSpec("A_rot180_plus_B", True, lambda a, b, c: np.rot90(a, 2) | b),
        CandidateSpec("A_rot90_plus_B_plus_C", True, lambda a, b, c: np.rot90(a, 1) | b | c),
        CandidateSpec(f"A_shift_row{shift_cells}_col{-shift_cells}_plus_B_plus_C", True, lambda a, b, c: _shift(a, shift_cells, -shift_cells) | b | c),
        CandidateSpec("B", False, lambda a, b, c: b),
        CandidateSpec("C", False, lambda a, b, c: c),
        CandidateSpec("B_plus_C", False, lambda a, b, c: b | c),
        CandidateSpec("B_rot90", False, lambda a, b, c: np.rot90(b, 1)),
        CandidateSpec(f"B_shift_row{-shift_cells}_col{shift_cells}", False, lambda a, b, c: _shift(b, -shift_cells, shift_cells)),
        CandidateSpec(f"B_scale_{down_label}", False, lambda a, b, c: _scale_about_center(b, scale_down)),
        CandidateSpec("C_rot90", False, lambda a, b, c: np.rot90(c, 1)),
        CandidateSpec("B_rot90_plus_C", False, lambda a, b, c: np.rot90(b, 1) | c),
    )


def _transform_label(value: float) -> str:
    return f"{value:.2f}".replace(".", "")


def _shift(mask: np.ndarray, row_shift: int, col_shift: int) -> np.ndarray:
    result = np.zeros_like(mask, dtype=bool)
    src_r0, src_r1 = max(0, -row_shift), mask.shape[0] - max(0, row_shift)
    src_c0, src_c1 = max(0, -col_shift), mask.shape[1] - max(0, col_shift)
    dst_r0, dst_r1 = max(0, row_shift), mask.shape[0] - max(0, -row_shift)
    dst_c0, dst_c1 = max(0, col_shift), mask.shape[1] - max(0, -col_shift)
    result[dst_r0:dst_r1, dst_c0:dst_c1] = mask[src_r0:src_r1, src_c0:src_c1]
    return result


def _scale_about_center(mask: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        raise ValueError("scale must be positive")
    height, width = mask.shape
    rows, cols = np.indices(mask.shape)
    center = np.asarray([(height - 1) / 2.0, (width - 1) / 2.0], dtype=np.float32)
    source_rows = np.rint((rows - center[0]) / scale + center[0]).astype(np.int64)
    source_cols = np.rint((cols - center[1]) / scale + center[1]).astype(np.int64)
    in_bounds = (source_rows >= 0) & (source_rows < height) & (source_cols >= 0) & (source_cols < width)
    scaled = np.zeros_like(mask, dtype=bool)
    scaled[in_bounds] = mask[source_rows[in_bounds], source_cols[in_bounds]]
    return scaled


def _grid(defects: np.ndarray, valid: np.ndarray, *, role: str, source_id: int) -> GridMaps:
    status = np.full(defects.shape, BACKGROUND, dtype=np.uint8)
    status[valid] = VALID_NO_DEFECT
    status[defects & valid] = VALID_HAS_DEFECT
    count = (status == VALID_HAS_DEFECT).astype(np.float32)
    density = count / max(float(count.sum()), 1.0)
    return GridMaps(
        count_map=count,
        binary_map=count.astype(np.uint8),
        density_map=density,
        status_map=status,
        representation_map=count,
        representation_maps={"count": count, "binary": count.astype(np.uint8), "density": density},
        metadata={"source": "wm38k_synthetic", "role": role, "source_id": source_id},
    )


def _assign_ranks(rows: list[dict], score_name: str, rank_name: str) -> None:
    for rank, row in enumerate(sorted(rows, key=lambda item: (-item[score_name], item["candidate_name"])), start=1):
        row[rank_name] = rank


def _average_precision(rows: list[dict], rank_name: str) -> float:
    ranked = sorted(rows, key=lambda row: row[rank_name])
    relevant_total = sum(row["relevant_contains_a"] for row in ranked)
    hits = 0
    precision_sum = 0.0
    for rank, row in enumerate(ranked, start=1):
        if row["relevant_contains_a"]:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / max(relevant_total, 1)


def _recall_at_k(rows: list[dict], rank_name: str, k: int) -> float:
    relevant_total = sum(row["relevant_contains_a"] for row in rows)
    hits = sum(row["relevant_contains_a"] for row in rows if row[rank_name] <= k)
    return hits / max(relevant_total, 1)


def _mean_score(rows: list[dict], score_name: str, relevant: bool) -> float:
    values = [row[score_name] for row in rows if bool(row["relevant_contains_a"]) == relevant]
    return float(np.mean(values)) if values else 0.0


def _summary(group_rows: list[dict], args: argparse.Namespace) -> dict:
    def mean(name: str) -> float:
        return float(np.mean([row[name] for row in group_rows]))

    compact_wins = sum(row["compact_ap"] > row["iou_ap"] for row in group_rows)
    return {
        "benchmark": "wm38k_single_class_synthetic_compact",
        "groups": len(group_rows),
        "images_per_group": 20,
        "candidates_per_group": 19,
        "anchor_class": args.anchor_class,
        "b_class": args.b_class,
        "c_class": args.c_class,
        "proposal_mode": args.proposal_mode,
        "rotation_tolerance": True,
        "baseline": "strict_pixel_iou",
        "compact_macro_ap": mean("compact_ap"),
        "iou_macro_ap": mean("iou_ap"),
        "compact_macro_recall_at_5": mean("compact_recall_at_5"),
        "iou_macro_recall_at_5": mean("iou_recall_at_5"),
        "compact_positive_mean": mean("compact_positive_mean"),
        "compact_negative_mean": mean("compact_negative_mean"),
        "iou_positive_mean": mean("iou_positive_mean"),
        "iou_negative_mean": mean("iou_negative_mean"),
        "compact_ap_wins": compact_wins,
        "compact_ap_win_rate": compact_wins / max(len(group_rows), 1),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_config(args: argparse.Namespace) -> dict:
    config = vars(args).copy()
    config["data_file"] = str(args.data_file)
    config["out_dir"] = str(args.out_dir)
    return config


if __name__ == "__main__":
    main()
