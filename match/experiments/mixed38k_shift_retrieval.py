"""Mixed38K single-class shifted-target retrieval experiment.

Experiment 1:
  For each single pattern class, sample one single-class query map. The gallery
  contains 100 maps: one shifted copy of the query as the only positive target,
  and 99 single-class distractors from other pattern classes. Metrics are
  top-10/top-5/top-3/top-1 target retrieval accuracy and rank metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from match.core.local_matching import explain_count_partial_match
from match.core.models import BACKGROUND, GridMaps, VALID_HAS_DEFECT, VALID_NO_DEFECT


CLASS_NAMES = (
    "center",
    "donut",
    "edge-loc",
    "edge-ring",
    "loc",
    "random",
    "scratch",
    "near-full",
)
DEFAULT_DATA_FILE = WORKSPACE_ROOT / "data" / "wm38k" / "Wafer_Map_Datasets.npz"
DEFAULT_OUT_DIR = PROJECT_ROOT / "match" / "experiments" / "artifacts" / "mixed38k_shift_retrieval"
TOP_KS = (10, 5, 3, 1)


def main() -> None:
    args = parse_args()
    outputs = run_experiment(args)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False, indent=2))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mixed38K shifted-target retrieval experiment.")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE, help="Mixed38K/WM38K npz file.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--trials-per-class", type=int, default=20, help="Number of query trials for each single pattern class.")
    parser.add_argument("--gallery-size", type=int, default=100, help="Gallery size including the shifted positive target.")
    parser.add_argument("--shift-row", type=int, default=3)
    parser.add_argument("--shift-col", type=int, default=-3)
    parser.add_argument(
        "--proposal-mode",
        choices=("cc", "compact", "arc", "tangential-ring", "sparse-density", "auto"),
        default="compact",
    )
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--top-k-proposals", type=int, default=6)
    parser.add_argument("--token-match-top-k", type=int, default=3)
    parser.add_argument("--map-match-top-k", type=int, default=20)
    parser.add_argument("--min-token-score", type=float, default=0.30)
    parser.add_argument("--score-shape-weight", type=float, default=0.60)
    parser.add_argument("--score-position-weight", type=float, default=0.25)
    parser.add_argument("--score-scale-weight", type=float, default=0.15)
    parser.add_argument("--min-relative-token-area", type=float, default=0.10)
    parser.add_argument("--scale-area-weight", type=float, default=0.30)
    parser.add_argument("--scale-pca-weight", type=float, default=0.70)
    parser.add_argument("--scale-ratio-min", type=float, default=0.20)
    parser.add_argument("--density-sigmas", nargs="+", type=float, default=(0.8, 1.6, 3.2))
    parser.add_argument("--density-threshold", type=float, default=0.20)
    parser.add_argument("--density-min-raw-points", type=int, default=3)
    parser.add_argument("--density-min-raw-mass", type=float, default=3.0)
    parser.add_argument("--density-merge-iou", type=float, default=0.60)
    parser.add_argument("--density-weight-transform", choices=("count", "sqrt", "log1p"), default="sqrt")
    parser.add_argument("--rotation-tolerance", action="store_true")
    parser.add_argument("--save-figures", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def run_experiment(args: argparse.Namespace) -> Dict[str, Path]:
    if args.trials_per_class <= 0:
        raise ValueError("--trials-per-class must be positive")
    if args.gallery_size < 2:
        raise ValueError("--gallery-size must be at least 2")

    maps, labels, original_ids = _load_mixed38k(args.data_file)
    single_indices_by_class = _single_class_indices(labels)
    _validate_class_pools(single_indices_by_class, args.gallery_size - 1)

    rng = np.random.default_rng(args.seed)
    rows: List[Dict] = []
    trial_rows: List[Dict] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.out_dir / "figures"
    if args.save_figures:
        figure_dir.mkdir(parents=True, exist_ok=True)

    for class_idx, class_name in enumerate(CLASS_NAMES):
        query_pool = single_indices_by_class[class_idx]
        query_positions = rng.choice(query_pool, size=args.trials_per_class, replace=len(query_pool) < args.trials_per_class)
        for trial_idx, query_pos in enumerate(query_positions, start=1):
            query_pos = int(query_pos)
            query_raw = _normalize_map(maps[query_pos])
            valid = query_raw > 0
            query_defects = query_raw == VALID_HAS_DEFECT
            target_defects = _shift_mask(query_defects, args.shift_row, args.shift_col)
            query_grid = _grid_from_defects(query_defects, valid, role="query", map_id=int(original_ids[query_pos]), class_name=class_name)
            target_grid = _grid_from_defects(target_defects, valid, role="target_shifted", map_id=-1, class_name=class_name)

            distractor_positions = _sample_distractors(
                single_indices_by_class,
                exclude_class=class_idx,
                count=args.gallery_size - 1,
                rng=rng,
            )
            candidates = [{
                "candidate_id": "target_shifted",
                "source_map_id": int(original_ids[query_pos]),
                "class_name": class_name,
                "is_target": True,
                "grid": target_grid,
            }]
            for pos in distractor_positions:
                raw = _normalize_map(maps[int(pos)])
                label_idx = int(np.flatnonzero(labels[int(pos)].astype(np.int32) == 1)[0])
                candidates.append({
                    "candidate_id": f"map_{int(original_ids[int(pos)])}",
                    "source_map_id": int(original_ids[int(pos)]),
                    "class_name": CLASS_NAMES[label_idx],
                    "is_target": False,
                    "grid": _grid_from_raw(raw, role="distractor", map_id=int(original_ids[int(pos)]), class_name=CLASS_NAMES[label_idx]),
                })
            rng.shuffle(candidates)

            ranked = _rank_candidates(query_grid, candidates, args)
            target_rank = next(row["rank"] for row in ranked if row["is_target"])
            trial_id = f"{class_name}_trial{trial_idx:03d}"
            rank_percentile = _rank_percentile(target_rank, args.gallery_size)
            reciprocal_rank = 1.0 / float(target_rank)
            for row in ranked:
                row.update({
                    "trial_id": trial_id,
                    "query_class": class_name,
                    "query_source_map_id": int(original_ids[query_pos]),
                    "target_rank": int(target_rank),
                })
                rows.append(row)
            trial_summary = {
                "trial_id": trial_id,
                "query_class": class_name,
                "query_source_map_id": int(original_ids[query_pos]),
                "target_rank": int(target_rank),
                "reciprocal_rank": reciprocal_rank,
                "rank_percentile": rank_percentile,
                "distractor_class_counts": json.dumps(
                    _distractor_class_counts(labels, distractor_positions),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                **{f"hit_at_{k}": int(target_rank <= k) for k in TOP_KS},
            }
            trial_rows.append(trial_summary)
            if args.save_figures:
                _save_trial_figure(figure_dir / f"{trial_id}.png", query_grid, ranked[:10], title=trial_id)

    metrics = _metrics(trial_rows, args)
    rankings_path = args.out_dir / "rankings.csv"
    trials_path = args.out_dir / "trials.csv"
    metrics_path = args.out_dir / "metrics.json"
    config_path = args.out_dir / "config.json"
    _write_csv(rankings_path, rows, _ranking_fieldnames())
    _write_csv(trials_path, trial_rows, _trial_fieldnames())
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config_path.write_text(json.dumps(_config_dict(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "rankings": rankings_path,
        "trials": trials_path,
        "metrics": metrics_path,
        "config": config_path,
        "figures": figure_dir,
    }


def _load_mixed38k(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Mixed38K dataset not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        maps = _pick_array(data, ("maps", "x", "X", "images", "arr_0"))
        labels = _pick_array(data, ("labels", "y", "Y", "targets", "arr_1")).astype(np.int32)
    valid = labels.sum(axis=1) > 0
    return maps[valid], labels[valid], np.where(valid)[0].astype(np.int64)


def _pick_array(npz, names: Iterable[str]) -> np.ndarray:
    for name in names:
        if name in npz.files:
            return npz[name]
    raise KeyError(f"None of {tuple(names)} found in npz file. Available keys: {npz.files}")


def _single_class_indices(labels: np.ndarray) -> dict[int, np.ndarray]:
    result = {}
    single = labels.sum(axis=1) == 1
    for class_idx in range(len(CLASS_NAMES)):
        result[class_idx] = np.flatnonzero(single & (labels[:, class_idx] == 1)).astype(np.int64)
    return result


def _validate_class_pools(single_indices_by_class: dict[int, np.ndarray], distractor_count: int) -> None:
    missing = [CLASS_NAMES[idx] for idx, pool in single_indices_by_class.items() if len(pool) == 0]
    if missing:
        raise ValueError(f"No single-class samples for: {', '.join(missing)}")
    total_by_other = {
        CLASS_NAMES[idx]: sum(len(pool) for other_idx, pool in single_indices_by_class.items() if other_idx != idx)
        for idx in range(len(CLASS_NAMES))
    }
    too_small = {name: total for name, total in total_by_other.items() if total < distractor_count}
    if too_small:
        raise ValueError(f"Not enough other-class distractors for gallery: {too_small}")


def _sample_distractors(
    single_indices_by_class: dict[int, np.ndarray],
    exclude_class: int,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    other_classes = [idx for idx in range(len(CLASS_NAMES)) if idx != exclude_class]
    rng.shuffle(other_classes)
    base = count // len(other_classes)
    remainder = count % len(other_classes)
    selected = []
    for order, class_idx in enumerate(other_classes):
        quota = base + int(order < remainder)
        pool = single_indices_by_class[class_idx]
        selected.extend(rng.choice(pool, size=quota, replace=len(pool) < quota).astype(np.int64).tolist())
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def _rank_candidates(query: GridMaps, candidates: List[Dict], args: argparse.Namespace) -> List[Dict]:
    ranked = []
    for candidate in candidates:
        explanation = explain_count_partial_match(
            query,
            candidate["grid"],
            min_area=args.min_area,
            top_k=args.top_k_proposals,
            token_match_top_k=args.token_match_top_k,
            map_match_top_k=args.map_match_top_k,
            proposal_mode=args.proposal_mode,
            rotation_tolerance=args.rotation_tolerance,
            min_token_score=args.min_token_score,
            score_shape_weight=args.score_shape_weight,
            score_position_weight=args.score_position_weight,
            score_scale_weight=args.score_scale_weight,
            min_relative_token_area=args.min_relative_token_area,
            scale_area_weight=args.scale_area_weight,
            scale_pca_weight=args.scale_pca_weight,
            scale_ratio_min=args.scale_ratio_min,
            density_sigmas=tuple(args.density_sigmas),
            density_threshold=args.density_threshold,
            density_min_raw_points=args.density_min_raw_points,
            density_min_raw_mass=args.density_min_raw_mass,
            density_merge_iou=args.density_merge_iou,
            density_weight_transform=args.density_weight_transform,
        )
        result = explanation["result"]
        ranked.append({
            "candidate_id": candidate["candidate_id"],
            "candidate_source_map_id": candidate["source_map_id"],
            "candidate_class": candidate["class_name"],
            "is_target": int(candidate["is_target"]),
            "score": float(result.score),
            "mean_shape": float(result.mean_shape),
            "mean_position": float(result.mean_position),
            "mean_scale": float(result.mean_scale),
            "matched_tokens": int(result.matched_tokens),
            "query_tokens": int(result.wbm_tokens),
            "candidate_tokens": int(result.wdm_tokens),
            "_grid": candidate["grid"],
        })
    ranked.sort(key=lambda row: (-row["score"], str(row["candidate_id"])))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _metrics(trial_rows: List[Dict], args: argparse.Namespace) -> Dict:
    overall = {
        f"top{k}_accuracy": float(np.mean([row[f"hit_at_{k}"] for row in trial_rows])) if trial_rows else 0.0
        for k in TOP_KS
    }
    overall.update(_rank_metrics(trial_rows, args.gallery_size))
    per_class = {}
    for class_name in CLASS_NAMES:
        rows = [row for row in trial_rows if row["query_class"] == class_name]
        per_class[class_name] = {
            "trials": len(rows),
            **_rank_metrics(rows, args.gallery_size),
            **{
                f"top{k}_accuracy": float(np.mean([row[f"hit_at_{k}"] for row in rows])) if rows else 0.0
                for k in TOP_KS
            },
        }
    return {
        "dataset": "Mixed38K",
        "proposal_mode": args.proposal_mode,
        "gallery_size": int(args.gallery_size),
        "trials_per_class": int(args.trials_per_class),
        "total_trials": len(trial_rows),
        "shift": {"row": int(args.shift_row), "col": int(args.shift_col)},
        "overall": overall,
        "per_class": per_class,
    }


def _rank_metrics(rows: List[Dict], gallery_size: int) -> Dict[str, float]:
    if not rows:
        return {
            "mean_target_rank": 0.0,
            "median_target_rank": 0.0,
            "mrr": 0.0,
            "mean_rank_percentile": 0.0,
        }
    ranks = np.asarray([row["target_rank"] for row in rows], dtype=np.float64)
    return {
        "mean_target_rank": float(np.mean(ranks)),
        "median_target_rank": float(np.median(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank_percentile": float(np.mean([_rank_percentile(int(rank), gallery_size) for rank in ranks])),
    }


def _rank_percentile(rank: int, gallery_size: int) -> float:
    return float(1.0 - (float(rank) - 1.0) / max(float(gallery_size), 1.0))


def _distractor_class_counts(labels: np.ndarray, positions: np.ndarray) -> Dict[str, int]:
    counts = {class_name: 0 for class_name in CLASS_NAMES}
    for pos in positions:
        label_idx = int(np.flatnonzero(labels[int(pos)].astype(np.int32) == 1)[0])
        counts[CLASS_NAMES[label_idx]] += 1
    return {class_name: count for class_name, count in counts.items() if count > 0}


def _normalize_map(raw: np.ndarray) -> np.ndarray:
    raw = raw.astype(np.uint8)
    return np.where(raw >= 3, VALID_HAS_DEFECT, raw).astype(np.uint8)


def _grid_from_raw(raw: np.ndarray, role: str, map_id: int, class_name: str) -> GridMaps:
    raw = _normalize_map(raw)
    return _grid_from_defects(raw == VALID_HAS_DEFECT, raw > 0, role=role, map_id=map_id, class_name=class_name)


def _grid_from_defects(defects: np.ndarray, valid: np.ndarray, role: str, map_id: int, class_name: str) -> GridMaps:
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
        metadata={"source": "mixed38k", "role": role, "map_id": int(map_id), "class_name": class_name},
    )


def _shift_mask(mask: np.ndarray, row_shift: int, col_shift: int) -> np.ndarray:
    result = np.zeros_like(mask, dtype=bool)
    src_r0, src_r1 = max(0, -row_shift), mask.shape[0] - max(0, row_shift)
    src_c0, src_c1 = max(0, -col_shift), mask.shape[1] - max(0, col_shift)
    dst_r0, dst_r1 = max(0, row_shift), mask.shape[0] - max(0, -row_shift)
    dst_c0, dst_c1 = max(0, col_shift), mask.shape[1] - max(0, -col_shift)
    result[dst_r0:dst_r1, dst_c0:dst_c1] = mask[src_r0:src_r1, src_c0:src_c1]
    return result


def _save_trial_figure(path: Path, query: GridMaps, top10: List[Dict], title: str) -> None:
    _ensure_mpl()
    import matplotlib.pyplot as plt

    panels = [("query", query, False, 0.0)] + [
        (f"#{row['rank']} {row['candidate_class']}", row["_grid"], bool(row["is_target"]), float(row["score"]))
        for row in top10
    ]
    cols = 11
    fig, axes = plt.subplots(1, cols, figsize=(2.35 * cols, 2.7))
    for ax, (label, grid, is_target, score) in zip(np.asarray(axes).reshape(-1), panels):
        _draw_grid(ax, grid, label, is_target=is_target, score=score)
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.04, top=0.80, wspace=0.08)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_grid(ax, grid: GridMaps, label: str, is_target: bool, score: float) -> None:
    image = np.zeros((*grid.status_map.shape, 3), dtype=np.float32)
    valid = grid.status_map == VALID_NO_DEFECT
    defects = grid.status_map == VALID_HAS_DEFECT
    image[valid] = (0.58, 0.58, 0.58)
    image[defects] = (0.92, 0.92, 0.92)
    if is_target:
        image[defects] = (0.95, 0.12, 0.10)
    ax.imshow(image, interpolation="nearest")
    score_text = "" if label == "query" else f"\nscore={score:.3f}"
    target_text = "\nTARGET" if is_target else ""
    ax.set_title(f"{label}{target_text}{score_text}", fontsize=7, color="#b91c1c" if is_target else "black")
    for spine in ax.spines.values():
        spine.set_visible(is_target)
        spine.set_linewidth(2.5)
        spine.set_color("#dc2626")
    ax.set_xticks([])
    ax.set_yticks([])


def _ensure_mpl() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _ranking_fieldnames() -> List[str]:
    return [
        "trial_id",
        "query_class",
        "query_source_map_id",
        "target_rank",
        "rank",
        "candidate_id",
        "candidate_source_map_id",
        "candidate_class",
        "is_target",
        "score",
        "mean_shape",
        "mean_position",
        "mean_scale",
        "matched_tokens",
        "query_tokens",
        "candidate_tokens",
    ]


def _trial_fieldnames() -> List[str]:
    return [
        "trial_id",
        "query_class",
        "query_source_map_id",
        "target_rank",
        "reciprocal_rank",
        "rank_percentile",
        "distractor_class_counts",
    ] + [f"hit_at_{k}" for k in TOP_KS]


def _config_dict(args: argparse.Namespace) -> Dict:
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value
    return result


if __name__ == "__main__":
    main()
