"""Mixed38K class-level retrieval experiment.

For each single pattern class, sample one single-class query map. The gallery
contains same-label positives and other-label negatives. Metrics include
precision@k, recall@k, hit@k, AP, nDCG, MRR, and positive-rank summaries.
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
from tqdm import tqdm

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
DEFAULT_OUT_DIR = PROJECT_ROOT / "match" / "experiments" / "artifacts" / "mixed38k_class_retrieval"
TOP_KS = (10, 5, 3, 1)
TOKEN_COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#ca8a04",
    "#9333ea",
    "#0891b2",
    "#ea580c",
    "#4f46e5",
    "#65a30d",
    "#be185d",
)


def main() -> None:
    args = parse_args()
    outputs = run_experiment(args)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False, indent=2))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mixed38K class-level retrieval experiment.")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE, help="Mixed38K/WM38K npz file.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--class-name",
        choices=CLASS_NAMES,
        default=None,
        help="Optional single class to evaluate. Default evaluates all classes.",
    )
    parser.add_argument("--trials-per-class", type=int, default=20, help="Number of query trials for each single pattern class.")
    parser.add_argument("--positives-per-trial", type=int, default=10, help="Same-label positive gallery maps per trial.")
    parser.add_argument("--negatives-per-trial", type=int, default=90, help="Other-label negative gallery maps per trial.")
    parser.add_argument(
        "--proposal-mode",
        choices=("cc", "compact", "arc", "arc-band-residual", "arc-ring-residual", "tangential-ring", "sparse-density", "auto"),
        default="arc-ring-residual",
    )
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--top-k-proposals", type=int, default=6)
    parser.add_argument("--token-match-top-k", type=int, default=3)
    parser.add_argument("--map-match-top-k", type=int, default=20)
    parser.add_argument("--min-token-score", type=float, default=0.30)
    parser.add_argument("--sigma-pos", type=float, default=0.35, help="Position affinity width for token matching.")
    parser.add_argument("--sigma-scale", type=float, default=1.5, help="Scale affinity width for token matching.")
    parser.add_argument("--score-shape-weight", type=float, default=0.60)
    parser.add_argument("--score-position-weight", type=float, default=0.25)
    parser.add_argument("--score-scale-weight", type=float, default=0.15)
    parser.add_argument("--min-relative-token-area", type=float, default=0.10)
    parser.add_argument("--scale-area-weight", type=float, default=0.30)
    parser.add_argument("--scale-pca-weight", type=float, default=0.70)
    parser.add_argument("--scale-ratio-min", type=float, default=0.50)
    parser.add_argument("--density-sigmas", nargs="+", type=float, default=(0.8, 1.6, 3.2))
    parser.add_argument("--density-threshold", type=float, default=0.20)
    parser.add_argument("--density-min-raw-points", type=int, default=3)
    parser.add_argument("--density-min-raw-mass", type=float, default=3.0)
    parser.add_argument("--density-merge-iou", type=float, default=0.60)
    parser.add_argument("--density-weight-transform", choices=("count", "sqrt", "log1p"), default="sqrt")
    parser.add_argument("--rotation-tolerance", action="store_true")
    parser.add_argument("--save-figures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-figures", type=int, default=None, help="Maximum number of trial figures to save. Default saves all figures.")
    parser.add_argument("--figure-top-k", type=int, default=5, help="Top ranked candidates to include in each trial figure.")
    return parser.parse_args(argv)


def run_experiment(args: argparse.Namespace) -> Dict[str, Path]:
    if args.trials_per_class <= 0:
        raise ValueError("--trials-per-class must be positive")
    if args.positives_per_trial <= 0:
        raise ValueError("--positives-per-trial must be positive")
    if args.negatives_per_trial <= 0:
        raise ValueError("--negatives-per-trial must be positive")

    maps, labels, original_ids = _load_mixed38k(args.data_file)
    single_indices_by_class = _single_class_indices(labels)
    _validate_class_pools(single_indices_by_class, args.positives_per_trial, args.negatives_per_trial)

    rng = np.random.default_rng(args.seed)
    ranking_rows: List[Dict] = []
    trial_rows: List[Dict] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.out_dir / "figures"
    figure_count = 0
    max_figures = None if args.max_figures is None else max(int(args.max_figures), 0)
    if args.save_figures and max_figures != 0:
        figure_dir.mkdir(parents=True, exist_ok=True)

    class_indices = _selected_class_indices(args.class_name)
    for class_idx in class_indices:
        class_name = CLASS_NAMES[class_idx]
        query_pool = single_indices_by_class[class_idx]
        query_positions = rng.choice(query_pool, size=args.trials_per_class, replace=len(query_pool) < args.trials_per_class)
        pbar = tqdm(enumerate(query_positions, start=1), total=args.trials_per_class, desc=f"Class: {class_name}", unit="trial")
        for trial_idx, query_pos in pbar:
            query_pos = int(query_pos)
            trial_id = f"{class_name}_trial{trial_idx:03d}"
            query_grid = _grid_from_raw(
                maps[query_pos],
                role="query",
                map_id=int(original_ids[query_pos]),
                class_name=class_name,
            )
            positive_positions = _sample_positives(
                single_indices_by_class[class_idx],
                query_pos=query_pos,
                count=args.positives_per_trial,
                rng=rng,
            )
            negative_positions = _sample_negatives_balanced(
                single_indices_by_class,
                exclude_class=class_idx,
                count=args.negatives_per_trial,
                rng=rng,
            )
            candidates = _candidate_records(
                maps,
                labels,
                original_ids,
                positive_positions,
                is_positive=True,
            )
            candidates.extend(_candidate_records(
                maps,
                labels,
                original_ids,
                negative_positions,
                is_positive=False,
            ))
            rng.shuffle(candidates)

            ranked = _rank_candidates(query_grid, candidates, args)
            trial_metrics = _trial_metrics(ranked, args.positives_per_trial)
            pbar.set_postfix(rank_1st=trial_metrics["first_positive_rank"])
            for row in ranked:
                row.update({
                    "trial_id": trial_id,
                    "query_class": class_name,
                    "query_source_map_id": int(original_ids[query_pos]),
                })
                ranking_rows.append(row)
            trial_rows.append({
                "trial_id": trial_id,
                "query_class": class_name,
                "query_source_map_id": int(original_ids[query_pos]),
                "positive_class_counts": json.dumps(_class_counts(labels, positive_positions), ensure_ascii=False, sort_keys=True),
                "negative_class_counts": json.dumps(_class_counts(labels, negative_positions), ensure_ascii=False, sort_keys=True),
                **trial_metrics,
            })
            if args.save_figures and (max_figures is None or figure_count < max_figures):
                _save_trial_figure(
                    figure_dir / f"{trial_id}.png",
                    query_grid,
                    ranked,
                    title=trial_id,
                    top_k=args.figure_top_k,
                )
                figure_count += 1

    metrics = _metrics(trial_rows, args)
    rankings_path = args.out_dir / "rankings.csv"
    trials_path = args.out_dir / "trials.csv"
    metrics_path = args.out_dir / "metrics.json"
    config_path = args.out_dir / "config.json"
    _write_csv(rankings_path, ranking_rows, _ranking_fieldnames())
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
    single = labels.sum(axis=1) == 1
    return {
        class_idx: np.flatnonzero(single & (labels[:, class_idx] == 1)).astype(np.int64)
        for class_idx in range(len(CLASS_NAMES))
    }


def _selected_class_indices(class_name: str | None) -> List[int]:
    if class_name is None:
        return list(range(len(CLASS_NAMES)))
    return [CLASS_NAMES.index(class_name)]


def _validate_class_pools(single_indices_by_class: dict[int, np.ndarray], positive_count: int, negative_count: int) -> None:
    missing = [CLASS_NAMES[idx] for idx, pool in single_indices_by_class.items() if len(pool) == 0]
    if missing:
        raise ValueError(f"No single-class samples for: {', '.join(missing)}")
    too_few_positives = {
        CLASS_NAMES[idx]: len(pool)
        for idx, pool in single_indices_by_class.items()
        if len(pool) <= positive_count
    }
    if too_few_positives:
        raise ValueError(f"Not enough same-class positives excluding the query: {too_few_positives}")
    total_by_other = {
        CLASS_NAMES[idx]: sum(len(pool) for other_idx, pool in single_indices_by_class.items() if other_idx != idx)
        for idx in range(len(CLASS_NAMES))
    }
    too_few_negatives = {name: total for name, total in total_by_other.items() if total < negative_count}
    if too_few_negatives:
        raise ValueError(f"Not enough other-class negatives: {too_few_negatives}")


def _sample_positives(pool: np.ndarray, query_pos: int, count: int, rng: np.random.Generator) -> np.ndarray:
    candidates = pool[pool != query_pos]
    return rng.choice(candidates, size=count, replace=len(candidates) < count).astype(np.int64)


def _sample_negatives_balanced(
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


def _candidate_records(
    maps: np.ndarray,
    labels: np.ndarray,
    original_ids: np.ndarray,
    positions: np.ndarray,
    is_positive: bool,
) -> List[Dict]:
    records = []
    for pos in positions:
        pos = int(pos)
        class_idx = int(np.flatnonzero(labels[pos].astype(np.int32) == 1)[0])
        class_name = CLASS_NAMES[class_idx]
        records.append({
            "candidate_id": f"map_{int(original_ids[pos])}",
            "source_map_id": int(original_ids[pos]),
            "class_name": class_name,
            "is_positive": bool(is_positive),
            "grid": _grid_from_raw(maps[pos], role="candidate", map_id=int(original_ids[pos]), class_name=class_name),
        })
    return records


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
            sigma_pos=args.sigma_pos,
            sigma_scale=args.sigma_scale,
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
            "is_positive": int(candidate["is_positive"]),
            "score": float(result.score),
            "mean_shape": float(result.mean_shape),
            "mean_position": float(result.mean_position),
            "mean_scale": float(result.mean_scale),
            "matched_tokens": int(result.matched_tokens),
            "query_tokens": int(result.wbm_tokens),
            "candidate_tokens": int(result.wdm_tokens),
            "_query_tokens": explanation["wbm_tokens"],
            "_tokens": explanation["wdm_tokens"],
            "_grid": candidate["grid"],
        })
    ranked.sort(key=lambda row: (-row["score"], str(row["candidate_id"])))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _trial_metrics(ranked: List[Dict], positive_count: int) -> Dict[str, float]:
    positive_ranks = [int(row["rank"]) for row in ranked if row["is_positive"]]
    metrics: Dict[str, float] = {
        "positive_count": int(positive_count),
        "mean_positive_rank": float(np.mean(positive_ranks)),
        "median_positive_rank": float(np.median(positive_ranks)),
        "first_positive_rank": int(min(positive_ranks)),
        "mrr": 1.0 / float(min(positive_ranks)),
        "average_precision": _average_precision(ranked, positive_count),
        "ndcg": _ndcg_at_k(ranked, len(ranked), positive_count),
    }
    for k in TOP_KS:
        positives_at_k = sum(int(row["is_positive"]) for row in ranked[:k])
        metrics[f"positives_at_{k}"] = int(positives_at_k)
        metrics[f"precision_at_{k}"] = float(positives_at_k / float(k))
        metrics[f"recall_at_{k}"] = float(positives_at_k / float(positive_count))
        metrics[f"hit_at_{k}"] = int(positives_at_k > 0)
        metrics[f"ndcg_at_{k}"] = _ndcg_at_k(ranked, k, positive_count)
    return metrics


def _average_precision(ranked: List[Dict], positive_count: int) -> float:
    hit_count = 0
    precision_sum = 0.0
    for rank, row in enumerate(ranked, start=1):
        if row["is_positive"]:
            hit_count += 1
            precision_sum += hit_count / float(rank)
    return float(precision_sum / float(max(positive_count, 1)))


def _ndcg_at_k(ranked: List[Dict], k: int, positive_count: int) -> float:
    gains = np.asarray([int(row["is_positive"]) for row in ranked[:k]], dtype=np.float64)
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2, dtype=np.float64))
    dcg = float(np.sum(gains * discounts))
    ideal_len = min(k, positive_count)
    if ideal_len <= 0:
        return 0.0
    ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_len + 2, dtype=np.float64))
    return float(dcg / float(np.sum(ideal_discounts)))


def _metrics(trial_rows: List[Dict], args: argparse.Namespace) -> Dict:
    metric_keys = _metric_fieldnames()
    overall = _mean_metrics(trial_rows, metric_keys)
    per_class = {}
    class_names = [CLASS_NAMES[idx] for idx in _selected_class_indices(args.class_name)]
    for class_name in class_names:
        rows = [row for row in trial_rows if row["query_class"] == class_name]
        per_class[class_name] = {
            "trials": len(rows),
            **_mean_metrics(rows, metric_keys),
        }
    return {
        "dataset": "Mixed38K",
        "task": "class_retrieval",
        "proposal_mode": args.proposal_mode,
        "class_name": args.class_name,
        "classes": class_names,
        "positives_per_trial": int(args.positives_per_trial),
        "negatives_per_trial": int(args.negatives_per_trial),
        "gallery_size": int(args.positives_per_trial + args.negatives_per_trial),
        "trials_per_class": int(args.trials_per_class),
        "total_trials": len(trial_rows),
        "overall": overall,
        "per_class": per_class,
    }


def _mean_metrics(rows: List[Dict], keys: List[str]) -> Dict[str, float]:
    if not rows:
        return {key: 0.0 for key in keys}
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def _class_counts(labels: np.ndarray, positions: np.ndarray) -> Dict[str, int]:
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


def _save_trial_figure(path: Path, query: GridMaps, ranked: List[Dict], title: str, top_k: int) -> None:
    _ensure_mpl()
    import matplotlib.pyplot as plt

    selected = _figure_candidate_rows(ranked, top_k=top_k)
    query_tokens = ranked[0].get("_query_tokens", []) if ranked else []
    panels = [{
        "label": "query",
        "grid": query,
        "tokens": query_tokens,
        "is_positive": False,
        "score": 0.0,
    }] + selected
    cols = len(panels)
    fig, axes = plt.subplots(2, cols, figsize=(2.05 * cols, 5.1))
    axes = np.asarray(axes)
    for col, panel in enumerate(panels):
        _draw_grid(
            axes[0, col],
            panel["grid"],
            panel["label"],
            is_positive=bool(panel["is_positive"]),
            score=float(panel["score"]),
        )
        _draw_proposal(
            axes[1, col],
            panel["grid"],
            panel["tokens"],
            is_positive=bool(panel["is_positive"]),
        )
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.03, top=0.88, wspace=0.08, hspace=0.16)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _figure_candidate_rows(ranked: List[Dict], top_k: int) -> List[Dict]:
    top_k = max(int(top_k), 0)
    top_ids = {id(row) for row in ranked[:top_k]}
    rows = list(ranked[:top_k])
    rows.extend(row for row in ranked if row["is_positive"] and id(row) not in top_ids)
    panels = []
    for row in rows:
        prefix = f"#{row['rank']}"
        if row["is_positive"] and id(row) not in top_ids:
            prefix = f"target #{row['rank']}"
        panels.append({
            "label": f"{prefix} {row['candidate_class']}",
            "grid": row["_grid"],
            "tokens": row.get("_tokens", []),
            "is_positive": bool(row["is_positive"]),
            "score": float(row["score"]),
        })
    return panels


def _draw_grid(ax, grid: GridMaps, label: str, is_positive: bool, score: float) -> None:
    image = np.zeros((*grid.status_map.shape, 3), dtype=np.float32)
    valid = grid.status_map == VALID_NO_DEFECT
    defects = grid.status_map == VALID_HAS_DEFECT
    image[valid] = (0.58, 0.58, 0.58)
    image[defects] = (0.92, 0.92, 0.92)
    if is_positive:
        image[defects] = (0.95, 0.12, 0.10)
    ax.imshow(image, interpolation="nearest")
    score_text = "" if label == "query" else f"\nscore={score:.3f}"
    positive_text = "\nPOS" if is_positive else ""
    ax.set_title(f"{label}{positive_text}{score_text}", fontsize=7, color="#b91c1c" if is_positive else "black")
    for spine in ax.spines.values():
        spine.set_visible(is_positive)
        spine.set_linewidth(2.5)
        spine.set_color("#dc2626")
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_proposal(ax, grid: GridMaps, tokens: List[Dict], is_positive: bool) -> None:
    image = _proposal_image(grid, tokens)
    ax.imshow(image, interpolation="nearest")
    ax.set_title(f"proposal n={len(tokens)}", fontsize=7, color="#b91c1c" if is_positive else "black")
    for spine in ax.spines.values():
        spine.set_visible(is_positive)
        spine.set_linewidth(2.0)
        spine.set_color("#dc2626")
    ax.set_xticks([])
    ax.set_yticks([])


def _proposal_image(grid: GridMaps, tokens: List[Dict]) -> np.ndarray:
    h, w = grid.status_map.shape
    image = np.zeros((h, w, 3), dtype=np.float32)
    valid = (grid.status_map == VALID_NO_DEFECT) | (grid.status_map == VALID_HAS_DEFECT)
    defects = grid.status_map == VALID_HAS_DEFECT
    image[valid] = (0.48, 0.48, 0.48)
    image[defects] = (0.92, 0.92, 0.92)
    for idx, token in enumerate(tokens):
        color = np.asarray(_hex_to_rgb(TOKEN_COLORS[idx % len(TOKEN_COLORS)]), dtype=np.float32)
        support_color = 0.45 * color
        for row, col in _token_visual_support_pixels(token):
            if 0 <= row < h and 0 <= col < w:
                image[row, col] = np.clip(0.55 * image[row, col] + support_color, 0.0, 1.0)
        for row, col in token.get("pixels", []):
            row = int(row)
            col = int(col)
            if 0 <= row < h and 0 <= col < w:
                image[row, col] = color
    return image


def _token_visual_support_pixels(token: Dict) -> List[tuple[int, int]]:
    if token.get("kde_support_pixels"):
        return [(int(row), int(col)) for row, col in token["kde_support_pixels"]]
    if token.get("ring_contour_pixels"):
        return [(int(row), int(col)) for row, col in token["ring_contour_pixels"]]
    return []


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    value = hex_color.lstrip("#")
    return (
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
    )


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
        "rank",
        "candidate_id",
        "candidate_source_map_id",
        "candidate_class",
        "is_positive",
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
        "positive_class_counts",
        "negative_class_counts",
    ] + _metric_fieldnames()


def _metric_fieldnames() -> List[str]:
    keys = [
        "positive_count",
        "mean_positive_rank",
        "median_positive_rank",
        "first_positive_rank",
        "mrr",
        "average_precision",
        "ndcg",
    ]
    for k in TOP_KS:
        keys.extend([
            f"positives_at_{k}",
            f"precision_at_{k}",
            f"recall_at_{k}",
            f"hit_at_{k}",
            f"ndcg_at_{k}",
        ])
    return keys


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
