"""Mixed38K transformed-target retrieval experiment.

For each single-label query map, generate transformed copies of the same map
as positive targets. The gallery also contains random and hard negative real
maps. The primary metric is top-5 target hit rate.
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
DEFAULT_OUT_DIR = PROJECT_ROOT / "match" / "experiments" / "artifacts" / "results_of_each_testss"
TOP_KS = (10, 5, 3, 1)
TRANSFORM_TYPES = ("translate", "rotate", "scale", "affine", "corrupted_affine")


def main() -> None:
    args = parse_args()
    outputs = run_experiment(args)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False, indent=2))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mixed38K transformed-target retrieval experiment.")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE, help="Mixed38K/WM38K npz file.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--class-name",
        choices=CLASS_NAMES,
        default=None,
        help="Optional single class to evaluate. Default evaluates all classes.",
    )
    parser.add_argument("--trials-per-class", type=int, default=10, help="Number of query trials for each selected class.")
    parser.add_argument("--positives-per-trial", type=int, default=5, help="Transformed targets per query.")
    parser.add_argument("--negatives-per-trial", type=int, default=95, help="Real-map distractors per query.")
    parser.add_argument("--random-negatives", type=int, default=70)
    parser.add_argument("--same-class-negatives", type=int, default=0)
    parser.add_argument("--hard-negatives", type=int, default=25)
    parser.add_argument("--min-defects", type=int, default=5, help="Minimum query defect count.")
    parser.add_argument("--max-defect-ratio", type=float, default=1.00, help="Maximum query defect fraction of valid area.")
    parser.add_argument("--max-transform-retries", type=int, default=20)
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
    parser.add_argument("--sigma-pos", type=float, default=0.35)
    parser.add_argument("--sigma-scale", type=float, default=1.5)
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
    parser.add_argument("--max-figures", type=int, default=None, help="Maximum number of top-10 figures to save.")
    return parser.parse_args(argv)


def run_experiment(args: argparse.Namespace) -> Dict[str, Path]:
    _validate_args(args)
    maps, labels, original_ids = _load_mixed38k(args.data_file)
    maps = np.asarray([_normalize_map(raw) for raw in maps], dtype=np.uint8)
    single_indices_by_class = _single_class_indices(labels)
    eligible_indices_by_class = _eligible_single_class_indices(maps, labels, args)
    _validate_pools(single_indices_by_class, eligible_indices_by_class, args)
    features = _map_features(maps)

    rng = np.random.default_rng(args.seed)
    ranking_rows: List[Dict] = []
    trial_rows: List[Dict] = []
    target_rows: List[Dict] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.out_dir / "figures"
    figure_count = 0
    max_figures = None if args.max_figures is None else max(int(args.max_figures), 0)
    if args.save_figures and max_figures != 0:
        figure_dir.mkdir(parents=True, exist_ok=True)

    class_indices = _selected_class_indices(args.class_name)
    for class_idx in class_indices:
        class_name = CLASS_NAMES[class_idx]
        query_pool = eligible_indices_by_class[class_idx]
        query_positions = rng.choice(query_pool, size=args.trials_per_class, replace=len(query_pool) < args.trials_per_class)
        pbar = tqdm(enumerate(query_positions, start=1), total=args.trials_per_class, desc=f"Class: {class_name}", unit="trial")
        for trial_idx, query_pos in pbar:
            query_pos = int(query_pos)
            query_raw = maps[query_pos]
            valid = query_raw > 0
            query_defects = query_raw == VALID_HAS_DEFECT
            query_grid = _grid_from_defects(
                query_defects,
                valid,
                role="query",
                map_id=int(original_ids[query_pos]),
                class_name=class_name,
            )
            trial_id = f"{class_name}_trial{trial_idx:03d}"
            positives = _positive_records(
                query_defects=query_defects,
                valid=valid,
                source_map_id=int(original_ids[query_pos]),
                class_name=class_name,
                count=args.positives_per_trial,
                rng=rng,
                max_retries=args.max_transform_retries,
            )
            negative_positions, negative_sources = _sample_negative_positions(
                maps,
                labels,
                single_indices_by_class,
                features,
                query_pos=query_pos,
                query_class_idx=class_idx,
                args=args,
                rng=rng,
            )
            negatives = _negative_records(maps, labels, original_ids, negative_positions, negative_sources)
            candidates = positives + negatives
            rng.shuffle(candidates)

            ranked = _rank_candidates(query_grid, candidates, args)
            trial_metrics = _trial_metrics(ranked, target_count=len(positives))
            pbar.set_postfix(top5=trial_metrics["hit_at_5"], best=trial_metrics["best_target_rank"])
            for row in ranked:
                row.update({
                    "trial_id": trial_id,
                    "query_class": class_name,
                    "query_source_map_id": int(original_ids[query_pos]),
                    "best_target_rank": int(trial_metrics["best_target_rank"]),
                })
                ranking_rows.append(row)
            trial_rows.append({
                "trial_id": trial_id,
                "query_class": class_name,
                "query_source_map_id": int(original_ids[query_pos]),
                "positive_count": len(positives),
                "negative_count": len(negatives),
                "negative_source_counts": json.dumps(_counts(negative_sources), ensure_ascii=False, sort_keys=True),
                "negative_class_counts": json.dumps(_class_counts(labels, negative_positions), ensure_ascii=False, sort_keys=True),
                **trial_metrics,
            })
            target_rows.extend(_target_rank_rows(trial_id, class_name, int(original_ids[query_pos]), ranked))
            if args.save_figures and (max_figures is None or figure_count < max_figures):
                _save_trial_figure(figure_dir / f"{trial_id}.png", query_grid, ranked[:10], title=trial_id)
                figure_count += 1

    metrics = _metrics(trial_rows, target_rows, args)
    rankings_path = args.out_dir / "rankings.csv"
    trials_path = args.out_dir / "trials.csv"
    targets_path = args.out_dir / "targets.csv"
    metrics_path = args.out_dir / "metrics.json"
    config_path = args.out_dir / "config.json"
    _write_csv(rankings_path, ranking_rows, _ranking_fieldnames())
    _write_csv(trials_path, trial_rows, _trial_fieldnames())
    _write_csv(targets_path, target_rows, _target_fieldnames())
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config_path.write_text(json.dumps(_config_dict(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "rankings": rankings_path,
        "trials": trials_path,
        "targets": targets_path,
        "metrics": metrics_path,
        "config": config_path,
        "figures": figure_dir,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.trials_per_class <= 0:
        raise ValueError("--trials-per-class must be positive")
    if args.positives_per_trial <= 0:
        raise ValueError("--positives-per-trial must be positive")
    if args.negatives_per_trial <= 0:
        raise ValueError("--negatives-per-trial must be positive")
    if args.random_negatives < 0 or args.same_class_negatives < 0 or args.hard_negatives < 0:
        raise ValueError("negative mix counts must be non-negative")
    mix_total = args.random_negatives + args.same_class_negatives + args.hard_negatives
    if mix_total != args.negatives_per_trial:
        raise ValueError("random/same-class/hard negatives must sum to --negatives-per-trial")


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


def _eligible_single_class_indices(maps: np.ndarray, labels: np.ndarray, args: argparse.Namespace) -> dict[int, np.ndarray]:
    single = labels.sum(axis=1) == 1
    eligible = {}
    for class_idx in range(len(CLASS_NAMES)):
        class_positions = np.flatnonzero(single & (labels[:, class_idx] == 1)).astype(np.int64)
        keep = []
        for pos in class_positions:
            raw = maps[int(pos)]
            valid_area = int((raw > 0).sum())
            defect_count = int((raw == VALID_HAS_DEFECT).sum())
            if valid_area <= 0:
                continue
            if defect_count < args.min_defects:
                continue
            if defect_count > valid_area * float(args.max_defect_ratio):
                continue
            keep.append(int(pos))
        eligible[class_idx] = np.asarray(keep, dtype=np.int64)
    return eligible


def _selected_class_indices(class_name: str | None) -> List[int]:
    if class_name is None:
        return list(range(len(CLASS_NAMES)))
    return [CLASS_NAMES.index(class_name)]


def _validate_pools(
    single_indices_by_class: dict[int, np.ndarray],
    eligible_indices_by_class: dict[int, np.ndarray],
    args: argparse.Namespace,
) -> None:
    selected = _selected_class_indices(args.class_name)
    missing_queries = [CLASS_NAMES[idx] for idx in selected if len(eligible_indices_by_class[idx]) == 0]
    if missing_queries:
        raise ValueError(f"No eligible query samples for: {', '.join(missing_queries)}")
    for class_idx in selected:
        same_class_pool = single_indices_by_class[class_idx]
        if len(same_class_pool) <= 1 and args.same_class_negatives > 0:
            raise ValueError(f"Not enough same-class negatives for: {CLASS_NAMES[class_idx]}")
        other_total = sum(len(pool) for idx, pool in single_indices_by_class.items() if idx != class_idx)
        if other_total <= 0 and (args.random_negatives > 0 or args.hard_negatives > 0):
            raise ValueError(f"Not enough non-query-class negatives for: {CLASS_NAMES[class_idx]}")


def _map_features(maps: np.ndarray, bins: int = 8) -> List[Dict]:
    return [_single_map_features(raw, bins=bins) for raw in maps]


def _single_map_features(raw: np.ndarray, bins: int = 8) -> Dict:
    valid = raw > 0
    defects = raw == VALID_HAS_DEFECT
    points = np.argwhere(defects).astype(np.float32)
    center, radius_ref = _wafer_center_and_radius(valid)
    if len(points) == 0:
        return {
            "defect_count": 0,
            "centroid": np.array([0.5, 0.5], dtype=np.float32),
            "radial_hist": np.zeros(bins, dtype=np.float32),
        }
    centroid = points.mean(axis=0)
    radial = np.linalg.norm(points - center, axis=1) / max(radius_ref, 1e-6)
    hist, _ = np.histogram(radial, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float32)
    hist /= max(float(np.linalg.norm(hist)), 1e-8)
    h, w = raw.shape
    return {
        "defect_count": int(len(points)),
        "centroid": np.array([centroid[0] / max(h, 1), centroid[1] / max(w, 1)], dtype=np.float32),
        "radial_hist": hist,
    }


def _positive_records(
    query_defects: np.ndarray,
    valid: np.ndarray,
    source_map_id: int,
    class_name: str,
    count: int,
    rng: np.random.Generator,
    max_retries: int,
) -> List[Dict]:
    records = []
    transform_types = [TRANSFORM_TYPES[idx % len(TRANSFORM_TYPES)] for idx in range(count)]
    for idx, transform_type in enumerate(transform_types, start=1):
        defects, params = _generate_valid_transform(query_defects, valid, transform_type, rng, max_retries)
        candidate_id = f"map_{source_map_id}_target{idx:02d}_{transform_type}"
        records.append({
            "candidate_id": candidate_id,
            "source_map_id": source_map_id,
            "class_name": class_name,
            "is_target": True,
            "negative_source": "",
            "transform_type": transform_type,
            "transform_params": json.dumps(params, ensure_ascii=False, sort_keys=True),
            "grid": _grid_from_defects(defects, valid, role="target_transformed", map_id=-idx, class_name=class_name),
        })
    return records


def _generate_valid_transform(
    query_defects: np.ndarray,
    valid: np.ndarray,
    transform_type: str,
    rng: np.random.Generator,
    max_retries: int,
) -> tuple[np.ndarray, Dict]:
    original_count = int(query_defects.sum())
    valid_area = int(valid.sum())
    min_count = max(3, int(np.ceil(original_count * 0.30)))
    for _ in range(max(int(max_retries), 1)):
        params = _sample_transform_params(transform_type, rng)
        transformed = _apply_transform(query_defects, valid, params)
        if params.get("dropout_rate", 0.0) > 0.0:
            transformed = _apply_dropout(transformed, float(params["dropout_rate"]), rng)
        if params.get("noise_rate", 0.0) > 0.0:
            noise_count = int(round(original_count * float(params["noise_rate"])))
            transformed = _add_noise(transformed, valid, noise_count, rng)
        transformed &= valid
        count = int(transformed.sum())
        if count < min_count or count > valid_area * 0.80:
            continue
        if _mask_iou(query_defects, transformed) >= 0.98:
            continue
        return transformed, params
    params = _sample_transform_params("translate", rng)
    params.update({"fallback": True, "dx": 3, "dy": -3})
    transformed = _apply_transform(query_defects, valid, params) & valid
    return transformed, params


def _sample_transform_params(transform_type: str, rng: np.random.Generator) -> Dict:
    if transform_type == "translate":
        dx, dy = 0, 0
        while np.hypot(dx, dy) < 2.0:
            dx = int(rng.integers(-4, 5))
            dy = int(rng.integers(-4, 5))
        return {"type": transform_type, "dx": dx, "dy": dy, "angle": 0.0, "scale": 1.0}
    if transform_type == "rotate":
        return {
            "type": transform_type,
            "dx": 0,
            "dy": 0,
            "angle": float(rng.choice([-30, -20, -10, 10, 20, 30])),
            "scale": 1.0,
        }
    if transform_type == "scale":
        return {
            "type": transform_type,
            "dx": 0,
            "dy": 0,
            "angle": 0.0,
            "scale": float(rng.choice([0.80, 0.90, 1.10, 1.20])),
        }
    if transform_type == "affine":
        return {
            "type": transform_type,
            "dx": int(rng.integers(-3, 4)),
            "dy": int(rng.integers(-3, 4)),
            "angle": float(rng.choice([-20, -10, 10, 20])),
            "scale": float(rng.choice([0.85, 0.95, 1.05, 1.15])),
        }
    if transform_type == "corrupted_affine":
        return {
            "type": transform_type,
            "dx": int(rng.integers(-3, 4)),
            "dy": int(rng.integers(-3, 4)),
            "angle": float(rng.choice([-20, -10, 10, 20])),
            "scale": float(rng.choice([0.85, 0.95, 1.05, 1.15])),
            "dropout_rate": float(rng.choice([0.10, 0.20, 0.30])),
            "noise_rate": float(rng.choice([0.05, 0.10])),
        }
    raise ValueError(f"Unsupported transform type: {transform_type}")


def _apply_transform(mask: np.ndarray, valid: np.ndarray, params: Dict) -> np.ndarray:
    points = np.argwhere(mask).astype(np.float32)
    result = np.zeros_like(mask, dtype=bool)
    if len(points) == 0:
        return result
    center, _ = _wafer_center_and_radius(valid)
    rel = (points - center) * float(params.get("scale", 1.0))
    angle = np.deg2rad(float(params.get("angle", 0.0)))
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    rotated = np.empty_like(rel)
    rotated[:, 0] = cos_a * rel[:, 0] - sin_a * rel[:, 1]
    rotated[:, 1] = sin_a * rel[:, 0] + cos_a * rel[:, 1]
    transformed = rotated + center
    transformed[:, 0] += float(params.get("dy", 0))
    transformed[:, 1] += float(params.get("dx", 0))
    ij = np.rint(transformed).astype(np.int64)
    keep = (
        (ij[:, 0] >= 0)
        & (ij[:, 0] < mask.shape[0])
        & (ij[:, 1] >= 0)
        & (ij[:, 1] < mask.shape[1])
    )
    ij = ij[keep]
    if len(ij):
        ij = ij[valid[ij[:, 0], ij[:, 1]]]
    if len(ij):
        result[ij[:, 0], ij[:, 1]] = True
    return result


def _apply_dropout(mask: np.ndarray, dropout_rate: float, rng: np.random.Generator) -> np.ndarray:
    points = np.argwhere(mask).astype(np.int64)
    if len(points) == 0:
        return mask.copy()
    keep = rng.random(len(points)) >= float(dropout_rate)
    result = np.zeros_like(mask, dtype=bool)
    kept = points[keep]
    if len(kept):
        result[kept[:, 0], kept[:, 1]] = True
    return result


def _add_noise(mask: np.ndarray, valid: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    result = mask.copy()
    available = np.argwhere(valid & ~result).astype(np.int64)
    if len(available) == 0 or count <= 0:
        return result
    chosen = available[rng.choice(len(available), size=min(int(count), len(available)), replace=False)]
    result[chosen[:, 0], chosen[:, 1]] = True
    return result


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    if union == 0:
        return 1.0
    return float(int((a & b).sum()) / union)


def _sample_negative_positions(
    maps: np.ndarray,
    labels: np.ndarray,
    single_indices_by_class: dict[int, np.ndarray],
    features: List[Dict],
    query_pos: int,
    query_class_idx: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, List[str]]:
    selected: List[int] = []
    sources: List[str] = []
    used = {int(query_pos)}
    same = _sample_from_pool(
        [int(pos) for pos in single_indices_by_class[query_class_idx] if int(pos) not in used],
        args.same_class_negatives,
        rng,
        used,
    )
    selected.extend(same)
    sources.extend(["same_class"] * len(same))

    random_pool = [
        int(pos)
        for class_idx, pool in single_indices_by_class.items()
        if class_idx != query_class_idx
        for pos in pool
        if int(pos) not in used
    ]
    random_negatives = _sample_balanced_other_classes(
        single_indices_by_class,
        exclude_class=query_class_idx,
        count=args.random_negatives,
        rng=rng,
        used=used,
    )
    selected.extend(random_negatives)
    sources.extend(["random"] * len(random_negatives))

    hard = _sample_hard_negatives(
        labels,
        single_indices_by_class,
        features,
        query_pos=query_pos,
        query_class_idx=query_class_idx,
        count=args.hard_negatives,
        rng=rng,
        used=used,
    )
    selected.extend(hard)
    sources.extend(["hard"] * len(hard))

    missing = args.negatives_per_trial - len(selected)
    if missing > 0:
        fill = _sample_from_pool(random_pool, missing, rng, used)
        selected.extend(fill)
        sources.extend(["fill"] * len(fill))
    return np.asarray(selected, dtype=np.int64), sources


def _sample_balanced_other_classes(
    single_indices_by_class: dict[int, np.ndarray],
    exclude_class: int,
    count: int,
    rng: np.random.Generator,
    used: set[int],
) -> List[int]:
    if count <= 0:
        return []
    other_classes = [idx for idx in range(len(CLASS_NAMES)) if idx != exclude_class]
    rng.shuffle(other_classes)
    base = count // len(other_classes)
    remainder = count % len(other_classes)
    selected: List[int] = []
    for order, class_idx in enumerate(other_classes):
        quota = base + int(order < remainder)
        pool = [int(pos) for pos in single_indices_by_class[class_idx] if int(pos) not in used]
        chosen = _sample_from_pool(pool, quota, rng, used)
        selected.extend(chosen)
    missing = count - len(selected)
    if missing > 0:
        pool = [
            int(pos)
            for class_idx in other_classes
            for pos in single_indices_by_class[class_idx]
            if int(pos) not in used
        ]
        selected.extend(_sample_from_pool(pool, missing, rng, used))
    return selected


def _sample_hard_negatives(
    labels: np.ndarray,
    single_indices_by_class: dict[int, np.ndarray],
    features: List[Dict],
    query_pos: int,
    query_class_idx: int,
    count: int,
    rng: np.random.Generator,
    used: set[int],
) -> List[int]:
    if count <= 0:
        return []
    qf = features[int(query_pos)]
    candidates = [
        int(pos)
        for class_idx, pool in single_indices_by_class.items()
        for pos in pool
        if class_idx != query_class_idx and int(pos) not in used and int(pos) != int(query_pos)
    ]
    scored = []
    q_count = max(float(qf["defect_count"]), 1.0)
    q_centroid = qf["centroid"]
    q_hist = qf["radial_hist"]
    for pos in candidates:
        cf = features[pos]
        c_count = max(float(cf["defect_count"]), 1.0)
        count_ratio = min(q_count, c_count) / max(q_count, c_count)
        centroid_dist = float(np.linalg.norm(q_centroid - cf["centroid"]))
        hist_sim = float(np.dot(q_hist, cf["radial_hist"]))
        hard_score = 0.45 * count_ratio + 0.35 * np.exp(-centroid_dist / 0.25) + 0.20 * hist_sim
        if count_ratio >= 0.50 and centroid_dist <= 0.35:
            hard_score += 0.20
        scored.append((hard_score, pos))
    scored.sort(reverse=True)
    top_pool = [pos for _, pos in scored[: max(count * 5, count)]]
    return _sample_from_pool(top_pool, count, rng, used)


def _sample_from_pool(pool: Sequence[int], count: int, rng: np.random.Generator, used: set[int]) -> List[int]:
    if count <= 0:
        return []
    pool = [int(pos) for pos in pool if int(pos) not in used]
    if not pool:
        return []
    replace = len(pool) < count
    chosen = rng.choice(np.asarray(pool, dtype=np.int64), size=count, replace=replace).astype(np.int64).tolist()
    result = []
    for pos in chosen:
        if int(pos) in used and not replace:
            continue
        result.append(int(pos))
        used.add(int(pos))
    return result


def _negative_records(
    maps: np.ndarray,
    labels: np.ndarray,
    original_ids: np.ndarray,
    positions: np.ndarray,
    sources: Sequence[str],
) -> List[Dict]:
    records = []
    for pos, source in zip(positions, sources):
        pos = int(pos)
        class_idx = int(np.flatnonzero(labels[pos].astype(np.int32) == 1)[0])
        class_name = CLASS_NAMES[class_idx]
        records.append({
            "candidate_id": f"map_{int(original_ids[pos])}",
            "source_map_id": int(original_ids[pos]),
            "class_name": class_name,
            "is_target": False,
            "negative_source": str(source),
            "transform_type": "",
            "transform_params": "",
            "grid": _grid_from_raw(maps[pos], role="negative", map_id=int(original_ids[pos]), class_name=class_name),
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
            "is_target": int(candidate["is_target"]),
            "negative_source": candidate.get("negative_source", ""),
            "transform_type": candidate.get("transform_type", ""),
            "transform_params": candidate.get("transform_params", ""),
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


def _trial_metrics(ranked: List[Dict], target_count: int) -> Dict[str, float]:
    target_ranks = [int(row["rank"]) for row in ranked if row["is_target"]]
    best_rank = min(target_ranks) if target_ranks else len(ranked) + 1
    result: Dict[str, float] = {
        "target_count": int(target_count),
        "best_target_rank": int(best_rank),
        "mean_target_rank": float(np.mean(target_ranks)) if target_ranks else 0.0,
        "median_target_rank": float(np.median(target_ranks)) if target_ranks else 0.0,
        "mrr_target": 1.0 / float(best_rank) if target_ranks else 0.0,
    }
    for k in TOP_KS:
        targets_at_k = sum(int(row["is_target"]) for row in ranked[:k])
        result[f"targets_at_{k}"] = int(targets_at_k)
        result[f"hit_at_{k}"] = int(targets_at_k > 0)
        result[f"recall_at_{k}"] = float(targets_at_k / max(int(target_count), 1))
        result[f"precision_at_{k}"] = float(targets_at_k / float(k))
    return result


def _target_rank_rows(trial_id: str, query_class: str, query_source_map_id: int, ranked: List[Dict]) -> List[Dict]:
    rows = []
    for row in ranked:
        if not row["is_target"]:
            continue
        rows.append({
            "trial_id": trial_id,
            "query_class": query_class,
            "query_source_map_id": query_source_map_id,
            "candidate_id": row["candidate_id"],
            "transform_type": row["transform_type"],
            "transform_params": row["transform_params"],
            "rank": int(row["rank"]),
            "score": float(row["score"]),
            "hit_at_5": int(int(row["rank"]) <= 5),
            "hit_at_10": int(int(row["rank"]) <= 10),
        })
    return rows


def _metrics(trial_rows: List[Dict], target_rows: List[Dict], args: argparse.Namespace) -> Dict:
    metric_keys = _metric_fieldnames()
    class_names = [CLASS_NAMES[idx] for idx in _selected_class_indices(args.class_name)]
    overall = _mean_metrics(trial_rows, metric_keys)
    per_class = _per_class_metrics(trial_rows, class_names, metric_keys)
    return {
        "dataset": "Mixed38K",
        "task": "transform_retrieval",
        "primary_metric": "average_score",
        "average_score": overall["average_score"],
        "proposal_mode": args.proposal_mode,
        "class_name": args.class_name,
        "classes": class_names,
        "positives_per_trial": int(args.positives_per_trial),
        "negatives_per_trial": int(args.negatives_per_trial),
        "gallery_size": int(args.positives_per_trial + args.negatives_per_trial),
        "negative_mix": {
            "random": int(args.random_negatives),
            "same_class": int(args.same_class_negatives),
            "hard": int(args.hard_negatives),
        },
        "trials_per_class": int(args.trials_per_class),
        "total_trials": len(trial_rows),
        "overall": overall,
        "per_class": per_class,
        "per_transform": _per_transform_metrics(target_rows),
    }


def _metric_fieldnames() -> List[str]:
    keys = [
        "best_target_rank",
        "mean_target_rank",
        "median_target_rank",
        "mrr_target",
    ]
    for k in TOP_KS:
        keys.extend([f"targets_at_{k}", f"hit_at_{k}", f"recall_at_{k}", f"precision_at_{k}"])
    return keys


def _mean_metrics(rows: List[Dict], keys: List[str]) -> Dict[str, float]:
    if not rows:
        return {key: 0.0 for key in keys}
    metrics = {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}
    metrics["top5_target_hit_rate"] = metrics.get("hit_at_5", 0.0)
    metrics["top5_target_recall"] = metrics.get("recall_at_5", 0.0)
    metrics["average_score"] = metrics["top5_target_hit_rate"]
    return metrics


def _per_class_metrics(trial_rows: List[Dict], class_names: Sequence[str], keys: List[str]) -> Dict[str, Dict]:
    result = {}
    for class_name in class_names:
        rows = [row for row in trial_rows if row["query_class"] == class_name]
        result[class_name] = {"trials": len(rows), **_mean_metrics(rows, keys)}
    return result


def _per_transform_metrics(target_rows: List[Dict]) -> Dict[str, Dict]:
    result = {}
    for transform_type in TRANSFORM_TYPES:
        rows = [row for row in target_rows if row["transform_type"] == transform_type]
        if not rows:
            result[transform_type] = {"targets": 0, "hit_at_5": 0.0, "hit_at_10": 0.0, "mean_rank": 0.0}
            continue
        result[transform_type] = {
            "targets": len(rows),
            "hit_at_5": float(np.mean([int(row["hit_at_5"]) for row in rows])),
            "hit_at_10": float(np.mean([int(row["hit_at_10"]) for row in rows])),
            "mean_rank": float(np.mean([int(row["rank"]) for row in rows])),
        }
    return result


def _class_counts(labels: np.ndarray, positions: np.ndarray) -> Dict[str, int]:
    counts = {class_name: 0 for class_name in CLASS_NAMES}
    for pos in positions:
        label_idx = int(np.flatnonzero(labels[int(pos)].astype(np.int32) == 1)[0])
        counts[CLASS_NAMES[label_idx]] += 1
    return {class_name: count for class_name, count in counts.items() if count > 0}


def _counts(values: Sequence[str]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for value in values:
        result[str(value)] = result.get(str(value), 0) + 1
    return result


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


def _wafer_center_and_radius(valid_mask: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = valid_mask.shape
    center = np.array([h / 2.0, w / 2.0], dtype=np.float32)
    valid_points = np.argwhere(valid_mask).astype(np.float32)
    radius_ref = float(np.linalg.norm(valid_points - center, axis=1).max()) if len(valid_points) else float(np.linalg.norm(center))
    return center, radius_ref


def _save_trial_figure(path: Path, query: GridMaps, top10: List[Dict], title: str) -> None:
    _ensure_mpl()
    import matplotlib.pyplot as plt

    panels = [("query", query, False, 0.0, "")] + [
        (
            f"#{row['rank']} {row['candidate_class']}",
            row["_grid"],
            bool(row["is_target"]),
            float(row["score"]),
            str(row.get("transform_type", "")),
        )
        for row in top10
    ]
    cols = len(panels)
    fig, axes = plt.subplots(1, cols, figsize=(2.25 * cols, 2.95))
    for ax, (label, grid, is_target, score, transform_type) in zip(np.asarray(axes).reshape(-1), panels):
        _draw_grid(ax, grid, label, is_target=is_target, score=score, transform_type=transform_type)
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.04, top=0.78, wspace=0.08)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_grid(ax, grid: GridMaps, label: str, is_target: bool, score: float, transform_type: str) -> None:
    image = np.zeros((*grid.status_map.shape, 3), dtype=np.float32)
    valid = grid.status_map == VALID_NO_DEFECT
    defects = grid.status_map == VALID_HAS_DEFECT
    image[valid] = (0.58, 0.58, 0.58)
    image[defects] = (0.92, 0.92, 0.92)
    if is_target:
        image[defects] = (0.95, 0.12, 0.10)
    ax.imshow(image, interpolation="nearest")
    score_text = "" if label == "query" else f"\nscore={score:.3f}"
    target_text = f"\nTARGET {transform_type}" if is_target else ""
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
        "best_target_rank",
        "rank",
        "candidate_id",
        "candidate_source_map_id",
        "candidate_class",
        "is_target",
        "negative_source",
        "transform_type",
        "transform_params",
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
        "positive_count",
        "negative_count",
        "negative_source_counts",
        "negative_class_counts",
        "target_count",
        "best_target_rank",
        "mean_target_rank",
        "median_target_rank",
        "mrr_target",
    ] + [field for k in TOP_KS for field in (f"targets_at_{k}", f"hit_at_{k}", f"recall_at_{k}", f"precision_at_{k}")]


def _target_fieldnames() -> List[str]:
    return [
        "trial_id",
        "query_class",
        "query_source_map_id",
        "candidate_id",
        "transform_type",
        "transform_params",
        "rank",
        "score",
        "hit_at_5",
        "hit_at_10",
    ]


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
