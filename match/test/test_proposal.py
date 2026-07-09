# -*- coding: utf-8 -*-
"""WM38K smoke test for match-local proposal, descriptor, and final matching.

The script samples one WM38K map as the reference and another N maps as
candidate maps. It then builds match.core.models.GridMaps objects and calls
match.core.local_matching.explain_count_partial_match, so proposal generation,
descriptor construction, and token matching all come from the match package.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from match.core.local_matching import explain_count_partial_match
from match.core.models import BACKGROUND, GridMaps, VALID_HAS_DEFECT, VALID_NO_DEFECT


CLASS_NAMES = [
    "center",
    "donut",
    "edge-loc",
    "edge-ring",
    "loc",
    "random",
    "scratch",
    "near-full",
]


def main() -> None:
    args = parse_args()
    run_test(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random WM38K test for match-local proposal, descriptor, and final matching."
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=str(ROOT.parent / "data/wm38k/Wafer_Map_Datasets.npz"),
        help="Path to WM38K Wafer_Map_Datasets.npz.",
    )
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "artifacts/match_test_proposal"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument(
        "--input-size",
        type=int,
        nargs="+",
        default=None,
        metavar=("H", "W"),
        help="Optional resize target. Use one value for square SxS or two values for H W.",
    )
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--top-k-proposals", type=int, default=6)
    parser.add_argument("--token-match-top-k", type=int, default=3)
    parser.add_argument("--map-match-top-k", type=int, default=20)
    parser.add_argument("--sigma-pos", type=float, default=0.35)
    parser.add_argument("--sigma-scale", type=float, default=1.5)
    parser.add_argument("--min-token-score", type=float, default=0.10)
    parser.add_argument("--proposal-mode", type=str, default="compact", choices=["cc", "compact"])
    parser.add_argument("--rotation-tolerance", action="store_true")
    parser.add_argument("--save-figures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--review-top-k", type=int, default=3)
    parser.add_argument("--step-max", type=int, default=3)
    parser.add_argument("--save-match-details", action="store_true")
    parser.add_argument("--match-detail-top-k", type=int, default=10)
    return parser.parse_args()


def run_test(args: argparse.Namespace) -> Dict[str, Path]:
    maps, labels, original_ids = _load_valid_wm38k(args.data_file)
    target_shape = _parse_input_size(args.input_size)
    sample_positions = _sample_positions(len(maps), args.num_candidates + 1, args.seed)
    reference_pos = int(sample_positions[0])
    candidate_positions = [int(pos) for pos in sample_positions[1:]]

    reference_raw = _maybe_resize(maps[reference_pos], target_shape)
    reference = _grid_from_wm38k(reference_raw, "reference", int(original_ids[reference_pos]), labels[reference_pos])
    candidate_records = []
    for pos in candidate_positions:
        raw = _maybe_resize(maps[pos], target_shape)
        candidate_records.append({
            "position": int(pos),
            "map_id": int(original_ids[pos]),
            "label": labels[pos].astype(np.int32),
            "grid": _grid_from_wm38k(raw, "candidate", int(original_ids[pos]), labels[pos]),
        })

    ranking_rows = []
    ranking_rows_mo = []
    token_rows = []
    detail_rows = []
    reference_token_rows_written = False
    for candidate in candidate_records:
        explanation = explain_count_partial_match(
            reference,
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
        )
        result = explanation["result"]
        result_mo = explanation["result_matched_only"]
        if not reference_token_rows_written:
            token_rows.extend(_token_rows("reference", int(original_ids[reference_pos]), labels[reference_pos], explanation["wbm_tokens"]))
            reference_token_rows_written = True
        token_rows.extend(_token_rows("candidate", candidate["map_id"], candidate["label"], explanation["wdm_tokens"]))
        ranking_rows.append({
            "query_id": int(original_ids[reference_pos]),
            "candidate_id": candidate["map_id"],
            "similarity_score": float(result.score),
            "mean_shape": float(result.mean_shape),
            "mean_position": float(result.mean_position),
            "mean_scale": float(result.mean_scale),
            "matched_tokens": int(result.matched_tokens),
            "reference_token_count": int(result.wbm_tokens),
            "candidate_token_count": int(result.wdm_tokens),
            "candidate_signature": _signature_text(candidate["label"]),
            "candidate_defect_count": int((candidate["grid"].count_map > 0).sum()),
        })
        ranking_rows_mo.append({
            "query_id": int(original_ids[reference_pos]),
            "candidate_id": candidate["map_id"],
            "similarity_score": float(result_mo.score),
            "mean_shape": float(result_mo.mean_shape),
            "mean_position": float(result_mo.mean_position),
            "mean_scale": float(result_mo.mean_scale),
            "matched_tokens": int(result_mo.matched_tokens),
            "reference_token_count": int(result_mo.wbm_tokens),
            "candidate_token_count": int(result_mo.wdm_tokens),
            "candidate_signature": _signature_text(candidate["label"]),
            "candidate_defect_count": int((candidate["grid"].count_map > 0).sum()),
        })
        if args.save_match_details:
            detail_rows.extend(_match_detail_rows(int(original_ids[reference_pos]), candidate["map_id"], result.score, explanation["matches"]))

    ranking_rows.sort(key=lambda row: row["similarity_score"], reverse=True)
    for rank, row in enumerate(ranking_rows, start=1):
        row["rank"] = rank

    ranking_rows_mo.sort(key=lambda row: row["similarity_score"], reverse=True)
    for rank, row in enumerate(ranking_rows_mo, start=1):
        row["rank"] = rank

    out_dir = _output_dir(args.out_dir, reference.count_map.shape, target_shape)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "rankings.csv", ranking_rows, _ranking_fieldnames())
    _write_csv(out_dir / "tokens.csv", token_rows, _token_fieldnames())
    _write_selection_json(out_dir / "selection.json", args, target_shape, reference, reference_pos, original_ids, labels, candidate_records)
    if args.save_match_details:
        top_ids = {int(row["candidate_id"]) for row in ranking_rows[:max(args.match_detail_top_k, 0)]}
        _write_csv(out_dir / "match_details.csv", [row for row in detail_rows if int(row["candidate_id"]) in top_ids], _detail_fieldnames())
    if args.save_figures:
        _save_review_figures(out_dir, args, reference, ranking_rows, candidate_records)
        _save_review_figures(out_dir, args, reference, ranking_rows_mo, candidate_records, review_subdir="count_partial_review_matched_only", result_key="result_matched_only")

    # --- matched-only rankings CSV ---
    _write_csv(out_dir / "rankings_matched_only.csv", ranking_rows_mo, _ranking_fieldnames())

    print(f"Reference map_id={int(original_ids[reference_pos])} signature={_signature_text(labels[reference_pos])}")
    print(f"Scored {len(candidate_records)} candidates at shape {reference.count_map.shape[0]}x{reference.count_map.shape[1]}")
    print(f"Saved rankings to {out_dir / 'rankings.csv'}")
    print(f"Saved matched-only rankings to {out_dir / 'rankings_matched_only.csv'}")
    return {"out_dir": out_dir, "rankings_path": out_dir / "rankings.csv", "tokens_path": out_dir / "tokens.csv"}


def _load_valid_wm38k(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    maps = _pick_array(data, ("maps", "x", "X", "images", "arr_0"))
    labels = _pick_array(data, ("labels", "y", "Y", "targets", "arr_1")).astype(np.int32)
    valid_mask = labels.sum(axis=1) > 0
    return maps[valid_mask], labels[valid_mask], np.where(valid_mask)[0].astype(np.int64)


def _pick_array(npz, names: Sequence[str]) -> np.ndarray:
    for name in names:
        if name in npz.files:
            return npz[name]
    raise KeyError(f"None of {names} found in npz file. Available keys: {npz.files}")


def _grid_from_wm38k(raw: np.ndarray, role: str, map_id: int, label: np.ndarray) -> GridMaps:
    status_map = np.where(raw == VALID_HAS_DEFECT, VALID_HAS_DEFECT, np.where(raw == VALID_NO_DEFECT, VALID_NO_DEFECT, BACKGROUND)).astype(np.uint8)
    count_map = (status_map == VALID_HAS_DEFECT).astype(np.float32)
    binary_map = count_map.astype(np.uint8)
    valid_mass = float(count_map.sum())
    density_map = count_map / valid_mass if valid_mass > 0 else count_map.copy()
    return GridMaps(
        count_map=count_map,
        binary_map=binary_map,
        density_map=density_map,
        status_map=status_map,
        representation_map=density_map,
        representation_maps={"count": count_map, "binary": binary_map, "density": density_map},
        metadata={"source": "wm38k", "role": role, "map_id": int(map_id), "signature": _signature_text(label)},
    )


def _sample_positions(total: int, needed: int, seed: int) -> np.ndarray:
    if needed > total:
        raise ValueError(f"Need {needed} valid WM38K samples, but only {total} are available.")
    rng = np.random.default_rng(seed)
    return rng.choice(total, size=needed, replace=False)


def _parse_input_size(values: Sequence[int] | None) -> Tuple[int, int] | None:
    if values is None:
        return None
    if len(values) == 1:
        h = w = int(values[0])
    elif len(values) == 2:
        h, w = int(values[0]), int(values[1])
    else:
        raise ValueError("--input-size expects one value S or two values H W.")
    if h <= 0 or w <= 0:
        raise ValueError("--input-size values must be positive.")
    return h, w


def _maybe_resize(raw: np.ndarray, target_shape: Tuple[int, int] | None) -> np.ndarray:
    return raw if target_shape is None else _resize_nearest(raw, target_shape)


def _resize_nearest(raw: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    h, w = raw.shape
    target_h, target_w = target_shape
    if (h, w) == (target_h, target_w):
        return raw
    rr = np.floor(np.arange(target_h) * h / target_h).astype(np.int64)
    cc = np.floor(np.arange(target_w) * w / target_w).astype(np.int64)
    return raw[np.clip(rr, 0, h - 1)][:, np.clip(cc, 0, w - 1)]


def _signature_text(label: np.ndarray) -> str:
    indices = np.where(np.asarray(label).astype(np.int32) == 1)[0].tolist()
    return "|".join(CLASS_NAMES[i] for i in indices)


def _output_dir(base: str, actual_shape: Tuple[int, int], target_shape: Tuple[int, int] | None) -> Path:
    suffix = "original" if target_shape is None else f"{actual_shape[0]}x{actual_shape[1]}"
    return Path(base) / suffix


def _token_rows(role: str, map_id: int, label: np.ndarray, tokens: List[Dict]) -> List[Dict]:
    rows = []
    for token_id, token in enumerate(tokens):
        descriptor = np.asarray(token.get("descriptor", []), dtype=np.float32)
        rows.append({
            "role": role,
            "map_id": int(map_id),
            "token_id": int(token_id),
            "signature": _signature_text(label),
            "geometry_type": token.get("geometry_type", ""),
            "area": float(token.get("area", 0.0)),
            "support_area_ratio": float(token.get("support_area_ratio", 0.0)),
            "mass": float(token.get("mass", 0.0)),
            "mass_ratio": float(token.get("mass_ratio", 0.0)),
            "centroid_row": float(token.get("centroid_row", 0.0)),
            "centroid_col": float(token.get("centroid_col", 0.0)),
            "bbox_row_min": int(token.get("bbox_row_min", 0)),
            "bbox_col_min": int(token.get("bbox_col_min", 0)),
            "bbox_row_max": int(token.get("bbox_row_max", 0)),
            "bbox_col_max": int(token.get("bbox_col_max", 0)),
            "bbox_height": int(token.get("bbox_height", 0)),
            "bbox_width": int(token.get("bbox_width", 0)),
            "proposal_mode": token.get("proposal_config", {}).get("proposal_mode", ""),
            "descriptor_mode": token.get("proposal_config", {}).get("descriptor_mode", ""),
            "descriptor_dim": int(descriptor.size),
            "descriptor_norm": float(np.linalg.norm(descriptor)) if descriptor.size else 0.0,
        })
    return rows


def _match_detail_rows(query_id: int, candidate_id: int, map_score: float, matches: List[Dict]) -> List[Dict]:
    rows = []
    for match in matches:
        rows.append({
            "query_id": int(query_id),
            "candidate_id": int(candidate_id),
            "map_similarity_score": float(map_score),
            "match_rank": int(match.get("rank", 0)),
            "query_token_id": int(match.get("query_token_id", 0)),
            "candidate_token_id": int(match.get("candidate_token_id", 0)),
            "query_type": match.get("query_token", {}).get("geometry_type", ""),
            "candidate_type": match.get("candidate_token", {}).get("geometry_type", ""),
            "query_area": float(match.get("query_token", {}).get("area", 0.0)),
            "candidate_area": float(match.get("candidate_token", {}).get("area", 0.0)),
            "score": float(match.get("score", 0.0)),
            "shape_sim": float(match.get("shape_sim", 0.0)),
            "moment_sim": float(match.get("moment_sim", 0.0)),
            "geometry_sim": float(match.get("geometry_sim", 0.0)),
            "position_affinity": float(match.get("position_affinity", 0.0)),
            "scale_affinity": float(match.get("scale_affinity", 0.0)),
            "type_affinity": float(match.get("type_affinity", 0.0)),
        })
    return rows


def _save_review_figures(
    out_dir: Path,
    args: argparse.Namespace,
    reference: GridMaps,
    ranking_rows: List[Dict],
    candidates: List[Dict],
    review_subdir: str = "count_partial_review",
    result_key: str = "result",
) -> None:
    _ensure_mpl()
    import matplotlib.pyplot as plt

    from match.viz.count_partial_visualization import plot_count_partial_steps, plot_count_partial_topk

    candidate_by_id = {int(candidate["map_id"]): candidate for candidate in candidates}
    top_rows = ranking_rows[:max(int(args.review_top_k), 1)]
    top_candidates = [
        (f"rank{int(row['rank']):02d}_map{int(row['candidate_id'])}", candidate_by_id[int(row["candidate_id"])]["grid"])
        for row in top_rows
        if int(row["candidate_id"]) in candidate_by_id
    ]
    review_dir = out_dir / review_subdir
    steps_dir = review_dir / "proposal_steps"
    review_dir.mkdir(parents=True, exist_ok=True)
    steps_dir.mkdir(parents=True, exist_ok=True)

    if top_candidates:
        topk_path = review_dir / f"top{len(top_candidates)}_count_partial.png"
        plot_count_partial_topk(
            reference,
            top_candidates,
            title="WM38K count-map partial matching top candidates",
            min_area=args.min_area,
            top_k=args.top_k_proposals,
            proposal_mode=args.proposal_mode,
            rotation_tolerance=args.rotation_tolerance,
            min_token_score=args.min_token_score,
            save_path=topk_path,
            result_key=result_key,
        )
        plt.close("all")
        print(f"Count-partial TopK figure saved: {topk_path}")

    for row in ranking_rows[:max(int(args.step_max), 0)]:
        candidate_id = int(row["candidate_id"])
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            continue
        rank = int(row["rank"])
        stem = f"rank{rank:02d}_map{candidate_id}"
        step_path = steps_dir / f"{stem}_steps.png"
        plot_count_partial_steps(
            reference,
            candidate["grid"],
            title=f"Rank {rank}: map {candidate_id} ({_signature_text(candidate['label'])})",
            min_area=args.min_area,
            top_k=args.top_k_proposals,
            proposal_mode=args.proposal_mode,
            rotation_tolerance=args.rotation_tolerance,
            min_token_score=args.min_token_score,
            save_path=step_path,
            result_key=result_key,
        )
        plt.close("all")
        print(f"Count-partial step figure saved: {step_path}")


def _ensure_mpl() -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")


def _write_selection_json(
    path: Path,
    args: argparse.Namespace,
    target_shape: Tuple[int, int] | None,
    reference: GridMaps,
    reference_pos: int,
    original_ids: np.ndarray,
    labels: np.ndarray,
    candidates: List[Dict],
) -> None:
    payload = {
        "data_file": str(args.data_file),
        "seed": int(args.seed),
        "actual_shape": list(reference.count_map.shape),
        "target_shape": None if target_shape is None else list(target_shape),
        "proposal_mode": args.proposal_mode,
        "min_area": int(args.min_area),
        "top_k_proposals": int(args.top_k_proposals),
        "token_match_top_k": int(args.token_match_top_k),
        "map_match_top_k": int(args.map_match_top_k),
        "min_token_score": float(args.min_token_score),
        "reference": {
            "map_id": int(original_ids[reference_pos]),
            "valid_position": int(reference_pos),
            "signature": _signature_text(labels[reference_pos]),
            "defect_count": int(reference.count_map.sum()),
        },
        "candidates": [
            {
                "map_id": int(candidate["map_id"]),
                "valid_position": int(candidate["position"]),
                "signature": _signature_text(candidate["label"]),
                "defect_count": int(candidate["grid"].count_map.sum()),
            }
            for candidate in candidates
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _ranking_fieldnames() -> List[str]:
    return [
        "query_id",
        "rank",
        "candidate_id",
        "similarity_score",
        "mean_shape",
        "mean_position",
        "mean_scale",
        "matched_tokens",
        "reference_token_count",
        "candidate_token_count",
        "candidate_signature",
        "candidate_defect_count",
    ]


def _token_fieldnames() -> List[str]:
    return [
        "role",
        "map_id",
        "token_id",
        "signature",
        "geometry_type",
        "area",
        "support_area_ratio",
        "mass",
        "mass_ratio",
        "centroid_row",
        "centroid_col",
        "bbox_row_min",
        "bbox_col_min",
        "bbox_row_max",
        "bbox_col_max",
        "bbox_height",
        "bbox_width",
        "proposal_mode",
        "descriptor_mode",
        "descriptor_dim",
        "descriptor_norm",
    ]


def _detail_fieldnames() -> List[str]:
    return [
        "query_id",
        "candidate_id",
        "map_similarity_score",
        "match_rank",
        "query_token_id",
        "candidate_token_id",
        "query_type",
        "candidate_type",
        "query_area",
        "candidate_area",
        "score",
        "shape_sim",
        "moment_sim",
        "geometry_sim",
        "position_affinity",
        "scale_affinity",
        "type_affinity",
    ]


if __name__ == "__main__":
    main()
