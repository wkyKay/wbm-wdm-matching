from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import List, Tuple

import numpy as np

from ..core.mappers import MAPPERS
from ..core.representations import REPRESENTATIONS
from ..core.pipeline import map_klarf_to_grid
from ..data.fileio import load_defect_tables, read_wbm_shape, read_wbm_png, save_grid_maps
from ..core.similarity import SIMILARITIES, compute_similarity, SimilarityResult
from ..core.local_matching import LocalMatchResult, compute_count_partial_match
from ..core.classnumber_matching import classnumber_scores_dict, compute_classnumber_matches
from .batch_io import format_score, write_result_log, write_topk_log
from .batch_viz import save_classnumber_figures, save_count_partial_figures


# ── 表头中 similarity 方法的顺序 ──────────────────────────────────
SIMILARITY_COLUMNS: List[str] = [
    "dice", "iou", "ncc", "cosine",
    "coverage", "leakage", "coverage-leakage", "chamfer",
]

PARTIAL_MATCH_COLUMNS: List[str] = [
    "count-partial",
    "count-partial-shape",
    "count-partial-position",
    "count-partial-scale",
    "count-partial-type",
    "count-partial-tokens",
]

CLASSNUMBER_COLUMNS: List[str] = [
    "classnumber-count",
    "best-classnumber",
    "best-classnumber-partial",
    "best-classnumber-tokens",
    "best-classnumber-binary",
    "best-classnumber-binary-coverage",
    "best-classnumber-binary-leakage",
    "best-classnumber-rank-mode",
    "best-classnumber-rank-score",
]

RESULT_COLUMNS: List[str] = SIMILARITY_COLUMNS + PARTIAL_MATCH_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch WBM-WDM matching: process a directory of KLARF files against a reference WBM."
    )
    _add_input_args(parser)
    _add_mapping_args(parser)
    _add_output_args(parser)
    _add_count_partial_args(parser)
    _add_classnumber_args(parser)
    return parser.parse_args()


def _add_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--klarf-dir",
        required=True,
        help="Directory containing KLARF files to process.",
    )
    parser.add_argument(
        "--klarf-glob",
        default="*.*",
        help="Glob pattern to match KLARF files (default: *.klarf).",
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Path to the reference WBM PNG (required).",
    )
    parser.add_argument(
        "--wbm",
        help="Path to the target WBM PNG for shape reference. Defaults to --reference.",
    )


def _add_mapping_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--height", type=int,
        help="Target WBM grid height. Overrides --wbm shape.",
    )
    parser.add_argument(
        "--width", type=int,
        help="Target WBM grid width. Overrides --wbm shape.",
    )
    parser.add_argument(
        "--mapper",
        choices=sorted(MAPPERS),
        default="die-index",
        help="Pluggable coordinate mapping strategy.",
    )
    parser.add_argument(
        "--representation",
        choices=sorted(REPRESENTATIONS),
        default="density",
        help="Pluggable grid map representation.",
    )
    parser.add_argument(
        "--die-x-range", nargs=2, type=int, metavar=("MIN", "MAX"),
        help="Fixed die index range along X (e.g. -20 20).",
    )
    parser.add_argument(
        "--die-y-range", nargs=2, type=int, metavar=("MIN", "MAX"),
        help="Fixed die index range along Y (e.g. -20 20).",
    )
    parser.add_argument(
        "--defect-table-index", type=int, default=0,
        help="Which DefectList to use when a KLARF contains multiple wafers.",
    )


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        default="match/output",
        help="Directory for per-file .npz outputs.",
    )
    parser.add_argument(
        "--log",
        default="match/output/batch_results.tsv",
        help="Path to the output TSV log file.",
    )
    parser.add_argument(
        "--defect-threshold",
        "--min-defects",
        dest="defect_threshold",
        type=int,
        default=5,
        help="Skip KLARF files with fewer than this many defects before any similarity or visualization work.",
    )
    parser.add_argument(
        "--topk", type=int, default=10,
        help="Number of top-K files to show per metric in the ranking log (0 = all).",
    )
    parser.add_argument(
        "--topk-log",
        default="match/output/batch_topk.tsv",
        help="Path to the top-K ranking log file.",
    )


def _add_count_partial_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--save-count-partial-figures",
        action="store_true",
        help="Save count-map partial matching TopK and proposal-step review figures.",
    )
    parser.add_argument(
        "--count-partial-fig-dir",
        default="match/output/count_partial_review",
        help="Directory for count-partial review figures when --save-count-partial-figures is set.",
    )
    parser.add_argument(
        "--count-partial-review-top-k",
        type=int,
        default=3,
        help="Number of candidates shown in the count-partial TopK figure.",
    )
    parser.add_argument(
        "--count-partial-step-max",
        type=int,
        default=3,
        help="Number of top count-partial candidates rendered as proposal-step figures.",
    )
    parser.add_argument(
        "--count-partial-min-area",
        type=int,
        default=5,
        help="Minimum support area for count-partial WBM/WDM tokens.",
    )
    parser.add_argument(
        "--count-partial-top-k-proposals",
        type=int,
        default=6,
        help="Maximum WBM/WDM proposal tokens retained for count-partial matching.",
    )


def _add_classnumber_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--use-classnumber",
        action="store_true",
        help="Split each KLARF by classnumber and run additional per-class WDM matching.",
    )
    parser.add_argument(
        "--save-classnumber-figures",
        action="store_true",
        help="Save WBM, full WDM, and classnumber-split WDM review figures. Requires --use-classnumber.",
    )
    parser.add_argument(
        "--classnumber-fig-dir",
        default="match/output/classnumber_review",
        help="Directory for classnumber split review figures.",
    )
    parser.add_argument(
        "--classnumber-match-mode",
        choices=("count", "binary", "both"),
        default="count",
        help="Scoring mode for classnumber split matching. 'count' keeps the existing count-partial behavior.",
    )
    parser.add_argument(
        "--classnumber-rank-by",
        choices=("count", "binary"),
        default="count",
        help="Score used to rank classnumber split outputs when --classnumber-match-mode=both.",
    )
    parser.add_argument(
        "--classnumber-binary-dilation",
        type=int,
        default=1,
        help="Pixel radius used to tolerate small offsets in classnumber binary matching.",
    )
    parser.add_argument(
        "--classnumber-binary-beta",
        type=float,
        default=0.5,
        help="Leakage penalty weight for classnumber binary score: coverage - beta * leakage.",
    )


def resolve_shape(args: argparse.Namespace) -> tuple[int, int]:
    if args.height is not None or args.width is not None:
        if args.height is None or args.width is None:
            raise ValueError("Pass both --height and --width, or pass neither and use --wbm.")
        return args.height, args.width

    wbm_path = args.wbm or args.reference
    if not wbm_path:
        raise ValueError("Pass --wbm, --reference, or both --height and --width.")
    return read_wbm_shape(wbm_path)


def load_reference(path: str | Path) -> "GridMaps":
    """Load reference WBM. Supports .png and .npz."""
    path = Path(path)
    if path.suffix.lower() == ".png":
        return read_wbm_png(path)
    elif path.suffix.lower() == ".npz":
        ref_data = dict(np.load(path, allow_pickle=True))
        from ..core.models import GridMaps as GM
        return GM(
            count_map=ref_data["count_map"],
            binary_map=ref_data["binary_map"],
            density_map=ref_data["density_map"],
            status_map=ref_data["status_map"],
            representation_map=ref_data["representation_map"],
            representation_maps={k: ref_data[k] for k in
                ["binary", "count", "density", "soft", "three-value", "mountain"]
                if k in ref_data},
            metadata=ref_data.get("metadata", {}),
        )
    else:
        raise ValueError(f"Unsupported reference format: {path.suffix}")


def process_one(
    klarf_path: Path,
    args: argparse.Namespace,
    shape: Tuple[int, int],
    ref_gm: "GridMaps",
) -> dict:
    """处理单个 KLARF 文件，返回 {method: score_or_result} 或 {"error": msg}。"""
    result: dict = {}

    skip_reason = _check_defect_threshold(klarf_path, args)
    if skip_reason is not None:
        return skip_reason

    grid_maps_or_error = _map_klarf_file(klarf_path, args, shape)
    if "_status" in grid_maps_or_error:
        return grid_maps_or_error
    grid_maps = grid_maps_or_error["_grid_maps"]

    result.update(_compute_similarity_scores(ref_gm, grid_maps))
    result.update(_compute_count_partial_scores(ref_gm, grid_maps, args))
    result.update(_compute_classnumber_scores(klarf_path, ref_gm, shape, args))
    result.update(_attach_metadata(grid_maps, args))
    result["_status"] = "OK"
    return result


def _check_defect_threshold(klarf_path: Path, args: argparse.Namespace) -> dict | None:
    if args.defect_threshold <= 0:
        return None
    try:
        tables = load_defect_tables(klarf_path)
        idx = args.defect_table_index
        if idx >= len(tables) or len(tables[idx].rows) < args.defect_threshold:
            actual = len(tables[idx].rows) if idx < len(tables) else 0
            return {
                "_status": "SKIPPED",
                "_reason": f"defect count {actual} < {args.defect_threshold}",
            }
    except Exception as e:
        return {"_status": "ERROR", "_reason": f"count check failed: {e}"}
    return None


def _map_klarf_file(klarf_path: Path, args: argparse.Namespace, shape: Tuple[int, int]) -> dict:
    try:
        grid_maps = map_klarf_to_grid(
            klarf_path,
            shape=shape,
            mapper_name=args.mapper,
            representation_name=args.representation,
            defect_table_index=args.defect_table_index,
            die_x_range=tuple(args.die_x_range) if args.die_x_range else None,
            die_y_range=tuple(args.die_y_range) if args.die_y_range else None,
        )
    except ValueError as e:
        msg = str(e)
        if "Single-die wafer" in msg or "not supported" in msg:
            return {"_status": "SKIPPED", "_reason": msg.split("\n")[0]}
        return {"_status": "ERROR", "_reason": msg}
    except Exception as e:
        return {"_status": "ERROR", "_reason": f"{type(e).__name__}: {e}"}

    if args.output_dir:
        out_path = Path(args.output_dir) / f"{klarf_path.stem}.npz"
        save_grid_maps(out_path, grid_maps)
    return {"_grid_maps": grid_maps}


def _compute_similarity_scores(ref_gm: "GridMaps", grid_maps: "GridMaps") -> dict:
    result: dict = {}
    for method in SIMILARITY_COLUMNS:
        try:
            result[method] = compute_similarity(
                ref_gm.representation_map,
                grid_maps.representation_map,
                method=method,
                reference_status=ref_gm.status_map,
                candidate_status=grid_maps.status_map,
            )
        except Exception as e:
            result[method] = f"ERR:{e}"
    return result


def _compute_count_partial_scores(ref_gm: "GridMaps", grid_maps: "GridMaps", args: argparse.Namespace) -> dict:
    try:
        partial = compute_count_partial_match(
            ref_gm,
            grid_maps,
            min_area=args.count_partial_min_area,
            top_k=args.count_partial_top_k_proposals,
        )
        return {
            "count-partial": partial.score,
            "count-partial-shape": partial.mean_shape,
            "count-partial-position": partial.mean_position,
            "count-partial-scale": partial.mean_scale,
            "count-partial-type": partial.mean_type,
            "count-partial-tokens": partial,
        }
    except Exception as e:
        return {col: f"ERR:{e}" for col in PARTIAL_MATCH_COLUMNS}


def _compute_classnumber_scores(
    klarf_path: Path,
    ref_gm: "GridMaps",
    shape: Tuple[int, int],
    args: argparse.Namespace,
) -> dict:
    if not args.use_classnumber:
        return {}
    try:
        class_result = compute_classnumber_matches(
            klarf_path,
            reference=ref_gm,
            shape=shape,
            mapper_name=args.mapper,
            representation_name=args.representation,
            defect_table_index=args.defect_table_index,
            die_x_range=tuple(args.die_x_range) if args.die_x_range else None,
            die_y_range=tuple(args.die_y_range) if args.die_y_range else None,
            min_area=args.count_partial_min_area,
            top_k=args.count_partial_top_k_proposals,
            match_mode=args.classnumber_match_mode,
            rank_by=args.classnumber_rank_by,
            binary_dilation=args.classnumber_binary_dilation,
            binary_beta=args.classnumber_binary_beta,
        )
        result = classnumber_scores_dict(class_result)
        if args.save_classnumber_figures:
            result["_classnumber_result"] = class_result
        return result
    except Exception as e:
        return {col: f"ERR:{e}" for col in CLASSNUMBER_COLUMNS}


def _attach_metadata(grid_maps: "GridMaps", args: argparse.Namespace) -> dict:
    result = {
        "_mapped": grid_maps.metadata.get("mapped_defects", 0),
        "_input": grid_maps.metadata.get("input_defects", 0),
    }
    if args.save_count_partial_figures or args.save_classnumber_figures:
        result["_grid_maps"] = grid_maps
    return result


def format_score(val) -> str:
    """将 similarity 结果格式化为 6 位小数字符串。"""
    if isinstance(val, SimilarityResult):
        return f"{val.score:.6f}"
    if isinstance(val, LocalMatchResult):
        return f"{val.matched_tokens}/{val.wbm_tokens}/{val.wdm_tokens}"
    if isinstance(val, (float, int)):
        return f"{val:.6f}"
    if isinstance(val, str):
        return val  # 错误信息
    return str(val)


def main() -> None:
    args = parse_args()

    klarf_files = _collect_klarf_files(args)
    ref_gm, shape = _load_reference_context(args)
    rows, summary = _run_batch(args, klarf_files, ref_gm, shape)
    write_result_log(args.log, rows, _result_columns(args), format_score)
    write_topk_log(args.topk_log, rows, _ranking_columns(args), args.topk)
    _save_requested_figures(args, ref_gm, rows)
    _print_summary_table(args, rows)
    print(
        f"\nDone in {summary['elapsed']:.1f}s: "
        f"{summary['ok']} OK, {summary['skipped']} skipped, {summary['error']} error"
    )


def _collect_klarf_files(args: argparse.Namespace) -> List[Path]:
    klarf_dir = Path(args.klarf_dir)
    if not klarf_dir.is_dir():
        print(f"ERROR: --klarf-dir is not a directory: {klarf_dir}", file=sys.stderr)
        sys.exit(1)

    klarf_files = sorted(klarf_dir.glob(args.klarf_glob))
    if not klarf_files:
        print(f"ERROR: No .klarf files found in {klarf_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(klarf_files)} KLARF file(s) in {klarf_dir}")
    return klarf_files


def _load_reference_context(args: argparse.Namespace) -> tuple["GridMaps", Tuple[int, int]]:
    print(f"Loading reference: {args.reference}")
    ref_gm = load_reference(args.reference)
    shape = resolve_shape(args)
    print(f"Grid shape: {shape[0]}×{shape[1]}")
    print(f"Mapper: {args.mapper}, Representation: {args.representation}")
    print()
    return ref_gm, shape


def _run_batch(
    args: argparse.Namespace,
    klarf_files: List[Path],
    ref_gm: "GridMaps",
    shape: Tuple[int, int],
) -> tuple[List[Tuple[str, dict]], dict]:
    rows: List[Tuple[str, dict]] = []
    ok_count = skipped_count = error_count = 0
    t_start = time.monotonic()

    for i, kf in enumerate(klarf_files, 1):
        status_line = f"[{i}/{len(klarf_files)}] {kf.name}"
        sys.stdout.write(f"  {status_line} ... ")
        sys.stdout.flush()

        res = process_one(kf, args, shape, ref_gm)
        status = res.pop("_status", "UNKNOWN")
        if not (
            status == "SKIPPED"
            and str(res.get("_reason", "")).startswith("defect count")
        ):
            rows.append((kf.name, res))

        if status == "OK":
            ok_count += 1
            print("OK")
        elif status == "SKIPPED":
            skipped_count += 1
            print(f"SKIPPED ({res.get('_reason', '')})")
        else:
            error_count += 1
            print(f"ERROR ({res.get('_reason', '')})")

    return rows, {
        "ok": ok_count,
        "skipped": skipped_count,
        "error": error_count,
        "elapsed": time.monotonic() - t_start,
    }


def _save_requested_figures(args: argparse.Namespace, ref_gm: "GridMaps", rows: List[Tuple[str, dict]]) -> None:
    if args.save_count_partial_figures:
        save_count_partial_figures(args, ref_gm, rows)
    if args.use_classnumber and args.save_classnumber_figures:
        save_classnumber_figures(args, ref_gm, rows)


def _print_summary_table(args: argparse.Namespace, rows: List[Tuple[str, dict]]) -> None:
    result_columns = _result_columns(args)
    header = ["file"] + result_columns + ["mapped_defects"]
    col_widths = [max(len(h), 10) for h in header]
    for fname, res in rows:
        col_widths[0] = max(col_widths[0], len(fname))
        for j, col in enumerate(result_columns):
            col_widths[j + 1] = max(col_widths[j + 1], len(format_score(res.get(col, ""))))

    def fmt_row(cells: list) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, col_widths))

    print("\n" + fmt_row(header))
    print("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for fname, res in rows:
        cells = [fname]
        cells.extend(format_score(res.get(col, "")) for col in result_columns)
        cells.append(f"{res.get('_mapped', '')}/{res.get('_input', '')}")
        print(fmt_row(cells))


def _result_columns(args: argparse.Namespace) -> List[str]:
    columns = list(RESULT_COLUMNS)
    if args.use_classnumber:
        columns.extend(CLASSNUMBER_COLUMNS)
    return columns


def _ranking_columns(args: argparse.Namespace) -> List[str]:
    columns = SIMILARITY_COLUMNS + ["count-partial"]
    if args.use_classnumber:
        columns.append("best-classnumber-rank-score")
    return columns


if __name__ == "__main__":
    main()
