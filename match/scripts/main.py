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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch WBM-WDM matching: process a directory of KLARF files against a reference WBM."
    )
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
        "--min-defects", type=int, default=0,
        help="Skip KLARF files with fewer than this many defects.",
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
    return parser.parse_args()


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

    # Threshold check: skip if too few defects
    if args.min_defects > 0:
        try:
            tables = load_defect_tables(klarf_path)
            idx = args.defect_table_index
            if idx >= len(tables) or len(tables[idx].rows) < args.min_defects:
                actual = len(tables[idx].rows) if idx < len(tables) else 0
                return {
                    "_status": "SKIPPED",
                    "_reason": f"defect count {actual} < {args.min_defects}",
                }
        except Exception as e:
            return {"_status": "ERROR", "_reason": f"count check failed: {e}"}

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

    # 可选：保存单文件 npz
    if args.output_dir:
        out_path = Path(args.output_dir) / f"{klarf_path.stem}.npz"
        save_grid_maps(out_path, grid_maps)

    # 计算全部 similarity
    for method in SIMILARITY_COLUMNS:
        try:
            r = compute_similarity(
                ref_gm.representation_map,
                grid_maps.representation_map,
                method=method,
                reference_status=ref_gm.status_map,
                candidate_status=grid_maps.status_map,
            )
        except Exception as e:
            result[method] = f"ERR:{e}"
            continue

        result[method] = r

    # Count-map partial matching: WBM failure tokens explained by WDM count-map evidence tokens.
    try:
        partial = compute_count_partial_match(
            ref_gm,
            grid_maps,
            min_area=args.count_partial_min_area,
            top_k=args.count_partial_top_k_proposals,
        )
        result["count-partial"] = partial.score
        result["count-partial-shape"] = partial.mean_shape
        result["count-partial-position"] = partial.mean_position
        result["count-partial-scale"] = partial.mean_scale
        result["count-partial-type"] = partial.mean_type
        result["count-partial-tokens"] = partial
    except Exception as e:
        for col in PARTIAL_MATCH_COLUMNS:
            result[col] = f"ERR:{e}"

    if args.use_classnumber:
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
            )
            result.update(classnumber_scores_dict(class_result))
            if args.save_classnumber_figures:
                result["_classnumber_result"] = class_result
        except Exception as e:
            for col in CLASSNUMBER_COLUMNS:
                result[col] = f"ERR:{e}"

    # 附加 metadata
    result["_mapped"] = grid_maps.metadata.get("mapped_defects", 0)
    result["_input"] = grid_maps.metadata.get("input_defects", 0)
    if args.save_count_partial_figures or args.save_classnumber_figures:
        result["_grid_maps"] = grid_maps
    result["_status"] = "OK"
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

    klarf_dir = Path(args.klarf_dir)
    if not klarf_dir.is_dir():
        print(f"ERROR: --klarf-dir is not a directory: {klarf_dir}", file=sys.stderr)
        sys.exit(1)

    klarf_files = sorted(klarf_dir.glob(args.klarf_glob))
    if not klarf_files:
        print(f"ERROR: No .klarf files found in {klarf_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(klarf_files)} KLARF file(s) in {klarf_dir}")

    # 加载参考 WBM
    print(f"Loading reference: {args.reference}")
    ref_gm = load_reference(args.reference)
    shape = resolve_shape(args)
    print(f"Grid shape: {shape[0]}×{shape[1]}")
    print(f"Mapper: {args.mapper}, Representation: {args.representation}")
    print()

    # 逐文件处理
    rows: List[Tuple[str, dict]] = []  # (filename, result_dict)
    ok_count = skipped_count = error_count = 0
    t_start = time.monotonic()

    for i, kf in enumerate(klarf_files, 1):
        status_line = f"[{i}/{len(klarf_files)}] {kf.name}"
        sys.stdout.write(f"  {status_line} ... ")
        sys.stdout.flush()

        res = process_one(kf, args, shape, ref_gm)
        status = res.pop("_status", "UNKNOWN")
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

    elapsed = time.monotonic() - t_start
    print(f"\nDone in {elapsed:.1f}s: {ok_count} OK, {skipped_count} skipped, {error_count} error")

    # ── 写 TSV log ──
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    extra_columns = CLASSNUMBER_COLUMNS if args.use_classnumber else []
    result_columns = SIMILARITY_COLUMNS + PARTIAL_MATCH_COLUMNS + extra_columns
    header = ["file"] + result_columns + ["mapped_defects"]
    with open(log_path, "w") as f:
        f.write("\t".join(header) + "\n")
        for fname, res in rows:
            scores = []
            for col in result_columns:
                scores.append(format_score(res.get(col, "")))
            mapped = f"{res.get('_mapped', '')}/{res.get('_input', '')}"
            scores.append(mapped)
            f.write(f"{fname}\t" + "\t".join(scores) + "\n")

    print(f"Log saved: {log_path}")

    # ── Top-K ranking per metric ──
    topk = args.topk
    topk_log_path = Path(args.topk_log)
    topk_log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(topk_log_path, "w") as f:
        f.write("metric\trank\tfile\tscore\n")
        ranking_columns = SIMILARITY_COLUMNS + ["count-partial"]
        if args.use_classnumber:
            ranking_columns.append("best-classnumber-partial")
        for col in ranking_columns:
            # (filename, score) pairs, only valid numeric scores
            scored: list[tuple[str, float]] = []
            for fname, res in rows:
                val = res.get(col)
                if isinstance(val, SimilarityResult):
                    scored.append((fname, val.score))
                elif isinstance(val, (float, int)):
                    scored.append((fname, float(val)))

            scored.sort(key=lambda x: x[1], reverse=True)
            limit = topk if topk > 0 else len(scored)
            for rank, (fname, score) in enumerate(scored[:limit], 1):
                f.write(f"{col}\t{rank}\t{fname}\t{score:.6f}\n")

    print(f"Top-K log saved: {topk_log_path}")

    if args.save_count_partial_figures:
        _save_count_partial_figures(args, ref_gm, rows)
    if args.use_classnumber and args.save_classnumber_figures:
        _save_classnumber_figures(args, ref_gm, rows)

    # ── 打印摘要表格到 stdout ──
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
        for col in result_columns:
            cells.append(format_score(res.get(col, "")))
        cells.append(f"{res.get('_mapped', '')}/{res.get('_input', '')}")
        print(fmt_row(cells))


def _save_count_partial_figures(args: argparse.Namespace, ref_gm: "GridMaps", rows: List[Tuple[str, dict]]) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..viz.count_partial_visualization import plot_count_partial_steps, plot_count_partial_topk

    scored: list[tuple[str, "GridMaps", float]] = []
    for fname, res in rows:
        score = res.get("count-partial")
        grid_maps = res.get("_grid_maps")
        if isinstance(score, (float, int)) and grid_maps is not None:
            scored.append((Path(fname).stem, grid_maps, float(score)))

    if not scored:
        print("Count-partial figures skipped: no valid count-partial GridMaps available")
        return

    scored.sort(key=lambda item: item[2], reverse=True)
    out_dir = Path(args.count_partial_fig_dir)
    steps_dir = out_dir / "proposal_steps"
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_dir.mkdir(parents=True, exist_ok=True)

    top_n = max(args.count_partial_review_top_k, 1)
    top_records = scored[:top_n]
    topk_path = out_dir / f"top{len(top_records)}_count_partial.png"
    plot_count_partial_topk(
        ref_gm,
        [(name, gm) for name, gm, _ in top_records],
        title="Count-map partial matching top candidates",
        min_area=args.count_partial_min_area,
        top_k=args.count_partial_top_k_proposals,
        save_path=topk_path,
    )
    plt.close("all")
    print(f"Count-partial TopK figure saved: {topk_path}")

    for rank, (name, gm, _) in enumerate(scored[: max(args.count_partial_step_max, 0)], start=1):
        step_path = steps_dir / f"rank{rank:02d}_{_safe_name(name)}_steps.png"
        plot_count_partial_steps(
            ref_gm,
            gm,
            title=f"Rank {rank}: {name}",
            min_area=args.count_partial_min_area,
            top_k=args.count_partial_top_k_proposals,
            save_path=step_path,
        )
        plt.close("all")
        print(f"Count-partial step figure saved: {step_path}")


def _save_classnumber_figures(args: argparse.Namespace, ref_gm: "GridMaps", rows: List[Tuple[str, dict]]) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..viz.classnumber_visualization import plot_classnumber_splits

    out_dir = Path(args.classnumber_fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for fname, res in rows:
        grid_maps = res.get("_grid_maps")
        class_result = res.get("_classnumber_result")
        if grid_maps is None or class_result is None or not class_result.splits:
            continue
        path = out_dir / f"{_safe_name(Path(fname).stem)}_classnumber_splits.png"
        plot_classnumber_splits(
            ref_gm,
            grid_maps,
            class_result,
            title=f"{fname} classnumber split matching",
            save_path=path,
        )
        plt.close("all")
        print(f"Classnumber split figure saved: {path}")
        saved += 1

    if saved == 0:
        print("Classnumber figures skipped: no valid classnumber split results available")


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)[:80]


if __name__ == "__main__":
    main()
