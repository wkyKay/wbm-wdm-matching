from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import List, Tuple

import numpy as np

try:
    from .mappers import MAPPERS
    from .representations import REPRESENTATIONS
    from .pipeline import map_klarf_to_grid
    from .fileio import load_defect_tables, read_wbm_shape, read_wbm_png, save_grid_maps
    from .similarity import SIMILARITIES, compute_similarity, SimilarityResult
except ImportError:
    from mappers import MAPPERS
    from representations import REPRESENTATIONS
    from pipeline import map_klarf_to_grid
    from fileio import load_defect_tables, read_wbm_shape, read_wbm_png, save_grid_maps
    from similarity import SIMILARITIES, compute_similarity, SimilarityResult


# ── 表头中 similarity 方法的顺序 ──────────────────────────────────
SIMILARITY_COLUMNS: List[str] = [
    "dice", "iou", "ncc", "cosine",
    "coverage", "leakage", "coverage-leakage", "chamfer",
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
        from .models import GridMaps as GM
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

    # 附加 metadata
    result["_mapped"] = grid_maps.metadata.get("mapped_defects", 0)
    result["_input"] = grid_maps.metadata.get("input_defects", 0)
    result["_status"] = "OK"
    return result


def format_score(val) -> str:
    """将 similarity 结果格式化为 6 位小数字符串。"""
    if isinstance(val, SimilarityResult):
        return f"{val.score:.6f}"
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
    header = ["file"] + SIMILARITY_COLUMNS + ["mapped_defects"]
    with open(log_path, "w") as f:
        f.write("\t".join(header) + "\n")
        for fname, res in rows:
            scores = []
            for col in SIMILARITY_COLUMNS:
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
        for col in SIMILARITY_COLUMNS:
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

    # ── 打印摘要表格到 stdout ──
    col_widths = [max(len(h), 10) for h in header]
    for fname, res in rows:
        col_widths[0] = max(col_widths[0], len(fname))
        for j, col in enumerate(SIMILARITY_COLUMNS):
            col_widths[j + 1] = max(col_widths[j + 1], len(format_score(res.get(col, ""))))

    def fmt_row(cells: list) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, col_widths))

    print("\n" + fmt_row(header))
    print("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for fname, res in rows:
        cells = [fname]
        for col in SIMILARITY_COLUMNS:
            cells.append(format_score(res.get(col, "")))
        cells.append(f"{res.get('_mapped', '')}/{res.get('_input', '')}")
        print(fmt_row(cells))


if __name__ == "__main__":
    main()
