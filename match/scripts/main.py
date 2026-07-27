from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cli_args import parse_args, SIMILARITY_COLUMNS, PARTIAL_MATCH_COLUMNS, PARTIAL_MATCH_MO_COLUMNS, RESULT_COLUMNS, CLASSNUMBER_COLUMNS
from .reference_loader import load_reference, resolve_shape
from .processing import process_one
from .batch_io import (
    write_map_match_log,
    write_result_log,
    write_token_match_log,
    write_topk_log,
    format_score,
)
from .batch_viz import save_baseline_figures, save_classnumber_baseline_figures, save_classnumber_figures, save_classnumber_summary_figure, save_count_partial_figures, save_summary_figure, save_wdm_raw_figures, save_classnumber_wdm_raw_figures


def main() -> None:
    """CLI 入口，解析命令行参数后调用 run()。"""
    run(parse_args())


def run(args: argparse.Namespace) -> List[Tuple[str, dict]]:
    """程序化入口，传入 args namespace 执行完整匹配流程，返回 rows。

    调用示例::

        from match.scripts.main import make_args, run

        rows = run(make_args(
            klarf_dir="/data/klarf/",
            reference="/data/ref.png",
            mode="count-partial",
            identifier="exp1",
        ))
    """
    klarf_files = _collect_klarf_files(args)
    ref_gm, shape = _load_reference_context(args)
    log_dir = _resolve_log_dir(args)
    log_dir.mkdir(parents=True, exist_ok=True)
    _save_args_json(log_dir.parent, args)
    rows, summary = _run_batch(args, klarf_files, ref_gm, shape, log_dir)

    write_result_log(log_dir / "results.tsv", rows, _result_columns(args), format_score)
    write_topk_log(log_dir / "topk.tsv", rows, _ranking_columns(args), args.topk)
    write_token_match_log(log_dir / "token_match.tsv", rows)
    write_map_match_log(log_dir / "map_match.tsv", rows)
    _save_requested_figures(args, ref_gm, rows, log_dir)
    print(
        f"\nDone in {summary['elapsed']:.1f}s: "
        f"{summary['ok']} OK, {summary['skipped']} skipped, {summary['error']} error"
    )
    return rows


def _resolve_log_dir(args: argparse.Namespace) -> Path:
    base = Path(args.output_dir)
    run_id = _safe_identifier(str(getattr(args, "identifier", "")).strip())
    mode = _safe_identifier(getattr(args, "mode", "count-partial"))
    if run_id:
        return base / run_id / mode
    return base / mode


def _collect_klarf_files(args) -> List[Path]:
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


def _load_reference_context(args) -> tuple["GridMaps", Tuple[int, int]]:
    print(f"Loading reference: {args.reference}")
    ref_gm = load_reference(args.reference)
    shape = resolve_shape(args)
    print(f"Grid shape: {shape[0]}×{shape[1]}")
    print(f"Mode: {args.mode}, Mapper: {args.mapper}, Representation: {args.representation}")
    print()
    return ref_gm, shape


def _run_batch(
    args,
    klarf_files: List[Path],
    ref_gm: "GridMaps",
    shape: Tuple[int, int],
    npz_dir: Path,
) -> tuple[List[Tuple[str, dict]], dict]:
    rows: List[Tuple[str, dict]] = []
    ok_count = skipped_count = error_count = 0
    t_start = time.monotonic()

    for i, kf in enumerate(klarf_files, 1):
        status_line = f"[{i}/{len(klarf_files)}] {kf.name}"
        sys.stdout.write(f"  {status_line} ... ")
        sys.stdout.flush()

        res = process_one(kf, args, shape, ref_gm, npz_dir)
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


def _save_requested_figures(
    args, ref_gm: "GridMaps", rows: List[Tuple[str, dict]], log_dir: Path
) -> None:
    if args.mode == "count-partial":
        save_baseline_figures(args, ref_gm, rows, log_dir)
        save_count_partial_figures(args, ref_gm, rows, log_dir)
        _safe_viz(save_wdm_raw_figures, args, rows, log_dir, label="WDM raw")
        _safe_viz(save_summary_figure, args, rows, log_dir, label="summary")
    elif args.mode == "classnumber":
        save_classnumber_baseline_figures(args, ref_gm, rows, log_dir)
        save_classnumber_figures(args, ref_gm, rows, log_dir)
        _safe_viz(save_classnumber_wdm_raw_figures, args, rows, log_dir, label="classnumber WDM raw")
        _safe_viz(save_classnumber_summary_figure, args, rows, log_dir, label="classnumber summary")


def _safe_viz(fn, *fn_args, label: str, **fn_kwargs) -> None:
    """安全调用可视化函数，失败时打印错误但不中断流程。"""
    try:
        fn(*fn_args, **fn_kwargs)
    except Exception as e:
        print(f"WARNING: {label} figure generation failed: {e}")


def _print_summary_table(args, rows: List[Tuple[str, dict]]) -> None:
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


def _result_columns(args) -> List[str]:
    columns = list(SIMILARITY_COLUMNS) + PARTIAL_MATCH_COLUMNS + PARTIAL_MATCH_MO_COLUMNS
    if args.mode == "classnumber":
        columns.extend(CLASSNUMBER_COLUMNS)
    return columns


def _ranking_columns(args) -> List[str]:
    columns = SIMILARITY_COLUMNS + ["count-partial", "count-partial-mo"]
    if args.mode == "classnumber":
        columns.extend(["best-classnumber-rank-score", "best-classnumber-mo-rank-score"])
    return columns


def _safe_identifier(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)[:80]


def _save_args_json(identifier_dir: Path, args: argparse.Namespace) -> None:
    """将 args 序列化为 JSON 保存到 identifier 文件夹下。"""
    def _serialize(val):
        if isinstance(val, Path):
            return str(val)
        if isinstance(val, tuple):
            return list(val)
        return val

    args_dict = {k: _serialize(v) for k, v in vars(args).items()}
    json_path = identifier_dir / "args.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False, default=str)
    print(f"Args saved: {json_path}")


def make_args(**overrides: Any) -> argparse.Namespace:
    """构造 args namespace，未指定的参数使用 cli_args 中的默认值。

    调用示例::

        from match.scripts.main import make_args, run

        for mode in ("count-partial", "classnumber"):
            rows = run(make_args(
                klarf_dir="/data/klarf/",
                reference="/data/ref.png",
                mode=mode,
                identifier=f"exp_{mode}",
            ))
    """
    defaults = parse_args([], validate=False)
    for key, value in overrides.items():
        setattr(defaults, key, value)
    _validate_experiment_args(defaults)
    return defaults


def _validate_experiment_args(args: argparse.Namespace) -> None:
    if not args.klarf_dir:
        raise ValueError("klarf_dir is required")
    if not args.reference:
        raise ValueError("reference is required")


if __name__ == "__main__":
    main()
