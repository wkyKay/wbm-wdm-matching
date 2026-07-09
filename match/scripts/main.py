from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple

from .cli_args import parse_args, SIMILARITY_COLUMNS, RESULT_COLUMNS, CLASSNUMBER_COLUMNS
from .reference_loader import load_reference, resolve_shape
from .processing import process_one
from .batch_io import (
    write_map_match_log,
    write_result_log,
    write_token_match_log,
    write_topk_log,
    format_score,
)
from .batch_viz import save_classnumber_figures, save_count_partial_figures


def main() -> None:
    args = parse_args()

    klarf_files = _collect_klarf_files(args)
    ref_gm, shape = _load_reference_context(args)
    rows, summary = _run_batch(args, klarf_files, ref_gm, shape)
    write_result_log(args.log, rows, _result_columns(args), format_score)
    write_topk_log(args.topk_log, rows, _ranking_columns(args), args.topk)
    write_token_match_log(args.token_match_log, rows)
    write_map_match_log(args.map_match_log, rows)
    _save_requested_figures(args, ref_gm, rows)
    _print_summary_table(args, rows)
    print(
        f"\nDone in {summary['elapsed']:.1f}s: "
        f"{summary['ok']} OK, {summary['skipped']} skipped, {summary['error']} error"
    )


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
    print(f"Mapper: {args.mapper}, Representation: {args.representation}")
    print()
    return ref_gm, shape


def _run_batch(
    args,
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


def _save_requested_figures(args, ref_gm: "GridMaps", rows: List[Tuple[str, dict]]) -> None:
    if args.save_count_partial_figures:
        save_count_partial_figures(args, ref_gm, rows)
    if args.use_classnumber and args.save_classnumber_figures:
        save_classnumber_figures(args, ref_gm, rows)


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
    columns = list(RESULT_COLUMNS)
    if args.use_classnumber:
        columns.extend(CLASSNUMBER_COLUMNS)
    return columns


def _ranking_columns(args) -> List[str]:
    columns = SIMILARITY_COLUMNS + ["count-partial", "count-partial-mo"]
    if args.use_classnumber:
        columns.append("best-classnumber-rank-score")
        columns.append("best-classnumber-mo-rank-score")
    return columns


if __name__ == "__main__":
    main()
