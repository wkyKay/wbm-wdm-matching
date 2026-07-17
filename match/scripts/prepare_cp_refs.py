from __future__ import annotations

import argparse
import csv
import json
import re
import struct
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


import sys


BACKGROUND = 0
VALID_NO_DEFECT = 127
VALID_HAS_DEFECT = 255


def main() -> None:
    args = parse_args()
    csv_paths = _collect_csv_paths(args)
    if not csv_paths:
        print("ERROR: no CSV files found. Provide --cp-csv paths or --cp-csv-dir.", file=sys.stderr)
        sys.exit(1)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    lot_summaries: Dict[str, List[Dict[str, object]]] = {}

    for csv_path in csv_paths:
        rows = _read_csv_rows(csv_path)
        _require_columns(rows, (args.lot_col, args.wafer_col, args.x_col, args.y_col, args.pf_col, args.hardbin_col))
        groups = _group_rows(rows, args.lot_col, args.wafer_col)
        count = 0
        for (lot_id, wafer_number), wafer_rows in sorted(groups.items()):
            _write_wafer_outputs(
                out_root=out_root,
                lot_id=lot_id,
                wafer_number=wafer_number,
                rows=wafer_rows,
                args=args,
            )
            count += 1
            lot_summaries.setdefault(lot_id, []).append({
                "lot_id": lot_id,
                "wafer_number": wafer_number,
                "csv_file": csv_path.name,
            })
        print(f"  {csv_path.name}: {count} wafer(s) -> {out_root}")

    total = 0
    for lot_id, rows in sorted(lot_summaries.items()):
        _write_summary(out_root / _safe_path_part(lot_id) / "summary.tsv", rows)
        total += len(rows)
    print(f"Total: {total} wafer(s) from {len(csv_paths)} CSV(s) under {out_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split CP test CSV(s) into per-wafer PF and hardbin reference PNGs."
    )
    parser.add_argument("--cp-csv", nargs="*", default=[], help="Input CP test CSV file(s).")
    parser.add_argument("--cp-csv-dir", default=None, help="Directory of CP test CSV files to process.")
    parser.add_argument("--cp-csv-glob", default="*.csv", help="Glob pattern when using --cp-csv-dir (default: *.csv).")
    parser.add_argument("--out-dir", default="match/output/cp_refs", help="Output cp_refs directory.")
    parser.add_argument("--lot-col", default="Lot_Id", help="CSV column used as lot directory name.")
    parser.add_argument("--wafer-col", default="Wafer_Number", help="CSV column used as wafer directory name.")
    parser.add_argument("--x-col", default="Die_X", help="Die X column.")
    parser.add_argument("--y-col", default="Die_Y", help="Die Y column.")
    parser.add_argument("--pf-col", default="Pf", help="Pass/fail result column.")
    parser.add_argument("--hardbin-col", default="Hardbin_Number", help="Hardbin number column.")
    parser.add_argument("--hardbin-name-col", default="Hardbin_Name", help="Optional hardbin name column.")
    parser.add_argument("--softbin-col", default="Softbin_Number", help="Softbin number column.")
    parser.add_argument("--softbin-name-col", default="Softbin_Name", help="Optional softbin name column.")
    return parser.parse_args()


def _collect_csv_paths(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []
    for p in args.cp_csv:
        path = Path(p)
        if path.is_file():
            paths.append(path.resolve())
        else:
            print(f"WARNING: --cp-csv file not found: {p}", file=sys.stderr)
    if args.cp_csv_dir:
        csv_dir = Path(args.cp_csv_dir)
        if csv_dir.is_dir():
            for p in sorted(csv_dir.glob(args.cp_csv_glob)):
                if p.is_file():
                    paths.append(p.resolve())
        else:
            print(f"WARNING: --cp-csv-dir not a directory: {args.cp_csv_dir}", file=sys.stderr)
    return paths


def _read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        return [dict(row) for row in reader]


def _require_columns(rows: List[Dict[str, str]], columns: Iterable[str]) -> None:
    if not rows:
        raise ValueError("Input CSV has no data rows.")
    available = set(rows[0].keys())
    missing = [column for column in columns if column and column not in available]
    if missing:
        raise ValueError(f"Input CSV missing required column(s): {', '.join(missing)}")


def _group_rows(rows: List[Dict[str, str]], lot_col: str, wafer_col: str) -> Dict[Tuple[str, str], List[Dict[str, str]]]:
    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in rows:
        lot_id = _cell(row, lot_col, fallback="UNKNOWN_LOT")
        wafer_number = _cell(row, wafer_col, fallback="UNKNOWN_WAFER")
        groups.setdefault((lot_id, wafer_number), []).append(row)
    return groups


def _write_wafer_outputs(
    out_root: Path,
    lot_id: str,
    wafer_number: str,
    rows: List[Dict[str, str]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    wafer_dir = out_root / _safe_path_part(lot_id) / _safe_path_part(wafer_number)
    hardbin_dir = wafer_dir / "hardbin"
    softbin_dir = wafer_dir / "softbin"
    hardbin_dir.mkdir(parents=True, exist_ok=True)
    softbin_dir.mkdir(parents=True, exist_ok=True)

    die_records = _die_records(rows, args.x_col, args.y_col)
    x_min, x_max, y_min, y_max = _die_bounds(die_records)
    shape = (y_max - y_min + 1, x_max - x_min + 1)
    valid_mask = _valid_die_mask(die_records, shape, x_min, y_max)

    _write_original_rows(wafer_dir / "die_results.csv", rows)
    _write_png(wafer_dir / "pf.png", _pf_pixels(rows, die_records, valid_mask, shape, x_min, y_max, args.pf_col))

    hardbin_rows = _write_hardbin_pngs(
        hardbin_dir=hardbin_dir,
        rows=rows,
        die_records=die_records,
        valid_mask=valid_mask,
        shape=shape,
        x_min=x_min,
        y_max=y_max,
        hardbin_col=args.hardbin_col,
        hardbin_name_col=args.hardbin_name_col,
    )
    _write_tsv(hardbin_dir / "hardbin_index.tsv", hardbin_rows)

    softbin_rows = _write_softbin_pngs(
        softbin_dir=softbin_dir,
        rows=rows,
        die_records=die_records,
        valid_mask=valid_mask,
        shape=shape,
        x_min=x_min,
        y_max=y_max,
        softbin_col=args.softbin_col,
        softbin_name_col=args.softbin_name_col,
    )
    if softbin_rows:
        _write_tsv(softbin_dir / "softbin_index.tsv", softbin_rows)

    metadata = _metadata(
        rows=rows,
        lot_id=lot_id,
        wafer_number=wafer_number,
        shape=shape,
        bounds=(x_min, x_max, y_min, y_max),
        valid_die_count=int(valid_mask.sum()),
        hardbin_count=len(hardbin_rows),
        softbin_count=len(softbin_rows),
        args=args,
    )
    (wafer_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "lot_id": lot_id,
        "wafer_number": wafer_number,
        "wafer_dir": str(wafer_dir),
        "height": shape[0],
        "width": shape[1],
        "valid_die_count": int(valid_mask.sum()),
        "hardbin_count": len(hardbin_rows),
        "softbin_count": len(softbin_rows),
    }


def _die_records(rows: List[Dict[str, str]], x_col: str, y_col: str) -> List[Tuple[Dict[str, str], int, int]]:
    records = []
    for row in rows:
        try:
            x = int(float(_cell(row, x_col)))
            y = int(float(_cell(row, y_col)))
        except ValueError:
            continue
        records.append((row, x, y))
    if not records:
        raise ValueError("No rows have valid die coordinates.")
    return records


def _die_bounds(die_records: List[Tuple[Dict[str, str], int, int]]) -> Tuple[int, int, int, int]:
    xs = [x for _, x, _ in die_records]
    ys = [y for _, _, y in die_records]
    return min(xs), max(xs), min(ys), max(ys)


def _valid_die_mask(
    die_records: List[Tuple[Dict[str, str], int, int]],
    shape: Tuple[int, int],
    x_min: int,
    y_max: int,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for _, x, y in die_records:
        row, col = _grid_position(x, y, x_min, y_max)
        mask[row, col] = True
    return mask


def _pf_pixels(
    rows: List[Dict[str, str]],
    die_records: List[Tuple[Dict[str, str], int, int]],
    valid_mask: np.ndarray,
    shape: Tuple[int, int],
    x_min: int,
    y_max: int,
    pf_col: str,
) -> np.ndarray:
    pixels = _base_pixels(valid_mask)
    for row, x, y in die_records:
        if _is_fail(_cell(row, pf_col)):
            grid_row, grid_col = _grid_position(x, y, x_min, y_max)
            pixels[grid_row, grid_col] = VALID_HAS_DEFECT
    return pixels


def _write_hardbin_pngs(
    hardbin_dir: Path,
    rows: List[Dict[str, str]],
    die_records: List[Tuple[Dict[str, str], int, int]],
    valid_mask: np.ndarray,
    shape: Tuple[int, int],
    x_min: int,
    y_max: int,
    hardbin_col: str,
    hardbin_name_col: str,
) -> List[Dict[str, object]]:
    hardbins = sorted(
        {
            _cell(row, hardbin_col)
            for row in rows
            if _cell(row, hardbin_col) != ""
        },
        key=_sort_key,
    )

    index_rows = []
    for hardbin in hardbins:
        pixels = _base_pixels(valid_mask)
        die_count = 0
        hardbin_names = set()
        for row, x, y in die_records:
            if _cell(row, hardbin_col) != hardbin:
                continue
            grid_row, grid_col = _grid_position(x, y, x_min, y_max)
            pixels[grid_row, grid_col] = VALID_HAS_DEFECT
            die_count += 1
            if hardbin_name_col in row and _cell(row, hardbin_name_col):
                hardbin_names.add(_cell(row, hardbin_name_col))

        hardbin_name = "|".join(sorted(hardbin_names))
        filename = _hardbin_filename(hardbin, hardbin_name)
        _write_png(hardbin_dir / filename, pixels)
        index_rows.append(
            {
                "hardbin_number": hardbin,
                "hardbin_name": hardbin_name,
                "die_count": die_count,
                "png_path": filename,
            }
        )
    return index_rows


def _write_softbin_pngs(
    softbin_dir: Path,
    rows: List[Dict[str, str]],
    die_records: List[Tuple[Dict[str, str], int, int]],
    valid_mask: np.ndarray,
    shape: Tuple[int, int],
    x_min: int,
    y_max: int,
    softbin_col: str,
    softbin_name_col: str,
) -> List[Dict[str, object]]:
    if softbin_col not in rows[0]:
        return []
    softbins = sorted(
        {
            _cell(row, softbin_col)
            for row in rows
            if _cell(row, softbin_col) != ""
        },
        key=_sort_key,
    )

    index_rows = []
    for softbin in softbins:
        pixels = _base_pixels(valid_mask)
        die_count = 0
        softbin_names = set()
        for row, x, y in die_records:
            if _cell(row, softbin_col) != softbin:
                continue
            grid_row, grid_col = _grid_position(x, y, x_min, y_max)
            pixels[grid_row, grid_col] = VALID_HAS_DEFECT
            die_count += 1
            if softbin_name_col in row and _cell(row, softbin_name_col):
                softbin_names.add(_cell(row, softbin_name_col))

        softbin_name = "|".join(sorted(softbin_names))
        filename = _softbin_filename(softbin, softbin_name)
        _write_png(softbin_dir / filename, pixels)
        index_rows.append(
            {
                "softbin_number": softbin,
                "softbin_name": softbin_name,
                "die_count": die_count,
                "png_path": filename,
            }
        )
    return index_rows


def _base_pixels(valid_mask: np.ndarray) -> np.ndarray:
    pixels = np.full(valid_mask.shape, BACKGROUND, dtype=np.uint8)
    pixels[valid_mask] = VALID_NO_DEFECT
    return pixels


def _grid_position(x: int, y: int, x_min: int, y_max: int) -> Tuple[int, int]:
    return y_max - y, x - x_min


def _is_fail(value: str) -> bool:
    normalized = value.strip().upper()
    if normalized in {"F", "FAIL", "FAILED", "FALSE", "0", "N", "NO"}:
        return True
    if normalized in {"P", "PASS", "PASSED", "TRUE", "1", "Y", "YES"}:
        return False
    return normalized not in {"", "NA", "N/A", "NULL", "NONE"}


def _metadata(
    rows: List[Dict[str, str]],
    lot_id: str,
    wafer_number: str,
    shape: Tuple[int, int],
    bounds: Tuple[int, int, int, int],
    valid_die_count: int,
    hardbin_count: int,
    softbin_count: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    x_min, x_max, y_min, y_max = bounds
    return {
        "lot_id": lot_id,
        "wafer_number": wafer_number,
        "row_count": len(rows),
        "valid_die_count": valid_die_count,
        "hardbin_count": hardbin_count,
        "softbin_count": softbin_count,
        "height": shape[0],
        "width": shape[1],
        "die_x_min": x_min,
        "die_x_max": x_max,
        "die_y_min": y_min,
        "die_y_max": y_max,
        "x_col": args.x_col,
        "y_col": args.y_col,
        "pf_col": args.pf_col,
        "hardbin_col": args.hardbin_col,
        "softbin_col": args.softbin_col,
        "product_id": _unique_text(rows, "Product_Id"),
        "test_program": _unique_text(rows, "Test_Program"),
        "stage": _unique_text(rows, "Stage"),
        "vendor": _unique_text(rows, "Vendor"),
        "tester": _unique_text(rows, "Tester"),
        "source_lot": _unique_text(rows, "Source_Lot"),
        "wafer_id": _unique_text(rows, "Wafer_Id"),
        "start_time_min": _min_text(rows, "Start_Time"),
        "end_time_max": _max_text(rows, "End_Time"),
    }


def _write_original_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, rows: List[Dict[str, object]]) -> None:
    _write_tsv(path, rows)


def _write_tsv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_png(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.asarray(pixels, dtype=np.uint8)
    height, width = pixels.shape
    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))
    png = [
        b"\x89PNG\r\n\x1a\n",
        _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)),
        _png_chunk(b"IDAT", zlib.compress(raw)),
        _png_chunk(b"IEND", b""),
    ]
    path.write_bytes(b"".join(png))


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)


def _cell(row: Dict[str, str], column: str, fallback: str = "") -> str:
    value = row.get(column, fallback)
    return "" if value is None else str(value).strip()


def _unique_text(rows: List[Dict[str, str]], column: str) -> str:
    values = sorted({_cell(row, column) for row in rows if _cell(row, column)})
    return "|".join(values)


def _min_text(rows: List[Dict[str, str]], column: str) -> str:
    values = [_cell(row, column) for row in rows if _cell(row, column)]
    return min(values) if values else ""


def _max_text(rows: List[Dict[str, str]], column: str) -> str:
    values = [_cell(row, column) for row in rows if _cell(row, column)]
    return max(values) if values else ""


def _safe_path_part(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:120] or "UNKNOWN"


def _hardbin_filename(hardbin_number: str, hardbin_name: str) -> str:
    name_part = _safe_path_part(hardbin_name) if hardbin_name else "hardbin"
    number_part = _safe_path_part(hardbin_number)
    return f"{number_part}_{name_part}.png"


def _softbin_filename(softbin_number: str, softbin_name: str) -> str:
    name_part = _safe_path_part(softbin_name) if softbin_name else "softbin"
    number_part = _safe_path_part(softbin_number)
    return f"{number_part}_{name_part}.png"


def _sort_key(value: str) -> Tuple[int, object]:
    try:
        return 0, int(float(value))
    except ValueError:
        return 1, value


if __name__ == "__main__":
    main()
