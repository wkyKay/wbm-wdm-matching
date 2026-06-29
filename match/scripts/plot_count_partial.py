#!/usr/bin/env python3
"""Render production count-map partial matching review figures."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..data.fileio import read_wbm_png, read_wbm_shape
from ..core.local_matching import compute_count_partial_match
from ..core.models import GridMaps
from ..core.pipeline import map_klarf_to_grid
from ..viz.count_partial_visualization import plot_count_partial_steps, plot_count_partial_topk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate WBM status + WDM count-map partial matching review figures."
    )
    parser.add_argument("--klarf-dir", required=True, help="Directory containing KLARF files to process.")
    parser.add_argument("--klarf-glob", default="*.*", help="Glob pattern to match KLARF files.")
    parser.add_argument("--reference", required=True, help="Path to reference WBM PNG or GridMaps npz.")
    parser.add_argument("--wbm", help="Path to WBM PNG for shape reference. Defaults to --reference.")
    parser.add_argument("--height", type=int, help="Target grid height.")
    parser.add_argument("--width", type=int, help="Target grid width.")
    parser.add_argument(
        "--mapper",
        default="die-index",
        choices=["die-index", "relative-coordinate", "physical-coordinate"],
        help="Coordinate mapper used for KLARF -> WBM grid.",
    )
    parser.add_argument("--representation", default="density", help="Grid representation kept in saved GridMaps metadata.")
    parser.add_argument("--die-x-range", nargs=2, type=int)
    parser.add_argument("--die-y-range", nargs=2, type=int)
    parser.add_argument("--defect-table-index", type=int, default=0)
    parser.add_argument("--out-dir", default="match/output/count_partial_review")
    parser.add_argument("--review-top-k", type=int, default=3, help="Number of top candidates shown in the TopK figure.")
    parser.add_argument("--step-max", type=int, default=3, help="Number of top candidates to render step figures for.")
    parser.add_argument("--min-area", type=int, default=5, help="Minimum token support area.")
    parser.add_argument("--top-k-proposals", type=int, default=6, help="Maximum WBM/WDM tokens retained per map.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ref_gm = _load_reference(args.reference)
    shape = _resolve_shape(args)

    klarf_dir = Path(args.klarf_dir)
    if not klarf_dir.is_dir():
        print(f"ERROR: --klarf-dir is not a directory: {klarf_dir}", file=sys.stderr)
        sys.exit(1)
    klarf_files = sorted(klarf_dir.glob(args.klarf_glob))
    if not klarf_files:
        print(f"ERROR: no files matched {args.klarf_glob!r} in {klarf_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    steps_dir = out_dir / "proposal_steps"
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i, path in enumerate(klarf_files, start=1):
        sys.stdout.write(f"[{i}/{len(klarf_files)}] {path.name} ... ")
        sys.stdout.flush()
        try:
            gm = map_klarf_to_grid(
                path,
                shape=shape,
                mapper_name=args.mapper,
                representation_name=args.representation,
                defect_table_index=args.defect_table_index,
                die_x_range=tuple(args.die_x_range) if args.die_x_range else None,
                die_y_range=tuple(args.die_y_range) if args.die_y_range else None,
            )
            score = compute_count_partial_match(
                ref_gm,
                gm,
                min_area=args.min_area,
                top_k=args.top_k_proposals,
            )
        except Exception as exc:
            print(f"ERROR ({type(exc).__name__}: {exc})")
            continue
        records.append((path.stem, gm, score))
        print(f"score={score.score:.4f}")

    if not records:
        print("ERROR: no KLARF files were processed successfully", file=sys.stderr)
        sys.exit(1)

    records.sort(key=lambda item: item[2].score, reverse=True)
    top_records = records[: max(args.review_top_k, 1)]
    topk_path = out_dir / f"top{len(top_records)}_count_partial.png"
    plot_count_partial_topk(
        ref_gm,
        [(name, gm) for name, gm, _ in top_records],
        title="Count-map partial matching top candidates",
        min_area=args.min_area,
        top_k=args.top_k_proposals,
        save_path=topk_path,
    )
    plt.close("all")
    print(f"Saved {topk_path}")

    for rank, (name, gm, score) in enumerate(records[: max(args.step_max, 0)], start=1):
        step_path = steps_dir / f"rank{rank:02d}_{_safe_name(name)}_steps.png"
        plot_count_partial_steps(
            ref_gm,
            gm,
            title=f"Rank {rank}: {name}",
            min_area=args.min_area,
            top_k=args.top_k_proposals,
            save_path=step_path,
        )
        plt.close("all")
        print(f"Saved {step_path}")


def _resolve_shape(args: argparse.Namespace) -> tuple[int, int]:
    if args.height is not None or args.width is not None:
        if args.height is None or args.width is None:
            raise ValueError("Pass both --height and --width, or neither.")
        return args.height, args.width
    return read_wbm_shape(args.wbm or args.reference)


def _load_reference(path: str | Path) -> GridMaps:
    path = Path(path)
    if path.suffix.lower() == ".png":
        return read_wbm_png(path)
    if path.suffix.lower() == ".npz":
        data = dict(np.load(path, allow_pickle=True))
        return GridMaps(
            count_map=data["count_map"],
            binary_map=data["binary_map"],
            density_map=data["density_map"],
            status_map=data["status_map"],
            representation_map=data["representation_map"],
            representation_maps={
                key: data[key]
                for key in ["binary", "count", "density", "soft", "three-value", "mountain"]
                if key in data
            },
            metadata=data.get("metadata", {}),
        )
    raise ValueError(f"Unsupported reference format: {path.suffix}")


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)[:80]


if __name__ == "__main__":
    main()
