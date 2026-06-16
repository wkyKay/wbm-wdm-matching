#!/usr/bin/env python3
"""
Usage:
    python -m match.plot_ref_cnd --ref reference.png --cnd candidate.npz
    python -m match.plot_ref_cnd --ref reference.png --klarf some_file

Shape always comes from --ref. No manual --height/--width needed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")  # headless

try:
    from .fileio import read_wbm_png
    from .pipeline import map_klarf_to_grid
    from .visualization import plot_comparison
except ImportError:
    from fileio import read_wbm_png
    from pipeline import map_klarf_to_grid
    from visualization import plot_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot WBM reference vs WDM candidate side-by-side.")
    parser.add_argument("--ref", required=True, help="Path to reference WBM PNG.")
    parser.add_argument(
        "--cnd", help="Path to candidate .npz (GridMaps). Mutually exclusive with --klarf."
    )
    parser.add_argument("--klarf", help="Path to a single KLARF file to process and plot.")
    parser.add_argument(
        "--mapper", default="die-index",
        choices=["die-index", "relative-coordinate", "physical-coordinate"],
    )
    parser.add_argument("--representation", default="density", help="Which map to compare.")
    parser.add_argument("--output", default=None, help="Output PNG path (default: auto-generated).")
    parser.add_argument("--die-x-range", nargs=2, type=int)
    parser.add_argument("--die-y-range", nargs=2, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ref_path = Path(args.ref)
    print(f"Loading reference: {ref_path}")
    ref_gm = read_wbm_png(ref_path)

    if args.cnd:
        import numpy as np
        cnd_data = dict(np.load(args.cnd, allow_pickle=True))
        try:
            from .models import GridMaps
        except ImportError:
            from models import GridMaps
        cnd_gm = GridMaps(
            count_map=cnd_data["count_map"],
            binary_map=cnd_data["binary_map"],
            density_map=cnd_data["density_map"],
            status_map=cnd_data["status_map"],
            representation_map=cnd_data["representation_map"],
            representation_maps={
                k: cnd_data[k]
                for k in ["binary", "count", "density", "soft", "three-value", "mountain"]
                if k in cnd_data
            },
            metadata=cnd_data.get("metadata", {}),
        )
        cnd_label = Path(args.cnd).stem
    elif args.klarf:
        # 用 reference 的 shape
        shape = (ref_gm.count_map.shape[0], ref_gm.count_map.shape[1])
        print(f"Processing KLARF: {args.klarf}  (shape={shape})")
        cnd_gm = map_klarf_to_grid(
            args.klarf,
            shape=shape,
            mapper_name=args.mapper,
            representation_name=args.representation,
            die_x_range=tuple(args.die_x_range) if args.die_x_range else None,
            die_y_range=tuple(args.die_y_range) if args.die_y_range else None,
        )
        cnd_label = Path(args.klarf).stem
    else:
        print("ERROR: provide --cnd or --klarf", file=sys.stderr)
        sys.exit(1)

    out = args.output or f"comparison_{cnd_label}_{args.representation}.png"
    print(f"Saving: {out}")
    plot_comparison(
        ref_gm, cnd_gm,
        representation=args.representation,
        ref_label=ref_path.stem,
        cnd_label=cnd_label,
        save_path=out,
    )
    print("Done.")


if __name__ == "__main__":
    main()
