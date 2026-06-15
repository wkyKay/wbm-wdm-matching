from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from .mappers import MAPPERS
    from .representations import REPRESENTATIONS
    from .pipeline import map_klarf_to_grid
    from .io import read_wbm_shape, read_wbm_png, save_grid_maps
    from .similarity import SIMILARITIES, compute_similarity
except ImportError:
    from mappers import MAPPERS
    from representations import REPRESENTATIONS
    from pipeline import map_klarf_to_grid
    from io import read_wbm_shape, read_wbm_png, save_grid_maps
    from similarity import SIMILARITIES, compute_similarity


def parse_args() -> argparse.Namespace:
    # CLI 明确区分坐标映射策略和网格表达方式。
    parser = argparse.ArgumentParser(description="Map a KLARF WDM DefectList to a target WBM grid.")
    parser.add_argument("--klarf", required=True, help="Path to the input KLARF file.")
    parser.add_argument("--wbm", help="Path to the target WBM PNG. Used only to read HxW.")
    parser.add_argument("--height", type=int, help="Target WBM grid height. Overrides --wbm shape.")
    parser.add_argument("--width", type=int, help="Target WBM grid width. Overrides --wbm shape.")
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
        "--defect-table-index",
        type=int,
        default=0,
        help="Which DefectList to use when a KLARF contains multiple wafers.",
    )
    parser.add_argument(
        "--die-x-range",
        nargs=2,
        type=int,
        metavar=("MIN", "MAX"),
        help="Fixed die index range along X (e.g. --die-x-range -20 20). "
        "If omitted, derived from defect data.",
    )
    parser.add_argument(
        "--die-y-range",
        nargs=2,
        type=int,
        metavar=("MIN", "MAX"),
        help="Fixed die index range along Y (e.g. --die-y-range -20 20). "
        "If omitted, derived from defect data.",
    )
    parser.add_argument(
        "--output",
        default="match/output/wdm_grid_maps.npz",
        help="Output .npz path for all grid maps and selected representation_map.",
    )
    parser.add_argument(
        "--similarity",
        choices=sorted(SIMILARITIES),
        default=None,
        help="Pluggable similarity method. If set, computes score against --reference.",
    )
    parser.add_argument(
        "--reference",
        help="Path to reference WBM .npz grid map (required when --similarity is set).",
    )
    return parser.parse_args()


def resolve_shape(args: argparse.Namespace) -> tuple[int, int]:
    # 优先使用显式传入的高宽；否则从 PNG 文件中读取。
    if args.height is not None or args.width is not None:
        if args.height is None or args.width is None:
            raise ValueError("Pass both --height and --width, or pass neither and use --wbm.")
        return args.height, args.width

    if not args.wbm:
        raise ValueError("Pass --wbm, or pass both --height and --width.")

    return read_wbm_shape(args.wbm)


def main() -> None:
    # 端到端入口：读取 KLARF，映射到目标网格，并保存 npz 输出。
    args = parse_args()
    shape = resolve_shape(args)
    grid_maps = map_klarf_to_grid(
        args.klarf,
        shape=shape,
        mapper_name=args.mapper,
        representation_name=args.representation,
        defect_table_index=args.defect_table_index,
        die_x_range=tuple(args.die_x_range) if args.die_x_range else None,
        die_y_range=tuple(args.die_y_range) if args.die_y_range else None,
    )
    save_grid_maps(args.output, grid_maps)

    print(f"saved: {Path(args.output)}")
    print(f"shape: {grid_maps.count_map.shape}")
    print(f"coordinate_mapper: {grid_maps.metadata['coordinate_mapper']}")
    print(f"representation: {grid_maps.metadata['representation']}")
    print(f"defects: {grid_maps.metadata['mapped_defects']}/{grid_maps.metadata['input_defects']} mapped")

    if args.similarity:
        if not args.reference:
            raise ValueError("--reference is required when --similarity is set.")

        # 支持两种 reference 来源：.npz（已处理的 GridMaps）或 .png（WBM 图）
        ref_path = Path(args.reference)
        if ref_path.suffix.lower() == ".png":
            ref_gm = read_wbm_png(ref_path)
        else:
            ref_data = dict(np.load(ref_path, allow_pickle=True))
            from .models import GridMaps
            ref_gm = GridMaps(
                count_map=ref_data["count_map"],
                binary_map=ref_data["binary_map"],
                density_map=ref_data["density_map"],
                status_map=ref_data["status_map"],
                representation_map=ref_data["representation_map"],
                representation_maps={k: ref_data[k] for k in
                    ["binary", "count", "density", "soft", "three-value", "mountain"] if k in ref_data},
                metadata=ref_data.get("metadata", {}),
            )

        result = compute_similarity(
            ref_gm.representation_map,
            grid_maps.representation_map,
            method=args.similarity,
            reference_status=ref_gm.status_map,
            candidate_status=grid_maps.status_map,
        )

        try:
            from .similarity import SimilarityResult
        except ImportError:
            from similarity import SimilarityResult

        if isinstance(result, SimilarityResult):
            print(f"\nsimilarity ({args.similarity}):")
            print(f"  score      = {result.score:.6f}")
            if result.coverage is not None:
                print(f"  coverage   = {result.coverage:.6f}")
            if result.leakage is not None:
                print(f"  leakage    = {result.leakage:.6f}")
        else:
            print(f"\nsimilarity ({args.similarity}): {result:.6f}")


if __name__ == "__main__":
    main()
