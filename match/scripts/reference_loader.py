from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.fileio import read_wbm_shape, read_wbm_png


def resolve_shape(args) -> tuple[int, int]:
    if args.height is not None or args.width is not None:
        if args.height is None or args.width is None:
            raise ValueError("Pass both --height and --width, or pass neither and use --wbm.")
        return args.height, args.width

    wbm_path = args.wbm or args.reference
    if not wbm_path:
        raise ValueError("Pass --wbm, --reference, or both --height and --width.")
    return read_wbm_shape(wbm_path)


def load_reference(path: str | Path) -> "GridMaps":
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