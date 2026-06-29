# Visualization for optional classnumber split matching.
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from ..core.classnumber_matching import ClassNumberMatchResult
from ..core.models import GridMaps, VALID_HAS_DEFECT, VALID_NO_DEFECT
from .count_partial_visualization import STATUS_CMAP, STATUS_NORM


def plot_classnumber_splits(
    reference: GridMaps,
    full_candidate: GridMaps,
    class_result: ClassNumberMatchResult,
    title: str = "",
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, List[plt.Axes]]:
    """Render WBM, full WDM, and one WDM heatmap per classnumber."""
    panels = [("WBM", reference, None), ("WDM all", full_candidate, None)]
    for split in class_result.splits:
        panels.append((f"class {split.classnumber}", split.grid_maps, split))

    log_images = [_masked_log_count(gm.count_map, reference.status_map) for _, gm, split in panels if split is not None or gm is full_candidate]
    vmax = _heatmap_vmax(log_images)
    n = len(panels)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.8 * rows))
    axes_list = np.asarray(axes).reshape(-1).tolist()

    best_class = class_result.best.classnumber if class_result.best is not None else None
    last_im = None
    for ax, (label, gm, split) in zip(axes_list, panels):
        if label == "WBM":
            ax.imshow(reference.status_map, cmap=STATUS_CMAP, norm=STATUS_NORM, interpolation="nearest")
            ax.set_title("WBM")
        else:
            log_count = _masked_log_count(gm.count_map, reference.status_map)
            last_im = ax.imshow(log_count, cmap="magma", vmin=0.0, vmax=vmax, interpolation="nearest")
            if split is None:
                ax.set_title("WDM all")
            else:
                marker = "  BEST" if split.classnumber == best_class else ""
                ax.set_title(f"class {split.classnumber}: {split.partial.score:.3f}{marker}")
                if split.classnumber == best_class:
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_color("#00ff66")
                        spine.set_linewidth(3.0)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_list[len(panels):]:
        ax.axis("off")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes_list[:len(panels)], fraction=0.025, pad=0.015)
        cbar.set_label("log1p(count)")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=0.03, right=0.92, bottom=0.04, top=0.90, wspace=0.18, hspace=0.28)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig, axes_list[:len(panels)]


def _masked_log_count(count_map: np.ndarray, reference_status: np.ndarray) -> np.ndarray:
    valid = (reference_status == VALID_NO_DEFECT) | (reference_status == VALID_HAS_DEFECT)
    log_count = np.log1p(count_map.astype(np.float32))
    log_count[~valid] = np.nan
    return log_count


def _heatmap_vmax(images: List[np.ndarray]) -> float:
    values = []
    for image in images:
        finite = image[np.isfinite(image)]
        finite = finite[finite > 0]
        if len(finite):
            values.append(finite)
    if not values:
        return 1.0
    combined = np.concatenate(values)
    return float(max(np.percentile(combined, 95), combined.max() * 0.25, 1e-6))
