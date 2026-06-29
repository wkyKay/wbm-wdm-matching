# Visualization for optional classnumber split matching.
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from ..core.classnumber_matching import ClassNumberMatchResult, split_score
from ..core.models import GridMaps, VALID_HAS_DEFECT, VALID_NO_DEFECT


CLASSNUMBER_CMAP = LinearSegmentedColormap.from_list(
    "classnumber_counts",
    ["#111111", "#22577a", "#4ea8de", "#9bd3f5", "#f4fbff"],
)


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
            ref_mask = (reference.status_map == VALID_HAS_DEFECT).astype(np.float32)
            ax.imshow(ref_mask, cmap=CLASSNUMBER_CMAP, vmin=0.0, vmax=1.0, interpolation="nearest")
            ax.set_title("WBM")
        else:
            log_count = _masked_log_count(gm.count_map, reference.status_map)
            last_im = ax.imshow(log_count, cmap=CLASSNUMBER_CMAP, vmin=0.0, vmax=vmax, interpolation="nearest")
            if split is None:
                ax.set_title("WDM all")
            else:
                marker = "  BEST" if split.classnumber == best_class else ""
                score = split_score(split, class_result.rank_by)
                ax.set_title(f"class {split.classnumber}: {score:.3f} ({class_result.rank_by}){marker}")
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


def plot_classnumber_topk_splits(
    reference: GridMaps,
    split_records: List[dict],
    top_k: int = 6,
    score_mode: str = "count",
    title: str = "",
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, List[plt.Axes]]:
    """Render global top-k classnumber split candidates sorted by selected score."""
    records = sorted(split_records, key=lambda item: split_score(item["split"], score_mode), reverse=True)
    records = records[: max(top_k, 1)]
    n = len(records)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.8 * rows))
    axes_list = np.asarray(axes).reshape(-1).tolist()

    if records:
        log_images = [_masked_log_count(rec["grid_maps"].count_map, reference.status_map) for rec in records]
        vmax = _heatmap_vmax(log_images)
    else:
        vmax = 1.0

    last_im = None
    for ax, rec in zip(axes_list, records):
        log_count = _masked_log_count(rec["grid_maps"].count_map, reference.status_map)
        last_im = ax.imshow(log_count, cmap=CLASSNUMBER_CMAP, vmin=0.0, vmax=vmax, interpolation="nearest")
        score = split_score(rec["split"], score_mode)
        ax.set_title(f"{rec['file']} / class {rec['classnumber']}\n{score_mode}: {score:.3f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_list[len(records):]:
        ax.axis("off")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes_list[:len(records)], fraction=0.025, pad=0.015)
        cbar.set_label("log1p(count)")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=0.03, right=0.92, bottom=0.04, top=0.90, wspace=0.18, hspace=0.28)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig, axes_list[:len(records)]


def plot_classnumber_step(
    reference: GridMaps,
    candidate: GridMaps,
    title: str = "",
    min_area: int = 5,
    top_k: int = 6,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, List[plt.Axes]]:
    """Render a classnumber split with its local partial matching steps."""
    from .count_partial_visualization import plot_count_partial_steps

    return plot_count_partial_steps(
        reference,
        candidate,
        title=title,
        min_area=min_area,
        top_k=top_k,
        save_path=save_path,
    )


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
