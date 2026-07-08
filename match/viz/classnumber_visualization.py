# Visualization for optional classnumber split matching.
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import MaxNLocator

from ..core.classnumber_matching import ClassNumberMatchResult, split_score
from ..core.models import GridMaps, VALID_HAS_DEFECT, VALID_NO_DEFECT


STATUS_CMAP = ListedColormap(["black", "#7f7f7f", "#f2f2f2", "#444444"])
STATUS_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], STATUS_CMAP.N)
CLASSNUMBER_CMAP = LinearSegmentedColormap.from_list(
    "classnumber_counts",
    ["#7f7f7f", "#fca5a5", "#ef4444", "#991b1b", "#450a0a"],
)
CLASSNUMBER_CMAP.set_bad("black")
CLASSNUMBER_CMAP.set_under("#7f7f7f")


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

    score_mode = class_result.rank_by
    images = [_masked_wdm_image(gm, reference.status_map, score_mode) for _, gm, split in panels if split is not None or gm is full_candidate]
    vmax = _heatmap_vmax(images)
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
            image = _masked_wdm_image(gm, reference.status_map, score_mode)
            last_im = ax.imshow(image, cmap=CLASSNUMBER_CMAP, vmin=1e-6, vmax=vmax, interpolation="nearest")
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

    if title:
        fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=0.03, right=0.88, bottom=0.04, top=0.90, wspace=0.18, hspace=0.28)
    if last_im is not None:
        _add_edge_colorbar(fig, last_im, _wdm_image_label(score_mode))

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
        images = [_masked_wdm_image(rec["grid_maps"], reference.status_map, score_mode) for rec in records]
        vmax = _heatmap_vmax(images)
    else:
        vmax = 1.0

    last_im = None
    for ax, rec in zip(axes_list, records):
        image = _masked_wdm_image(rec["grid_maps"], reference.status_map, score_mode)
        last_im = ax.imshow(image, cmap=CLASSNUMBER_CMAP, vmin=1e-6, vmax=vmax, interpolation="nearest")
        score = split_score(rec["split"], score_mode)
        ax.set_title(f"{rec['file']} / class {rec['classnumber']}\n{score_mode}: {score:.3f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_list[len(records):]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=0.03, right=0.88, bottom=0.04, top=0.90, wspace=0.18, hspace=0.28)
    if last_im is not None:
        _add_edge_colorbar(fig, last_im, _wdm_image_label(score_mode))

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig, axes_list[:len(records)]


def plot_classnumber_step(
    reference: GridMaps,
    candidate: GridMaps,
    score_mode: str = "count",
    title: str = "",
    min_area: int = 5,
    top_k: int = 6,
    proposal_mode: str = "cc",
    rotation_tolerance: bool = False,
    min_token_score: float = 0.10,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, List[plt.Axes]]:
    """Render a classnumber split with its local partial matching steps."""
    from .count_partial_visualization import plot_count_partial_steps
    from ..core.local_matching import explain_binary_partial_match, explain_count_partial_match

    explain_fn = explain_binary_partial_match if score_mode == "binary" else explain_count_partial_match

    return plot_count_partial_steps(
        reference,
        candidate,
        title=title,
        min_area=min_area,
        top_k=top_k,
        proposal_mode=proposal_mode,
        rotation_tolerance=rotation_tolerance,
        min_token_score=min_token_score,
        save_path=save_path,
        explain_fn=explain_fn,
        map_mode=score_mode,
    )


def _masked_count(count_map: np.ndarray, reference_status: np.ndarray) -> np.ndarray:
    valid = (reference_status == VALID_NO_DEFECT) | (reference_status == VALID_HAS_DEFECT)
    image = count_map.astype(np.float32)
    image[~valid] = np.nan
    return image


def _masked_binary(binary_map: np.ndarray, reference_status: np.ndarray) -> np.ndarray:
    valid = (reference_status == VALID_NO_DEFECT) | (reference_status == VALID_HAS_DEFECT)
    image = (binary_map > 0).astype(np.float32)
    image[~valid] = np.nan
    return image


def _masked_wdm_image(grid_maps: GridMaps, reference_status: np.ndarray, score_mode: str) -> np.ndarray:
    if score_mode == "binary":
        return _masked_binary(grid_maps.binary_map, reference_status)
    return _masked_count(grid_maps.count_map, reference_status)


def _wdm_image_label(score_mode: str) -> str:
    if score_mode == "binary":
        return "binary defect"
    return "count"


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
    vmax = float(max(np.percentile(combined, 95), combined.max() * 0.25, 1e-6))
    return float(np.ceil(vmax))


def _add_edge_colorbar(fig: plt.Figure, image, label: str) -> None:
    cax = fig.add_axes([0.925, 0.14, 0.018, 0.68])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label(label)
    if label in {"count", "binary defect"}:
        cbar.locator = MaxNLocator(integer=True)
        cbar.update_ticks()
