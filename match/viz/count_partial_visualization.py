# Visualization for production count-map partial matching.
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from ..core.local_matching import explain_count_partial_match
from ..core.models import GridMaps, VALID_HAS_DEFECT, VALID_NO_DEFECT


STATUS_CMAP = ListedColormap(["black", "#7f7f7f", "#f2f2f2", "#444444"])
STATUS_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], STATUS_CMAP.N)
COUNT_PARTIAL_CMAP = LinearSegmentedColormap.from_list(
    "count_partial_counts",
    ["#f2f2f2", "#fca5a5", "#ef4444", "#991b1b", "#450a0a"],
)
COUNT_PARTIAL_CMAP.set_bad("black")
TOKEN_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#17becf",
    "#e377c2", "#bcbd22", "#7f7f7f", "#d62728",
]


def plot_count_partial_steps(
    reference: GridMaps,
    candidate: GridMaps,
    title: str = "",
    min_area: int = 5,
    top_k: int = 6,
    proposal_mode: str = "cc",
    rotation_tolerance: bool = False,
    save_path: str | Path | None = None,
    explain_fn=explain_count_partial_match,
    map_mode: str = "count",
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Render one WBM/WDM pair as proposal-to-match steps."""
    explanation = explain_fn(
        reference,
        candidate,
        min_area=min_area,
        top_k=top_k,
        proposal_mode=proposal_mode,
        rotation_tolerance=rotation_tolerance,
    )
    result = explanation["result"]
    wbm_tokens = explanation["wbm_tokens"]
    wdm_tokens = explanation["wdm_tokens"]
    matches = explanation["matches"]

    wdm_image = _masked_wdm_image(candidate, reference.status_map, map_mode=map_mode)
    vmax = _heatmap_vmax([wdm_image])
    label = _wdm_image_label(map_mode)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))
    _plot_wbm_defects(axes[0], reference, "WBM defects")
    _plot_wbm_status(axes[1], reference, "WBM tokens")
    _draw_tokens(axes[1], wbm_tokens)

    im = _plot_wdm_heatmap(axes[2], wdm_image, f"WDM {map_mode}", vmax=vmax)
    _plot_wdm_heatmap(axes[3], wdm_image, "WDM tokens", vmax=vmax)
    _draw_tokens(axes[3], wdm_tokens)

    _plot_wdm_heatmap(axes[4], wdm_image, "Local matches", vmax=vmax)
    _draw_tokens(axes[4], wbm_tokens, linestyle="--", linewidth=1.2)
    _draw_tokens(axes[4], wdm_tokens, linewidth=1.6)
    _draw_matches(axes[4], matches)

    subtitle = (
        f"count-partial={result.score:.3f}  "
        f"shape={result.mean_shape:.3f} pos={result.mean_position:.3f} "
        f"scale={result.mean_scale:.3f} type={result.mean_type:.3f}  "
        f"tokens={result.matched_tokens}/{result.wbm_tokens}/{result.wdm_tokens}"
    )
    fig.suptitle(f"{title}\n{subtitle}" if title else subtitle, fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.90, bottom=0.04, top=0.80, wspace=0.12)
    _add_edge_colorbar(fig, im, label)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig, list(axes)


def plot_count_partial_topk(
    reference: GridMaps,
    candidates: List[Tuple[str, GridMaps]],
    title: str = "Top candidates",
    min_area: int = 5,
    top_k: int = 6,
    proposal_mode: str = "cc",
    rotation_tolerance: bool = False,
    save_path: str | Path | None = None,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Render reference WBM tokens against top candidate WDM count heatmaps."""
    explanations = [
        (
            name,
            gm,
            explain_count_partial_match(
                reference,
                gm,
                min_area=min_area,
                top_k=top_k,
                proposal_mode=proposal_mode,
                rotation_tolerance=rotation_tolerance,
            ),
        )
        for name, gm in candidates
    ]
    count_images = [_masked_count(gm.count_map, reference.status_map) for _, gm, _ in explanations]
    vmax = _heatmap_vmax(count_images)

    n = max(len(explanations), 1)
    fig, axes = plt.subplots(n, 2, figsize=(9.5, 3.4 * n))
    axes_arr = np.asarray(axes).reshape(n, 2)
    all_axes: List[plt.Axes] = []
    last_im = None

    for row, (name, gm, explanation) in enumerate(explanations):
        result = explanation["result"]
        wbm_tokens = explanation["wbm_tokens"]
        wdm_tokens = explanation["wdm_tokens"]
        matches = explanation["matches"]

        ax_ref, ax_cnd = axes_arr[row]
        _plot_wbm_defects(ax_ref, reference, "Reference WBM tokens")
        _draw_tokens(ax_ref, wbm_tokens)

        count_image = _masked_count(gm.count_map, reference.status_map)
        last_im = _plot_wdm_heatmap(ax_cnd, count_image, f"{row + 1}. {name}", vmax=vmax)
        _draw_tokens(ax_cnd, wdm_tokens)
        _draw_tokens(ax_cnd, wbm_tokens, linestyle="--", linewidth=1.0)
        _draw_matches(ax_cnd, matches)
        ax_cnd.text(
            0.01,
            -0.08,
            (
                f"score={result.score:.3f}  shape={result.mean_shape:.3f} "
                f"pos={result.mean_position:.3f} scale={result.mean_scale:.3f} "
                f"type={result.mean_type:.3f}  tokens={result.matched_tokens}/{result.wbm_tokens}/{result.wdm_tokens}"
            ),
            transform=ax_cnd.transAxes,
            fontsize=8,
            va="top",
        )
        all_axes.extend([ax_ref, ax_cnd])

    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=0.04, right=0.88, bottom=0.05, top=0.92, wspace=0.12, hspace=0.35)
    if last_im is not None:
        _add_edge_colorbar(fig, last_im, "count")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig, all_axes


def _plot_wbm_status(ax: plt.Axes, grid_maps: GridMaps, title: str) -> None:
    ax.imshow(grid_maps.status_map, cmap=STATUS_CMAP, norm=STATUS_NORM, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def _plot_wbm_defects(ax: plt.Axes, grid_maps: GridMaps, title: str) -> None:
    ax.imshow(grid_maps.status_map, cmap=STATUS_CMAP, norm=STATUS_NORM, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def _plot_wdm_heatmap(ax: plt.Axes, image: np.ndarray, title: str, vmax: float):
    im = ax.imshow(image, cmap=COUNT_PARTIAL_CMAP, vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    return im


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


def _masked_wdm_image(candidate: GridMaps, reference_status: np.ndarray, map_mode: str) -> np.ndarray:
    if map_mode == "binary":
        return _masked_binary(candidate.binary_map, reference_status)
    return _masked_count(candidate.count_map, reference_status)


def _wdm_image_label(map_mode: str) -> str:
    if map_mode == "binary":
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


def _draw_tokens(
    ax: plt.Axes,
    tokens: List[Dict],
    linestyle: str = "-",
    linewidth: float = 1.8,
) -> None:
    for idx, token in enumerate(tokens):
        color = TOKEN_COLORS[idx % len(TOKEN_COLORS)]
        mask = _token_mask(token)
        if mask is not None and mask.any():
            ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=linewidth, linestyles=linestyle)
        ax.text(
            token.get("centroid_col", 0.0),
            token.get("centroid_row", 0.0),
            str(idx),
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            bbox={"boxstyle": "circle,pad=0.18", "fc": color, "ec": "white", "lw": 0.5, "alpha": 0.9},
        )


def _draw_matches(ax: plt.Axes, matches: List[Dict]) -> None:
    for match in matches:
        qt = match["query_token"]
        ct = match["candidate_token"]
        color = TOKEN_COLORS[int(match["query_token_id"]) % len(TOKEN_COLORS)]
        ax.plot(
            [qt.get("centroid_col", 0.0), ct.get("centroid_col", 0.0)],
            [qt.get("centroid_row", 0.0), ct.get("centroid_row", 0.0)],
            color=color,
            linewidth=1.0,
            alpha=0.75,
        )


def _token_mask(token: Dict) -> np.ndarray | None:
    pixels = token.get("pixels", [])
    if not pixels:
        return None
    rows = [int(p[0]) for p in pixels]
    cols = [int(p[1]) for p in pixels]
    shape = token.get("map_shape")
    if shape:
        height, width = int(shape[0]), int(shape[1])
    else:
        height = max(rows) + 1
        width = max(cols) + 1
    mask = np.zeros((height, width), dtype=bool)
    mask[rows, cols] = True
    return mask
