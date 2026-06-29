# Visualization for production count-map partial matching.
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.colors import LinearSegmentedColormap

from ..core.local_matching import explain_count_partial_match
from ..core.models import GridMaps, VALID_HAS_DEFECT, VALID_NO_DEFECT


STATUS_CMAP = ListedColormap(["black", "#b8b8b8", "#d62728", "#444444"])
STATUS_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], STATUS_CMAP.N)
COUNT_PARTIAL_CMAP = LinearSegmentedColormap.from_list(
    "count_partial_counts",
    ["#111111", "#7f1d1d", "#dc2626", "#fca5a5", "#fff1f2"],
)
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
    save_path: str | Path | None = None,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Render one WBM/WDM pair as proposal-to-match steps."""
    explanation = explain_count_partial_match(reference, candidate, min_area=min_area, top_k=top_k)
    result = explanation["result"]
    wbm_tokens = explanation["wbm_tokens"]
    wdm_tokens = explanation["wdm_tokens"]
    matches = explanation["matches"]

    log_count = _masked_log_count(candidate.count_map, reference.status_map)
    vmax = _heatmap_vmax([log_count])

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))
    _plot_wbm_defects(axes[0], reference, "WBM defects")
    _plot_wbm_status(axes[1], reference, "WBM tokens")
    _draw_tokens(axes[1], wbm_tokens)

    im = _plot_wdm_heatmap(axes[2], log_count, "WDM count", vmax=vmax)
    _plot_wdm_heatmap(axes[3], log_count, "WDM tokens", vmax=vmax)
    _draw_tokens(axes[3], wdm_tokens)

    _plot_wdm_heatmap(axes[4], log_count, "Local matches", vmax=vmax)
    _draw_tokens(axes[4], wbm_tokens, linestyle="--", linewidth=1.2)
    _draw_tokens(axes[4], wdm_tokens, linewidth=1.6)
    _draw_matches(axes[4], matches)

    cbar = fig.colorbar(im, ax=axes[2:5], fraction=0.025, pad=0.015)
    cbar.set_label("log1p(count)")

    subtitle = (
        f"count-partial={result.score:.3f}  "
        f"shape={result.mean_shape:.3f} pos={result.mean_position:.3f} "
        f"scale={result.mean_scale:.3f} type={result.mean_type:.3f}  "
        f"tokens={result.matched_tokens}/{result.wbm_tokens}/{result.wdm_tokens}"
    )
    fig.suptitle(f"{title}\n{subtitle}" if title else subtitle, fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.94, bottom=0.04, top=0.80, wspace=0.12)

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
    save_path: str | Path | None = None,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Render reference WBM tokens against top candidate WDM count heatmaps."""
    explanations = [
        (name, gm, explain_count_partial_match(reference, gm, min_area=min_area, top_k=top_k))
        for name, gm in candidates
    ]
    log_counts = [_masked_log_count(gm.count_map, reference.status_map) for _, gm, _ in explanations]
    vmax = _heatmap_vmax(log_counts)

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

        log_count = _masked_log_count(gm.count_map, reference.status_map)
        last_im = _plot_wdm_heatmap(ax_cnd, log_count, f"{row + 1}. {name}", vmax=vmax)
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

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes_arr[:, 1].ravel().tolist(), fraction=0.025, pad=0.015)
        cbar.set_label("log1p(count)")
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=0.04, right=0.92, bottom=0.05, top=0.92, wspace=0.12, hspace=0.35)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig, all_axes


def _plot_wbm_status(ax: plt.Axes, grid_maps: GridMaps, title: str) -> None:
    ax.imshow(grid_maps.status_map, cmap=STATUS_CMAP, norm=STATUS_NORM, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def _plot_wbm_defects(ax: plt.Axes, grid_maps: GridMaps, title: str) -> None:
    defect_mask = (grid_maps.status_map == VALID_HAS_DEFECT).astype(np.float32)
    ax.imshow(defect_mask, cmap=COUNT_PARTIAL_CMAP, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def _plot_wdm_heatmap(ax: plt.Axes, log_count: np.ndarray, title: str, vmax: float):
    im = ax.imshow(log_count, cmap=COUNT_PARTIAL_CMAP, vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    return im


def _masked_log_count(count_map: np.ndarray, reference_status: np.ndarray) -> np.ndarray:
    valid = (reference_status == VALID_NO_DEFECT) | (reference_status == VALID_HAS_DEFECT)
    log_count = np.log1p(count_map.astype(np.float32))
    log_count[~valid] = 0.0
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
