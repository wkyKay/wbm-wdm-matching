# Visualization for production count-map partial matching.
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from ..core.local_matching import explain_count_partial_match
from ..core.models import GridMaps, VALID_HAS_DEFECT, VALID_NO_DEFECT


STATUS_CMAP = ListedColormap(["black", "#7f7f7f", "#f2f2f2", "#444444"])
STATUS_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], STATUS_CMAP.N)
COUNT_PARTIAL_CMAP = LinearSegmentedColormap.from_list(
    "count_partial_counts",
    ["#93c5fd", "#60a5fa", "#3b82f6", "#1d4ed8", "#1e3a5f"],
)
COUNT_PARTIAL_CMAP.set_bad("black")
COUNT_PARTIAL_CMAP.set_under("#7f7f7f")
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
    min_token_score: float = 0.30,
    score_shape_weight: float = 0.60,
    score_position_weight: float = 0.25,
    score_scale_weight: float = 0.15,
    min_relative_token_area: float = 0.10,
    scale_area_weight: float = 0.50,
    scale_pca_weight: float = 0.50,
    scale_ratio_min: float = 0.20,
    density_sigmas: tuple[float, ...] = (0.8, 1.6, 3.2),
    density_threshold: float = 0.20,
    density_min_raw_points: int = 3,
    density_min_raw_mass: float = 3.0,
    density_merge_iou: float = 0.60,
    density_weight_transform: str = "sqrt",
    ring_min_area: int | None = None,
    ring_edge_r_min: float | None = None,
    ring_band_width: float | None = None,
    ring_min_angular_coverage: float | None = None,
    ring_angular_bins: int | None = None,
    ring_max_radial_std: float | None = None,
    ring_max_defect_ratio: float | None = None,
    ring_min_edge_defect_fraction: float | None = None,
    save_path: str | Path | None = None,
    explain_fn=explain_count_partial_match,
    map_mode: str = "count",
    result_key: str = "result",
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Render one WBM/WDM pair as proposal-to-match steps."""
    explanation = explain_fn(
        reference,
        candidate,
        min_area=min_area,
        top_k=top_k,
        proposal_mode=proposal_mode,
        rotation_tolerance=rotation_tolerance,
        min_token_score=min_token_score,
        score_shape_weight=score_shape_weight,
        score_position_weight=score_position_weight,
        score_scale_weight=score_scale_weight,
        min_relative_token_area=min_relative_token_area,
        scale_area_weight=scale_area_weight,
        scale_pca_weight=scale_pca_weight,
        scale_ratio_min=scale_ratio_min,
        density_sigmas=density_sigmas,
        density_threshold=density_threshold,
        density_min_raw_points=density_min_raw_points,
        density_min_raw_mass=density_min_raw_mass,
        density_merge_iou=density_merge_iou,
        density_weight_transform=density_weight_transform,
        ring_min_area=ring_min_area,
        ring_edge_r_min=ring_edge_r_min,
        ring_band_width=ring_band_width,
        ring_min_angular_coverage=ring_min_angular_coverage,
        ring_angular_bins=ring_angular_bins,
        ring_max_radial_std=ring_max_radial_std,
        ring_max_defect_ratio=ring_max_defect_ratio,
        ring_min_edge_defect_fraction=ring_min_edge_defect_fraction,
    )
    result = explanation[result_key]
    wbm_tokens = explanation["wbm_tokens"]
    wdm_tokens = explanation["wdm_tokens"]
    matches = explanation["matches"]

    fig = plt.figure(figsize=(15.5, 7.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[3.2, 1.7], hspace=0.18, wspace=0.08)
    image_axes = [fig.add_subplot(gs[0, idx]) for idx in range(4)]
    table_ax = fig.add_subplot(gs[1, :])

    _plot_wbm_defects(image_axes[0], reference, "WBM original")
    _plot_cluster_color_map(
        image_axes[1],
        reference.status_map,
        wbm_tokens,
        title="WBM cluster color map",
        source="wbm",
    )
    _plot_wdm_original(image_axes[2], candidate, reference.status_map, f"WDM original ({map_mode})", map_mode=map_mode)
    _plot_cluster_color_map(
        image_axes[3],
        reference.status_map,
        wdm_tokens,
        title="WDM cluster color map",
        source="wdm",
        count_map=candidate.count_map,
        map_mode=map_mode,
    )
    _plot_match_evidence_table(table_ax, result, matches)

    subtitle = (
        f"count-partial={result.score:.3f}  "
        f"\u25b6 shape={result.mean_shape:.3f} "
        f"pos={result.mean_position:.3f} "
        f"scale={result.mean_scale:.3f} \u25c0  "
        f"tokens={result.matched_tokens}/{result.wbm_tokens}/{result.wdm_tokens}"
    )
    fig.suptitle(f"{title}\n{subtitle}" if title else subtitle, fontsize=11)
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.04, top=0.86)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig, image_axes + [table_ax]


def plot_count_partial_topk(
    reference: GridMaps,
    candidates: List[Tuple[str, GridMaps]],
    title: str = "Top candidates",
    min_area: int = 5,
    top_k: int = 6,
    proposal_mode: str = "cc",
    rotation_tolerance: bool = False,
    min_token_score: float = 0.30,
    score_shape_weight: float = 0.60,
    score_position_weight: float = 0.25,
    score_scale_weight: float = 0.15,
    min_relative_token_area: float = 0.10,
    scale_area_weight: float = 0.50,
    scale_pca_weight: float = 0.50,
    scale_ratio_min: float = 0.20,
    density_sigmas: tuple[float, ...] = (0.8, 1.6, 3.2),
    density_threshold: float = 0.20,
    density_min_raw_points: int = 3,
    density_min_raw_mass: float = 3.0,
    density_merge_iou: float = 0.60,
    density_weight_transform: str = "sqrt",
    ring_min_area: int | None = None,
    ring_edge_r_min: float | None = None,
    ring_band_width: float | None = None,
    ring_min_angular_coverage: float | None = None,
    ring_angular_bins: int | None = None,
    ring_max_radial_std: float | None = None,
    ring_max_defect_ratio: float | None = None,
    ring_min_edge_defect_fraction: float | None = None,
    save_path: str | Path | None = None,
    result_key: str = "result",
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
                min_token_score=min_token_score,
                score_shape_weight=score_shape_weight,
                score_position_weight=score_position_weight,
                score_scale_weight=score_scale_weight,
                min_relative_token_area=min_relative_token_area,
                scale_area_weight=scale_area_weight,
                scale_pca_weight=scale_pca_weight,
                scale_ratio_min=scale_ratio_min,
                density_sigmas=density_sigmas,
                density_threshold=density_threshold,
                density_min_raw_points=density_min_raw_points,
                density_min_raw_mass=density_min_raw_mass,
                density_merge_iou=density_merge_iou,
                density_weight_transform=density_weight_transform,
                ring_min_area=ring_min_area,
                ring_edge_r_min=ring_edge_r_min,
                ring_band_width=ring_band_width,
                ring_min_angular_coverage=ring_min_angular_coverage,
                ring_angular_bins=ring_angular_bins,
                ring_max_radial_std=ring_max_radial_std,
                ring_max_defect_ratio=ring_max_defect_ratio,
                ring_min_edge_defect_fraction=ring_min_edge_defect_fraction,
            ),
        )
        for name, gm in candidates
    ]
    n = max(len(explanations), 1)
    fig, axes = plt.subplots(n, 2, figsize=(9.5, 3.4 * n))
    axes_arr = np.asarray(axes).reshape(n, 2)
    all_axes: List[plt.Axes] = []

    for row, (name, gm, explanation) in enumerate(explanations):
        result = explanation[result_key]
        wbm_tokens = explanation["wbm_tokens"]
        wdm_tokens = explanation["wdm_tokens"]

        ax_ref, ax_cnd = axes_arr[row]
        _plot_cluster_color_map(
            ax_ref,
            reference.status_map,
            wbm_tokens,
            title="Reference WBM clusters",
            source="wbm",
        )
        _plot_cluster_color_map(
            ax_cnd,
            reference.status_map,
            wdm_tokens,
            title=f"{row + 1}. {name}",
            source="wdm",
            count_map=gm.count_map,
        )
        ax_cnd.text(
            0.01,
            -0.08,
            (
                f"score={result.score:.3f}  "
                f"\u25b6 shape={result.mean_shape:.3f} "
                f"pos={result.mean_position:.3f} "
                f"scale={result.mean_scale:.3f} \u25c0  "
                f"tokens={result.matched_tokens}/{result.wbm_tokens}/{result.wdm_tokens}"
            ),
            transform=ax_cnd.transAxes,
            fontsize=8,
            va="top",
        )
        all_axes.extend([ax_ref, ax_cnd])

    fig.suptitle(title, fontsize=12, y=0.98)
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.08, top=0.84, wspace=0.12, hspace=0.35)

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
    im = ax.imshow(image, cmap=COUNT_PARTIAL_CMAP, vmin=1e-6, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    return im


def _plot_wdm_original(ax: plt.Axes, candidate: GridMaps, reference_status: np.ndarray, title: str, map_mode: str) -> None:
    if map_mode == "binary":
        ax.imshow(_wdm_status_image(candidate, reference_status, map_mode=map_mode), interpolation="nearest")
    else:
        image = _masked_count(candidate.count_map, reference_status)
        ax.imshow(image, cmap=COUNT_PARTIAL_CMAP, vmin=1e-6, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def _plot_cluster_color_map(
    ax: plt.Axes,
    reference_status: np.ndarray,
    tokens: List[Dict],
    title: str,
    source: str,
    count_map: np.ndarray | None = None,
    map_mode: str = "count",
) -> None:
    image = _cluster_color_image(reference_status, tokens, source=source, count_map=count_map, map_mode=map_mode)
    ax.imshow(image, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    _draw_token_support_contours(ax, reference_status.shape, tokens)
    _draw_token_ids(ax, tokens)


def _cluster_color_image(
    reference_status: np.ndarray,
    tokens: List[Dict],
    source: str,
    count_map: np.ndarray | None = None,
    map_mode: str = "count",
) -> np.ndarray:
    h, w = reference_status.shape
    image = np.zeros((h, w, 3), dtype=np.float32)
    valid = (reference_status == VALID_NO_DEFECT) | (reference_status == VALID_HAS_DEFECT)
    defects = reference_status == VALID_HAS_DEFECT
    image[valid] = np.array(to_rgb("#7f7f7f"), dtype=np.float32)

    if source == "wbm":
        image[defects] = np.array(to_rgb("#f2f2f2"), dtype=np.float32)
    elif count_map is not None:
        candidate_defects = (count_map > 0) & valid
        image[candidate_defects] = np.array(to_rgb("#f2f2f2"), dtype=np.float32)

    for idx, token in enumerate(tokens):
        color = np.array(to_rgb(TOKEN_COLORS[idx % len(TOKEN_COLORS)]), dtype=np.float32)
        support_color = 0.45 * color
        for r, c in _token_visual_support_pixels(token):
            rr = int(r)
            cc = int(c)
            if 0 <= rr < h and 0 <= cc < w:
                image[rr, cc] = np.clip(0.55 * image[rr, cc] + support_color, 0.0, 1.0)
        for r, c in token.get("pixels", []):
            rr = int(r)
            cc = int(c)
            if 0 <= rr < h and 0 <= cc < w:
                image[rr, cc] = color
    return image


def _token_visual_support_pixels(token: Dict) -> list[tuple[int, int]]:
    if token.get("kde_support_pixels"):
        return [(int(r), int(c)) for r, c in token.get("kde_support_pixels", [])]
    if token.get("ring_contour_pixels"):
        return [(int(r), int(c)) for r, c in token.get("ring_contour_pixels", [])]
    return []


def _draw_token_support_contours(ax: plt.Axes, shape: tuple[int, int], tokens: List[Dict]) -> None:
    h, w = shape
    for idx, token in enumerate(tokens):
        support_pixels = _token_visual_support_pixels(token)
        if not support_pixels:
            continue
        mask = np.zeros((h, w), dtype=np.float32)
        for r, c in support_pixels:
            if 0 <= int(r) < h and 0 <= int(c) < w:
                mask[int(r), int(c)] = 1.0
        if mask.any():
            ax.contour(
                mask,
                levels=[0.5],
                colors=[TOKEN_COLORS[idx % len(TOKEN_COLORS)]],
                linewidths=1.2,
                alpha=0.95,
            )


def _wdm_status_image(candidate: GridMaps, reference_status: np.ndarray, map_mode: str = "count") -> np.ndarray:
    h, w = reference_status.shape
    image = np.zeros((h, w, 3), dtype=np.float32)
    valid = (reference_status == VALID_NO_DEFECT) | (reference_status == VALID_HAS_DEFECT)
    image[valid] = np.array(to_rgb("#7f7f7f"), dtype=np.float32)
    if map_mode == "binary":
        defects = (candidate.binary_map > 0) & valid
    else:
        defects = (candidate.count_map > 0) & valid
    image[defects] = np.array(to_rgb("#f2f2f2"), dtype=np.float32)
    return image


def _plot_match_evidence_table(ax: plt.Axes, result, matches: List[Dict]) -> None:
    rows = []
    for match in matches:
        query = match.get("query_token", {})
        candidate = match.get("candidate_token", {})
        rows.append([
            int(match.get("rank", 0)),
            f"{int(match.get('query_token_id', 0))}->{int(match.get('candidate_token_id', 0))}",
            f"{match.get('score', 0.0):.3f}",
            f"{match.get('shape_sim', 0.0):.3f}",
            f"{match.get('moment_sim', 0.0):.3f}",
            f"{match.get('geometry_sim', 0.0):.3f}",
            f"{match.get('position_affinity', 0.0):.3f}",
            f"{match.get('scale_affinity', 0.0):.3f}",
            f"{match.get('support_area_affinity', 0.0):.3f}",
            f"{match.get('pca_extent_affinity', 0.0):.3f}",
            f"{float(query.get('area', 0.0)):.0f}",
            f"{float(candidate.get('area', 0.0)):.0f}",
        ])
    if not rows:
        rows = [["", "no match", "", "", "", "", "", "", "", "", "", ""]]

    columns = ["rank", "pair", "score", "shape", "moment", "geom", "pos", "scale", "area_s", "pca_s", "q_area", "c_area"]
    ax.axis("off")
    ax.set_title(
        (
            f"match evidence  map score={result.score:.3f}  shape={result.mean_shape:.3f}  "
            f"pos={result.mean_position:.3f}  scale={result.mean_scale:.3f}  "
            f"tokens={result.matched_tokens}/{result.wbm_tokens}/{result.wdm_tokens}"
        ),
        fontsize=10,
        pad=8,
    )
    table = ax.table(cellText=rows, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.12)
    _highlight_columns = {3, 6, 7}  # shape, pos, scale
    for (r, c), cell in table.get_celld().items():
        if c in _highlight_columns:
            if r == 0:
                cell.set_facecolor("#bfdbfe")
            else:
                cell.set_facecolor("#eff6ff")
            cell.set_text_props(weight="bold")
        elif r == 0:
            cell.set_facecolor("#e5e7eb")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#f9fafb" if r % 2 else "#ffffff")


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


def _draw_token_ids(ax: plt.Axes, tokens: List[Dict]) -> None:
    for idx, token in enumerate(tokens):
        color = TOKEN_COLORS[idx % len(TOKEN_COLORS)]
        ax.text(
            token.get("centroid_col", 0.0),
            token.get("centroid_row", 0.0),
            str(idx),
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            bbox={"boxstyle": "circle,pad=0.18", "fc": color, "ec": "white", "lw": 0.5, "alpha": 0.95},
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
