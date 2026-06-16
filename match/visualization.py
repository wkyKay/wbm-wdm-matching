# Visualization utilities for WBM（reference）vs WDM（candidate）comparison.
#
# 核心设计：同样的 colormap + 同样的 vmin/vmax + status 统一背景 → 可比。
# 支持 binary、count、density 三种视图。
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from .models import GridMaps, BACKGROUND, UNINSPECTED

# ── 状态掩码（掩掉 wafer 外的区域）────────────────────────────────


def _mask_background(
    map_data: np.ndarray,
    status_map: np.ndarray,
    bg_value: float = 0.0,
) -> np.ndarray:
    """将 background / uninspected 区域置为 bg_value，确保两张图空白区一致。"""
    masked = map_data.astype(np.float32).copy()
    invalid = (status_map == BACKGROUND) | (status_map == UNINSPECTED)
    masked[invalid] = bg_value
    return masked


def _unified_range(ref: np.ndarray, cnd: np.ndarray) -> Tuple[float, float]:
    """取两张图非零区域共同的 vmin/vmax，使 colormap 映射一致。"""
    combined = np.concatenate([ref.ravel(), cnd.ravel()])
    nonzero = combined[combined > 0]
    if len(nonzero) == 0:
        return 0.0, 1.0
    return float(combined.min()), float(nonzero.max())


# ── 单图绘图 ─────────────────────────────────────────────────────


def plot_single(
    grid_maps: GridMaps,
    title: str = "",
    representation: str = "density",
    cmap: str = "hot",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """绘制单个 GridMaps。

    Parameters
    ----------
    grid_maps : 要绘制的 GridMaps。
    title : 图标题。
    representation : "binary" | "count" | "density"。
    cmap : colormap 名称，binary 默认用 'gray'。
    ax : 可选的 matplotlib Axes。
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    rep_map = grid_maps.representation_maps.get(representation)
    if rep_map is None:
        raise ValueError(f"Unknown representation: {representation}")

    # binary 视图默认用灰度
    if representation == "binary" and cmap == "hot":
        cmap = "gray"

    masked = _mask_background(rep_map, grid_maps.status_map)
    vmin = 0.0
    vmax = float(masked.max()) if masked.max() > 0 else 1.0

    ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title)
    ax.axis("off")
    return ax


# ── 并排对比 ─────────────────────────────────────────────────────


def plot_comparison(
    reference: GridMaps,
    candidate: GridMaps,
    representation: str = "density",
    cmap: str = "hot",
    ref_label: str = "Reference WBM",
    cnd_label: str = "Candidate WDM",
    figsize: Tuple[float, float] = (12, 5),
    save_path: str | Path | None = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """并排对比 reference 与 candidate，使用统一的 colormap 范围。

    Parameters
    ----------
    reference : WBM reference GridMaps。
    candidate : KLARF candidate GridMaps。
    representation : 使用哪个 map 做比较（"binary" | "count" | "density"）。
    cmap : colormap，binary 默认用 'gray'。
    ref_label / cnd_label : 左右子图标题。
    figsize : 图尺寸。
    save_path : 如果不为 None，保存到此路径。
    """
    rep_map_key = representation
    ref_map = reference.representation_maps.get(rep_map_key)
    cnd_map = candidate.representation_maps.get(rep_map_key)
    if ref_map is None or cnd_map is None:
        raise ValueError(f"Representation {representation!r} not found in reference or candidate maps")

    # 统一 colormap 范围
    ref_masked = _mask_background(ref_map, reference.status_map)
    cnd_masked = _mask_background(cnd_map, candidate.status_map)
    vmin, vmax = _unified_range(ref_masked, cnd_masked)

    if representation == "binary":
        cmap = "gray"
        vmin, vmax = 0.0, 1.0

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize)

    ax_l.imshow(ref_masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax_l.set_title(ref_label)
    ax_l.axis("off")

    im = ax_r.imshow(cnd_masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax_r.set_title(cnd_label)
    ax_r.axis("off")

    # colorbar（基于 candidate 的图，因为范围统一所以两图可用同一个 bar）
    plt.colorbar(im, ax=[ax_l, ax_r], fraction=0.046, pad=0.04, label=representation)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, (ax_l, ax_r)


# ── 叠加差异图 ───────────────────────────────────────────────────


def plot_overlay(
    reference: GridMaps,
    candidate: GridMaps,
    representation: str = "binary",
    ref_label: str = "Reference WBM",
    cnd_label: str = "Candidate WDM",
    figsize: Tuple[float, float] = (6, 6),
    save_path: str | Path | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """差值叠加：ref only=蓝, candidate only=红, 双方匹配=白。

    色彩含义：
        黑色  — 双方均无缺陷
        蓝    — 仅 reference 有（漏检）
        红    — 仅 candidate 有（多报 / leakage）
        白/黄 — 双方匹配
    """
    ref_map = reference.representation_maps.get(representation)
    cnd_map = candidate.representation_maps.get(representation)
    if ref_map is None or cnd_map is None:
        raise ValueError(f"Representation {representation!r} not found in maps")

    # 用 reference 的 status 定义"有意义区域"，统一背景
    ref_masked = _mask_background(ref_map, reference.status_map)
    cnd_masked = _mask_background(cnd_map, reference.status_map)

    ref_bin = ref_masked > 0
    cnd_bin = cnd_masked > 0

    overlay = np.zeros((*ref_bin.shape, 3), dtype=np.uint8)
    overlay[ref_bin & ~cnd_bin] = [0, 0, 255]      # 蓝 = ref only
    overlay[~ref_bin & cnd_bin] = [255, 0, 0]      # 红 = cnd only（leakage）
    overlay[ref_bin & cnd_bin] = [255, 255, 255]   # 白 = 匹配

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(overlay, aspect="equal")
    ax.set_title(f"{ref_label}  vs  {cnd_label}\nBlue=ref only  Red=cand only  White=match")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


# ── 多 representation 对比面板 ────────────────────────────────────


def plot_representation_panel(
    grid_maps: GridMaps,
    title: str = "",
    representations: Tuple[str, ...] = ("binary", "count", "density"),
    cmaps: Tuple[str, ...] = ("gray", "hot", "hot"),
    figsize: Tuple[float, float] = (16, 5),
    save_path: str | Path | None = None,
) -> Tuple[plt.Figure, list[plt.Axes]]:
    """在一个图中展示多个 representation。

    Parameters
    ----------
    grid_maps : 要展示的 GridMaps。
    title : 总标题。
    representations : 要展示的 representation 名列表。
    cmaps : 对应的 colormap 列表。
    figsize : 图尺寸。
    save_path : 如果不为 None，保存到此路径。
    """
    n = len(representations)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, rep, cm in zip(axes, representations, cmaps):
        plot_single(grid_maps, title=rep.capitalize(), representation=rep, cmap=cm, ax=ax)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, axes
