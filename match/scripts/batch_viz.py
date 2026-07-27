from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from .cli_args import derive_classnumber_match_mode


def ensure_mpl() -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")


def save_baseline_figures(args, ref_gm, rows, log_dir: Path) -> None:
    """保存 baseline 对比图：按 IoU (|A∩B|/|A∪B|) 排序 top-K，左侧 WBM、右侧 WDM 按 representation 上色。"""
    ensure_mpl()
    import matplotlib.pyplot as plt
    import numpy as np

    from ..core.models import BACKGROUND, VALID_HAS_DEFECT
    from ..viz.count_partial_visualization import COUNT_PARTIAL_CMAP

    from ..core.similarity import SimilarityResult

    scored: list[tuple[str, "GridMaps", float]] = []
    for fname, res in rows:
        score = res.get("iou")
        grid_maps = res.get("_grid_maps")
        if isinstance(score, SimilarityResult):
            score = score.score
        if isinstance(score, (float, int)) and grid_maps is not None:
            scored.append((Path(fname).stem, grid_maps, float(score)))

    if not scored:
        print("Baseline figures skipped: no valid IoU GridMaps available")
        return

    scored.sort(key=lambda item: item[2], reverse=True)
    top_k = max(getattr(args, "review_top_k", 3), 1)
    top_records = scored[:top_k]

    out_dir = log_dir / "baseline_review"
    out_dir.mkdir(parents=True, exist_ok=True)

    representation = getattr(args, "representation", "count")

    # 根据 representation 选择 cmap
    if representation in ("count", "density"):
        cmap = COUNT_PARTIAL_CMAP
    elif representation == "binary":
        cmap = "gray"
    else:
        cmap = "hot"

    # ── 单图多列：第 1 列 WBM reference，后续列 top-K WDM candidates ──
    n_cols = 1 + len(top_records)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    # 第 1 列：WBM reference
    wbm_status = ref_gm.status_map.copy()
    wbm_img = np.zeros((*wbm_status.shape, 3), dtype=np.uint8)
    wbm_img[wbm_status == BACKGROUND] = [0, 0, 0]
    valid = wbm_status != BACKGROUND
    wbm_img[valid] = [127, 127, 127]
    wbm_img[wbm_status == VALID_HAS_DEFECT] = [255, 255, 255]
    axes[0].imshow(wbm_img, aspect="equal", interpolation="nearest")
    axes[0].set_title("WBM Reference", fontsize=12)
    axes[0].axis("off")

    # 后续列：top-K WDM candidates
    for idx, (name, gm, score) in enumerate(top_records):
        ax = axes[idx + 1]
        rep_map = gm.representation_maps.get(representation)
        if rep_map is not None:
            img = _render_wdm_baseline(rep_map, ref_gm.status_map, cmap)
            ax.imshow(img, aspect="equal", interpolation="nearest")
        ax.set_title(f"#{idx + 1}  {_safe_name(name)}\ncl={score:.4f}", fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    save_path = out_dir / f"baseline_top{len(top_records)}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Baseline figure saved: {save_path}")


def save_classnumber_baseline_figures(args, ref_gm, rows, log_dir: Path) -> None:
    """Save the split-aware IoU baseline: best-IoU classnumber split per KLARF."""
    ensure_mpl()
    import matplotlib.pyplot as plt
    import numpy as np

    from ..core.models import BACKGROUND, VALID_HAS_DEFECT
    from ..viz.count_partial_visualization import COUNT_PARTIAL_CMAP

    # Match the classnumber candidate granularity without using local-match rank_score.
    scored: list[tuple[str, "GridMaps", float, int]] = []
    for fname, res in rows:
        class_result = res.get("_classnumber_result")
        if class_result is None or not class_result.splits:
            continue
        best_iou_split = max(class_result.splits, key=lambda split: split.baseline_iou)
        scored.append((
            Path(fname).stem,
            best_iou_split.grid_maps,
            float(best_iou_split.baseline_iou),
            int(best_iou_split.classnumber),
        ))

    if not scored:
        print("Classnumber baseline figures skipped: no valid split IoU results")
        return

    scored.sort(key=lambda item: item[2], reverse=True)
    top_k = max(getattr(args, "review_top_k", 3), 1)
    top_records = scored[:top_k]

    out_dir = log_dir / "classnumber_baseline_review"
    out_dir.mkdir(parents=True, exist_ok=True)

    representation = getattr(args, "representation", "count")
    if representation in ("count", "density"):
        cmap = COUNT_PARTIAL_CMAP
    elif representation == "binary":
        cmap = "gray"
    else:
        cmap = "hot"

    n_cols = 1 + len(top_records)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    # 第 1 列：WBM reference
    wbm_status = ref_gm.status_map.copy()
    wbm_img = np.zeros((*wbm_status.shape, 3), dtype=np.uint8)
    wbm_img[wbm_status == BACKGROUND] = [0, 0, 0]
    valid = wbm_status != BACKGROUND
    wbm_img[valid] = [127, 127, 127]
    wbm_img[wbm_status == VALID_HAS_DEFECT] = [255, 255, 255]
    axes[0].imshow(wbm_img, aspect="equal", interpolation="nearest")
    axes[0].set_title("WBM Reference", fontsize=12)
    axes[0].axis("off")

    # 后续列：每个 KLARF 的最高 split-IoU，再跨文件取 top-K。
    for idx, (name, gm, score, classnumber) in enumerate(top_records):
        ax = axes[idx + 1]
        rep_map = gm.representation_maps.get(representation)
        if rep_map is not None:
            img = _render_wdm_baseline(rep_map, ref_gm.status_map, cmap)
            ax.imshow(img, aspect="equal", interpolation="nearest")
        ax.set_title(f"#{idx + 1}  {_safe_name(name)} / class {classnumber}\nIoU={score:.4f}", fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    save_path = out_dir / f"baseline_top{len(top_records)}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Classnumber baseline figure saved: {save_path}")


def _mask_background_for_baseline(map_data, status_map: "np.ndarray") -> "np.ndarray":
    """将背景区域置零，WDM 晶圆外与 WBM 晶圆外一致。"""
    import numpy as np

    from ..core.models import BACKGROUND, UNINSPECTED

    masked = map_data.astype(np.float32).copy()
    invalid = (status_map == BACKGROUND) | (status_map == UNINSPECTED)
    masked[invalid] = 0.0
    return masked


def _render_wdm_baseline(rep_map, status_map, cmap_name):
    """构建 RGB 图层：黑色背景、灰色 valid 无缺陷区域、cmap 叠缺陷区域。"""
    import matplotlib.pyplot as _plt
    import numpy as np
    from matplotlib.colors import Normalize
    from ..core.models import BACKGROUND, UNINSPECTED

    h, w = status_map.shape
    img = np.zeros((h, w, 3), dtype=np.float32)
    valid = (status_map != BACKGROUND) & (status_map != UNINSPECTED)
    img[valid] = [0.498, 0.498, 0.498]  # #7f7f7f gray

    raw = rep_map.astype(np.float32)
    defect = (raw > 0) & valid
    if defect.any():
        vmax_val = float(raw.max()) if raw.max() > 0 else 1.0
        norm = Normalize(vmin=1e-6, vmax=vmax_val)
        colored = _plt.get_cmap(cmap_name)(norm(raw))[..., :3]
        img[defect] = colored[defect]

    return img


def save_count_partial_figures(args, ref_gm, rows, log_dir: Path) -> None:
    ensure_mpl()
    import matplotlib.pyplot as plt

    from ..viz.count_partial_visualization import plot_count_partial_steps

    scored: list[tuple[str, "GridMaps", float]] = []
    for fname, res in rows:
        score = res.get("count-partial")
        grid_maps = res.get("_grid_maps")
        if isinstance(score, (float, int)) and grid_maps is not None:
            scored.append((Path(fname).stem, grid_maps, float(score)))

    if not scored:
        print("Count-partial figures skipped: no valid count-partial GridMaps available")
        return

    scored.sort(key=lambda item: item[2], reverse=True)

    _save_count_partial_figures_for_key(
        args, ref_gm, scored, log_dir,
        result_key="result",
        review_subdir="count_partial_review",
        plot_count_partial_steps=plot_count_partial_steps,
        close_all=plt.close,
    )

    _save_count_partial_figures_for_key(
        args, ref_gm, scored, log_dir,
        result_key="result_matched_only",
        review_subdir="count_partial_review_matched_only",
        plot_count_partial_steps=plot_count_partial_steps,
        close_all=plt.close,
    )


def _save_count_partial_figures_for_key(
    args, ref_gm, scored, log_dir: Path, result_key, review_subdir,
    plot_count_partial_steps, close_all,
) -> None:
    out_dir = log_dir / review_subdir
    steps_dir = out_dir / "proposal_steps"
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_dir.mkdir(parents=True, exist_ok=True)

    top_n = max(args.review_top_k, 1)
    top_records = scored[:top_n]

    # ── 横排 top-K 总览图：第 1 列 WBM reference，后续列 top-K WDM candidates ──
    _save_count_partial_topk_horizontal(
        args, ref_gm, top_records, result_key, out_dir, top_n,
    )

    for rank, (name, gm, _) in enumerate(scored[: max(args.step_max, 0)], start=1):
        step_path = steps_dir / f"rank{rank:02d}_{_safe_name(name)}_steps.png"
        plot_count_partial_steps(
            ref_gm,
            gm,
            title=f"Rank {rank}: {name}",
            min_area=args.proposal_min_area,
            top_k=args.proposal_top_k,
            proposal_mode=args.proposal_mode,
            rotation_tolerance=args.proposal_rotation_tolerance,
            min_token_score=args.token_min_score,
            score_shape_weight=args.token_score_shape_weight,
            score_position_weight=args.token_score_position_weight,
            score_scale_weight=args.token_score_scale_weight,
            min_relative_token_area=args.proposal_min_relative_token_area,
            scale_area_weight=args.token_scale_area_weight,
            scale_pca_weight=args.token_scale_pca_weight,
            scale_ratio_min=args.token_scale_ratio_min,
            density_sigmas=tuple(args.density_sigmas),
            density_threshold=args.density_threshold,
            density_min_raw_points=args.density_min_raw_points,
            density_min_raw_mass=args.density_min_raw_mass,
            density_merge_iou=args.density_merge_iou,
            density_weight_transform=args.density_weight_transform,
            ring_min_area=args.ring_min_area,
            ring_edge_r_min=args.ring_edge_r_min,
            ring_band_width=args.ring_band_width,
            ring_min_angular_coverage=args.ring_min_angular_coverage,
            ring_angular_bins=args.ring_angular_bins,
            ring_max_radial_std=args.ring_max_radial_std,
            ring_max_defect_ratio=args.ring_max_defect_ratio,
            ring_min_edge_defect_fraction=args.ring_min_edge_defect_fraction,
            save_path=step_path,
            result_key=result_key,
        )
        close_all("all")
        print(f"Count-partial step figure saved [{result_key}]: {step_path}")


def _save_count_partial_topk_horizontal(
    args, ref_gm, top_records, result_key: str, out_dir: Path, top_n: int,
) -> None:
    """横排 top-K count-partial 总览图：左 WBM reference，右依次为 top-K WDM candidates。"""
    import matplotlib.pyplot as plt
    import numpy as np

    from ..core.models import BACKGROUND, VALID_HAS_DEFECT
    from ..viz.count_partial_visualization import COUNT_PARTIAL_CMAP

    n_cols = 1 + len(top_records)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    representation = getattr(args, "representation", "count")
    if representation in ("count", "density"):
        cmap = COUNT_PARTIAL_CMAP
    elif representation == "binary":
        cmap = "gray"
    else:
        cmap = "hot"

    # 第 1 列: WBM reference
    wbm_status = ref_gm.status_map.copy()
    wbm_img = np.zeros((*wbm_status.shape, 3), dtype=np.uint8)
    wbm_img[wbm_status == BACKGROUND] = [0, 0, 0]
    valid = wbm_status != BACKGROUND
    wbm_img[valid] = [127, 127, 127]
    wbm_img[wbm_status == VALID_HAS_DEFECT] = [255, 255, 255]
    axes[0].imshow(wbm_img, aspect="equal", interpolation="nearest")
    axes[0].set_title("WBM Reference", fontsize=12)
    axes[0].axis("off")

    # 后续列: top-K WDM candidates
    for idx, (name, gm, _score) in enumerate(top_records):
        ax = axes[idx + 1]
        rep_map = gm.representation_maps.get(representation)
        if rep_map is not None:
            img = _render_wdm_baseline(rep_map, ref_gm.status_map, cmap)
            ax.imshow(img, aspect="equal", interpolation="nearest")
        ax.set_title(f"Rank #{idx + 1}  {_safe_name(name)}", fontsize=11)
        ax.axis("off")

    out_key = result_key.replace("result", "count_partial")
    save_path = out_dir / f"top{top_n}_{_safe_name(out_key)}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Count-partial TopK figure saved [{result_key}]: {save_path}")


def _save_classnumber_topk_horizontal(
    ref_gm, top_records, rank_by: str, out_dir: Path, save_path: Path,
) -> None:
    """横排 top-K classnumber split 总览图：左 WBM reference，右依次为 top-K split WDM。"""
    import matplotlib.pyplot as plt
    import numpy as np

    from ..core.classnumber_matching import split_score
    from ..core.models import BACKGROUND, VALID_HAS_DEFECT
    from ..viz.classnumber_visualization import CLASSNUMBER_CMAP

    n_cols = 1 + len(top_records)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    # 第 1 列: WBM reference
    wbm_status = ref_gm.status_map.copy()
    wbm_img = np.zeros((*wbm_status.shape, 3), dtype=np.uint8)
    wbm_img[wbm_status == BACKGROUND] = [0, 0, 0]
    valid = wbm_status != BACKGROUND
    wbm_img[valid] = [127, 127, 127]
    wbm_img[wbm_status == VALID_HAS_DEFECT] = [255, 255, 255]
    axes[0].imshow(wbm_img, aspect="equal", interpolation="nearest")
    axes[0].set_title("WBM Reference", fontsize=12)
    axes[0].axis("off")

    # 后续列: top-K classnumber split WDM candidates
    for idx, rec in enumerate(top_records):
        ax = axes[idx + 1]
        rep_map = rec["grid_maps"].representation_maps.get(rank_by)
        if rep_map is not None:
            img = _render_wdm_baseline(rep_map, ref_gm.status_map, CLASSNUMBER_CMAP)
            ax.imshow(img, aspect="equal", interpolation="nearest")
        score = split_score(rec["split"], rank_by)
        ax.set_title(
            f"Rank #{idx + 1}  {_safe_name(rec['file'])} / class {rec['classnumber']}\n{rank_by}={score:.3f}",
            fontsize=10,
        )
        ax.axis("off")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_classnumber_figures(args, ref_gm, rows, log_dir: Path) -> None:
    ensure_mpl()
    import matplotlib.pyplot as plt

    from ..core.classnumber_matching import split_score
    from ..viz.classnumber_visualization import (
        plot_classnumber_splits,
        plot_classnumber_step,
    )

    review_name = _classnumber_review_name(args)
    out_dir = log_dir / review_name
    steps_dir = out_dir / "topk_steps"
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_dir.mkdir(parents=True, exist_ok=True)

    split_records = []
    for fname, res in rows:
        grid_maps = res.get("_grid_maps")
        class_result = res.get("_classnumber_result")
        if grid_maps is None or class_result is None or not class_result.splits:
            continue
        file_stem = Path(fname).stem
        for split in class_result.splits:
            split_records.append({
                "file": file_stem,
                "file_name": fname,
                "classnumber": split.classnumber,
                "split": split,
                "grid_maps": split.grid_maps,
                "full_grid_maps": grid_maps,
                "class_result": class_result,
                "partial": split.partial,
                "binary": split.binary,
                "rank_score": split.rank_score,
                "rank_mode": class_result.match_mode,
            })

    if not split_records:
        print("Classnumber figures skipped: no valid classnumber split results available")
        return

    rank_by = derive_classnumber_match_mode(args.representation)
    split_records.sort(key=lambda item: split_score(item["split"], rank_by), reverse=True)
    top_k = max(args.review_top_k, 1)
    top_records = split_records[:top_k]
    _save_topk_classnumber_split_maps(
        ref_gm=ref_gm,
        top_records=top_records,
        out_dir=out_dir,
        plot_classnumber_splits=plot_classnumber_splits,
        close_all=plt.close,
    )

    ranking_path = out_dir / "classnumber_topk.tsv"
    with open(ranking_path, "w") as file:
        file.write(
            "rank\tfile\tclassnumber\trank_by\trank_score\t"
            "count_partial\tshape\tposition\tscale\ttype\ttokens\t"
            "binary\tbinary_shape\tbinary_position\tbinary_scale\tbinary_type\tbinary_tokens\n"
        )
        for rank, record in enumerate(split_records, start=1):
            partial = record["partial"]
            binary = record["binary"]
            if partial is None:
                count_cells = ["", "", "", "", "", ""]
            else:
                count_cells = [
                    f"{partial.score:.6f}",
                    f"{partial.mean_shape:.6f}",
                    f"{partial.mean_position:.6f}",
                    f"{partial.mean_scale:.6f}",
                    f"{partial.mean_type:.6f}",
                    f"{partial.matched_tokens}/{partial.wbm_tokens}/{partial.wdm_tokens}",
                ]
            if binary is None:
                binary_cells = ["", "", "", "", "", ""]
            else:
                binary_cells = [
                    f"{binary.score:.6f}",
                    f"{binary.mean_shape:.6f}",
                    f"{binary.mean_position:.6f}",
                    f"{binary.mean_scale:.6f}",
                    f"{binary.mean_type:.6f}",
                    f"{binary.matched_tokens}/{binary.wbm_tokens}/{binary.wdm_tokens}",
                ]
            file.write(
                f"{rank}\t{record['file']}\t{record['classnumber']}\t"
                f"{rank_by}\t{split_score(record['split'], rank_by):.6f}\t"
                + "\t".join(count_cells + binary_cells)
                + "\n"
            )
    print(f"Classnumber ranking saved: {ranking_path}")

    topk_path = out_dir / f"classnumber_top{min(top_k, len(split_records))}.png"
    _save_classnumber_topk_horizontal(
        ref_gm, top_records, rank_by, out_dir, topk_path,
    )
    print(f"Classnumber TopK figure saved: {topk_path}")

    for rank, record in enumerate(split_records[:top_k], start=1):
        step_path = steps_dir / f"rank{rank:02d}_{_safe_name(record['file'])}_class{record['classnumber']}_steps.png"
        plot_classnumber_step(
            ref_gm,
            record["grid_maps"],
            score_mode=rank_by,
            title=f"Rank {rank}: {record['file']} class {record['classnumber']} ({rank_by})",
            min_area=args.proposal_min_area,
            top_k=args.proposal_top_k,
            proposal_mode=args.proposal_mode,
            rotation_tolerance=args.proposal_rotation_tolerance,
            min_token_score=args.token_min_score,
            score_shape_weight=args.token_score_shape_weight,
            score_position_weight=args.token_score_position_weight,
            score_scale_weight=args.token_score_scale_weight,
            min_relative_token_area=args.proposal_min_relative_token_area,
            scale_area_weight=args.token_scale_area_weight,
            scale_pca_weight=args.token_scale_pca_weight,
            scale_ratio_min=args.token_scale_ratio_min,
            density_sigmas=tuple(args.density_sigmas),
            density_threshold=args.density_threshold,
            density_min_raw_points=args.density_min_raw_points,
            density_min_raw_mass=args.density_min_raw_mass,
            density_merge_iou=args.density_merge_iou,
            density_weight_transform=args.density_weight_transform,
            ring_min_area=args.ring_min_area,
            ring_edge_r_min=args.ring_edge_r_min,
            ring_band_width=args.ring_band_width,
            ring_min_angular_coverage=args.ring_min_angular_coverage,
            ring_angular_bins=args.ring_angular_bins,
            ring_max_radial_std=args.ring_max_radial_std,
            ring_max_defect_ratio=args.ring_max_defect_ratio,
            ring_min_edge_defect_fraction=args.ring_min_edge_defect_fraction,
            save_path=step_path,
        )
        plt.close("all")
        print(f"Classnumber TopK step figure saved: {step_path}")

    # --- matched-only classnumber step figures ---
    mo_review_name = review_name.replace("classnumber_review", "classnumber_review_matched_only")
    mo_out_dir = log_dir / mo_review_name
    mo_steps_dir = mo_out_dir / "topk_steps"
    mo_steps_dir.mkdir(parents=True, exist_ok=True)

    mo_split_records = []
    for fname, res in rows:
        class_result = res.get("_classnumber_result")
        if class_result is None or not class_result.splits:
            continue
        file_stem = Path(fname).stem
        for split in class_result.splits:
            mo_score = _mo_split_rank_score(split, rank_by)
            mo_split_records.append({
                "file": file_stem,
                "file_name": fname,
                "classnumber": split.classnumber,
                "split": split,
                "grid_maps": split.grid_maps,
                "mo_score": mo_score,
            })

    if mo_split_records:
        mo_split_records.sort(key=lambda item: item["mo_score"], reverse=True)
        mo_top_records = mo_split_records[:top_k]

        # ── 横排 top-K 总览图 ──
        mo_topk_path = mo_out_dir / f"classnumber_mo_top{min(top_k, len(mo_split_records))}.png"
        _save_classnumber_topk_horizontal(
            ref_gm, mo_top_records, rank_by, mo_out_dir, mo_topk_path,
        )

        for rank, record in enumerate(mo_top_records, start=1):
            step_path = mo_steps_dir / f"rank{rank:02d}_{_safe_name(record['file'])}_class{record['classnumber']}_mo_steps.png"
            plot_classnumber_step(
                ref_gm,
                record["grid_maps"],
                score_mode=rank_by,
                title=f"Rank {rank}: {record['file']} class {record['classnumber']} ({rank_by}, matched-only)",
                min_area=args.proposal_min_area,
                top_k=args.proposal_top_k,
                proposal_mode=args.proposal_mode,
                rotation_tolerance=args.proposal_rotation_tolerance,
                min_token_score=args.token_min_score,
                score_shape_weight=args.token_score_shape_weight,
                score_position_weight=args.token_score_position_weight,
                score_scale_weight=args.token_score_scale_weight,
                min_relative_token_area=args.proposal_min_relative_token_area,
                scale_area_weight=args.token_scale_area_weight,
                scale_pca_weight=args.token_scale_pca_weight,
                scale_ratio_min=args.token_scale_ratio_min,
                density_sigmas=tuple(args.density_sigmas),
                density_threshold=args.density_threshold,
                density_min_raw_points=args.density_min_raw_points,
                density_min_raw_mass=args.density_min_raw_mass,
                density_merge_iou=args.density_merge_iou,
                density_weight_transform=args.density_weight_transform,
                ring_min_area=args.ring_min_area,
                ring_edge_r_min=args.ring_edge_r_min,
                ring_band_width=args.ring_band_width,
                ring_min_angular_coverage=args.ring_min_angular_coverage,
                ring_angular_bins=args.ring_angular_bins,
                ring_max_radial_std=args.ring_max_radial_std,
                ring_max_defect_ratio=args.ring_max_defect_ratio,
                ring_min_edge_defect_fraction=args.ring_min_edge_defect_fraction,
                save_path=step_path,
                result_key="result_matched_only",
            )
            plt.close("all")
            print(f"Classnumber matched-only step figure saved: {step_path}")


def _save_topk_classnumber_split_maps(
    ref_gm,
    top_records,
    out_dir: Path,
    plot_classnumber_splits,
    close_all,
) -> None:
    saved_files = set()
    for record in top_records:
        file_key = record["file"]
        if file_key in saved_files:
            continue
        saved_files.add(file_key)
        path = out_dir / f"{_safe_name(file_key)}_classnumber_splits.png"
        plot_classnumber_splits(
            ref_gm,
            record["full_grid_maps"],
            record["class_result"],
            title=f"{record['file_name']} classnumber split matching",
            save_path=path,
        )
        close_all("all")
        print(f"Classnumber split figure saved: {path}")


def _classnumber_review_name(args) -> str:
    mode = derive_classnumber_match_mode(getattr(args, "representation", "count"))
    return f"classnumber_review_{_safe_name(mode)}"


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)[:80]


def save_wdm_raw_figures(args, rows: list, log_dir: Path) -> None:
    """保存 count-partial 模式 top-K 的 WDM 原图：所有缺陷绘制，CLASSNUMBER=0 红色，!=0 蓝色。"""
    ensure_mpl()
    import matplotlib.pyplot as plt
    from ..viz.klarfkit import WaferMap

    scored: list[tuple[str, str, float]] = []
    for fname, res in rows:
        score = res.get("count-partial")
        grid_maps = res.get("_grid_maps")
        if not (isinstance(score, (float, int)) and grid_maps is not None):
            continue
        klarf_path = grid_maps.metadata.get("klarf_path", "")
        if klarf_path:
            scored.append((fname, klarf_path, float(score)))

    if not scored:
        print("WDM raw figures skipped: no valid klarf paths found")
        return

    scored.sort(key=lambda item: item[2], reverse=True)
    top_k = max(getattr(args, "review_top_k", 3), 1)
    top_records = scored[:top_k]

    out_dir = log_dir / "wdm_raw_review"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 横排总览图：所有 top-K 在一个 figure 中 ──
    n_cols = len(top_records)
    if n_cols == 0:
        return
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 7))
    if n_cols == 1:
        axes = [axes]

    for ax_idx, (rank, (fname, klarf_path, score)) in enumerate(
        enumerate(top_records, start=1)
    ):
        ax = axes[ax_idx]
        try:
            wm = WaferMap.read_klarf(klarf_path)
        except Exception as e:
            print(f"WDM raw figure skipped [{fname}]: {e}")
            ax.axis("off")
            continue

        df = wm.defect_list
        cls_col = None
        for col in df.columns:
            if col.upper() == "CLASSNUMBER":
                cls_col = col
                break

        _draw_wafer_base(wm, ax)

        x = df["_XACTUAL"].to_numpy()
        y = df["_YACTUAL"].to_numpy()

        if cls_col is not None:
            class_vals = df[cls_col].to_numpy()
            mask_zero = class_vals == 0
            mask_nonzero = ~mask_zero
            if mask_zero.any():
                ax.scatter(x[mask_zero], y[mask_zero], color="red", s=8, zorder=10, label="classnumber=0")
            if mask_nonzero.any():
                ax.scatter(x[mask_nonzero], y[mask_nonzero], color="blue", s=8, zorder=10, label="classnumber!=0")
            ax.legend(loc="lower left", fontsize=8)
        else:
            ax.scatter(x, y, color="red", s=8, zorder=10)

        file_stem = Path(fname).stem
        ax.set_title(f"#{rank}  {file_stem}\ncount-partial={score:.4f}", fontsize=11)
        ax.set_aspect("equal")
        ax.axis("off")

    save_path = out_dir / f"wdm_raw_top{len(top_records)}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"WDM raw figure saved: {save_path}")


def save_classnumber_wdm_raw_figures(args, rows: list, log_dir: Path) -> None:
    """保存 classnumber 模式 top-K split 的 WDM 原图：仅绘制最佳 classnumber 的缺陷，蓝色。"""
    ensure_mpl()
    import matplotlib.pyplot as plt
    from ..core.classnumber_matching import split_score
    from ..viz.klarfkit import WaferMap

    split_records = []
    for fname, res in rows:
        grid_maps = res.get("_grid_maps")
        class_result = res.get("_classnumber_result")
        if grid_maps is None or class_result is None or not class_result.splits:
            continue
        klarf_path = grid_maps.metadata.get("klarf_path", "")
        if not klarf_path:
            continue
        for split in class_result.splits:
            split_records.append({
                "file_name": fname,
                "klarf_path": klarf_path,
                "classnumber": split.classnumber,
                "split": split,
                "rank_score": split.rank_score,
                "rank_mode": class_result.match_mode,
            })

    if not split_records:
        print("Classnumber WDM raw figures skipped: no valid splits")
        return

    rank_by = getattr(args, "representation", "count")
    if rank_by not in ("count", "binary"):
        rank_by = "count"
    split_records.sort(key=lambda item: split_score(item["split"], rank_by), reverse=True)
    top_k = max(getattr(args, "review_top_k", 3), 1)
    top_records = split_records[:top_k]

    out_dir = log_dir / "wdm_raw_classnumber_review"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 横排总览图：所有 top-K 在一个 figure 中 ──
    n_cols = len(top_records)
    if n_cols == 0:
        return
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 7))
    if n_cols == 1:
        axes = [axes]

    for ax_idx, (rank, record) in enumerate(enumerate(top_records, start=1)):
        ax = axes[ax_idx]
        try:
            wm = WaferMap.read_klarf(record["klarf_path"])
        except Exception as e:
            print(f"Classnumber WDM raw figure skipped [{record['file_name']}]: {e}")
            ax.axis("off")
            continue

        df = wm.defect_list
        target_class = record["classnumber"]

        cls_col = None
        for col in df.columns:
            if col.upper() == "CLASSNUMBER":
                cls_col = col
                break

        if cls_col is not None:
            mask = df[cls_col].to_numpy() == target_class
            df_filtered = df[mask]
        else:
            df_filtered = df

        if not df_filtered.empty:
            wm_tmp = WaferMap(defect_record=df_filtered.copy(), die_pitch=wm.die_pitch,
                             sample_center_location=wm.center_location, sample_size=wm.sample_size)
            _draw_wafer_base(wm_tmp, ax)
            ax.scatter(
                wm_tmp.defect_list["_XACTUAL"].to_numpy(),
                wm_tmp.defect_list["_YACTUAL"].to_numpy(),
                color="blue", s=8, zorder=10,
            )
        else:
            _draw_wafer_base(wm, ax)

        file_stem = Path(record["file_name"]).stem
        ax.set_title(f"#{rank}  {file_stem}  class={target_class}\n{rank_by}-score={split_score(record['split'], rank_by):.4f}", fontsize=11)
        ax.set_aspect("equal")
        ax.axis("off")

    save_path = out_dir / f"wdm_raw_classnumber_top{len(top_records)}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Classnumber WDM raw figure saved: {save_path}")


def _draw_wafer_base(wm, ax) -> None:
    """绘制 wafer 外圆轮廓和 die 网格线。"""
    import numpy as np

    radius = wm.sample_size / 2
    theta = np.linspace(0, 2 * np.pi, 200)
    wafer_x = radius * np.cos(theta)
    wafer_y = radius * np.sin(theta)
    ax.plot(wafer_x, wafer_y, color="black", linewidth=0.5)

    die_pitch_x, die_pitch_y = wm.die_pitch
    cx, cy = wm.center_location
    n_dice_x = int(wm.sample_size // die_pitch_x) + 2
    n_dice_y = int(wm.sample_size // die_pitch_y) + 2

    radius_sq = radius ** 2
    for i in range(-n_dice_x, n_dice_x + 1):
        line_x = -cx + i * die_pitch_x
        h_sq = radius_sq - line_x ** 2
        if h_sq > 0:
            h = np.sqrt(h_sq)
            ax.vlines(line_x, -h, h, color="gray", alpha=0.15, linewidth=0.5)

    for i in range(-n_dice_y, n_dice_y + 1):
        line_y = -cy + i * die_pitch_y
        w_sq = radius_sq - line_y ** 2
        if w_sq > 0:
            w = np.sqrt(w_sq)
            ax.hlines(line_y, -w, w, color="gray", alpha=0.15, linewidth=0.5)


def _mo_split_rank_score(split, match_mode: str) -> float:
    if match_mode == "count":
        if split.partial_matched_only is not None:
            return float(split.partial_matched_only.score)
        return float(split.partial.score) if split.partial is not None else float("-inf")
    if match_mode == "binary":
        if split.binary_matched_only is not None:
            return float(split.binary_matched_only.score)
        return float(split.binary.score) if split.binary is not None else float("-inf")
    return float("-inf")
