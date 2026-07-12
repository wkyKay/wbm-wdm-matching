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
    """保存 baseline 对比图：按 coverage-leakage 排序 top-K，左侧 WBM、右侧 WDM 按 representation 上色。"""
    ensure_mpl()
    import matplotlib.pyplot as plt
    import numpy as np

    from ..core.models import BACKGROUND, VALID_HAS_DEFECT
    from ..viz.count_partial_visualization import COUNT_PARTIAL_CMAP

    scored: list[tuple[str, "GridMaps", float]] = []
    for fname, res in rows:
        score = res.get("coverage-leakage")
        grid_maps = res.get("_grid_maps")
        if isinstance(score, (float, int)) and grid_maps is not None:
            scored.append((Path(fname).stem, grid_maps, float(score)))

    if not scored:
        print("Baseline figures skipped: no valid coverage-leakage GridMaps available")
        return

    scored.sort(key=lambda item: item[2], reverse=True)
    top_k = max(getattr(args, "count_partial_review_top_k", 3), 1)
    top_records = scored[:top_k]

    out_dir = log_dir / "baseline_review"
    out_dir.mkdir(parents=True, exist_ok=True)

    representation = getattr(args, "representation", "count")

    # 根据 representation 选择 cmap
    if representation == "count":
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
            masked = _mask_background_for_baseline(rep_map, gm.status_map)
            if representation == "binary":
                ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="equal", interpolation="nearest")
            else:
                vmax_val = float(masked.max()) if masked.max() > 0 else 1.0
                ax.imshow(masked, cmap=cmap, vmin=1e-6, vmax=vmax_val, aspect="equal", interpolation="nearest")
        ax.set_title(f"#{idx + 1}  {_safe_name(name)}\ncl={score:.4f}", fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    save_path = out_dir / f"baseline_top{len(top_records)}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Baseline figure saved: {save_path}")


def save_classnumber_baseline_figures(args, ref_gm, rows, log_dir: Path) -> None:
    """保存 classnumber 模式 baseline 对比图：用各 KLARF 的 best split map 替代 sum map，按 rank_score 排名 top-K。"""
    ensure_mpl()
    import matplotlib.pyplot as plt
    import numpy as np

    from ..core.models import BACKGROUND, VALID_HAS_DEFECT
    from ..viz.count_partial_visualization import COUNT_PARTIAL_CMAP

    # 收集每个 KLARF 的最佳 classnumber split
    scored: list[tuple[str, "GridMaps", float]] = []
    for fname, res in rows:
        class_result = res.get("_classnumber_result")
        if class_result is None or class_result.best is None:
            continue
        file_stem = Path(fname).stem
        scored.append((file_stem, class_result.best.grid_maps, float(class_result.best.rank_score)))

    if not scored:
        print("Classnumber baseline figures skipped: no valid classnumber split results")
        return

    scored.sort(key=lambda item: item[2], reverse=True)
    top_k = max(getattr(args, "count_partial_review_top_k", 3), 1)
    top_records = scored[:top_k]

    out_dir = log_dir / "baseline_review"
    out_dir.mkdir(parents=True, exist_ok=True)

    representation = getattr(args, "representation", "count")
    if representation == "count":
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

    # 后续列：top-K best split maps
    for idx, (name, gm, score) in enumerate(top_records):
        ax = axes[idx + 1]
        rep_map = gm.representation_maps.get(representation)
        if rep_map is not None:
            masked = _mask_background_for_baseline(rep_map, gm.status_map)
            if representation == "binary":
                ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="equal", interpolation="nearest")
            else:
                vmax_val = float(masked.max()) if masked.max() > 0 else 1.0
                ax.imshow(masked, cmap=cmap, vmin=1e-6, vmax=vmax_val, aspect="equal", interpolation="nearest")
        ax.set_title(f"#{idx + 1}  {_safe_name(name)}\nsplit rank={score:.4f}", fontsize=11)
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


def save_count_partial_figures(args, ref_gm, rows, log_dir: Path) -> None:
    ensure_mpl()
    import matplotlib.pyplot as plt

    from ..viz.count_partial_visualization import plot_count_partial_steps, plot_count_partial_topk

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
        plot_count_partial_topk=plot_count_partial_topk,
        plot_count_partial_steps=plot_count_partial_steps,
        close_all=plt.close,
    )

    _save_count_partial_figures_for_key(
        args, ref_gm, scored, log_dir,
        result_key="result_matched_only",
        review_subdir="count_partial_review_matched_only",
        plot_count_partial_topk=plot_count_partial_topk,
        plot_count_partial_steps=plot_count_partial_steps,
        close_all=plt.close,
    )


def _save_count_partial_figures_for_key(
    args, ref_gm, scored, log_dir: Path, result_key, review_subdir,
    plot_count_partial_topk, plot_count_partial_steps, close_all,
) -> None:
    out_dir = log_dir / review_subdir
    steps_dir = out_dir / "proposal_steps"
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_dir.mkdir(parents=True, exist_ok=True)

    top_n = max(args.count_partial_review_top_k, 1)
    top_records = scored[:top_n]
    topk_path = out_dir / f"top{len(top_records)}_count_partial.png"
    plot_count_partial_topk(
        ref_gm,
        [(name, gm) for name, gm, _ in top_records],
        title="Count-map partial matching top candidates",
        min_area=args.count_partial_min_area,
        top_k=args.count_partial_top_k_proposals,
        proposal_mode=args.count_partial_proposal_mode,
        rotation_tolerance=args.count_partial_rotation_tolerance,
        min_token_score=args.count_partial_min_token_score,
        score_shape_weight=args.count_partial_score_shape_weight,
        score_position_weight=args.count_partial_score_position_weight,
        score_scale_weight=args.count_partial_score_scale_weight,
        min_relative_token_area=args.count_partial_min_relative_token_area,
        scale_area_weight=args.count_partial_scale_area_weight,
        scale_pca_weight=args.count_partial_scale_pca_weight,
        save_path=topk_path,
        result_key=result_key,
    )
    close_all("all")
    print(f"Count-partial TopK figure saved [{result_key}]: {topk_path}")

    for rank, (name, gm, _) in enumerate(scored[: max(args.count_partial_step_max, 0)], start=1):
        step_path = steps_dir / f"rank{rank:02d}_{_safe_name(name)}_steps.png"
        plot_count_partial_steps(
            ref_gm,
            gm,
            title=f"Rank {rank}: {name}",
            min_area=args.count_partial_min_area,
            top_k=args.count_partial_top_k_proposals,
            proposal_mode=args.count_partial_proposal_mode,
            rotation_tolerance=args.count_partial_rotation_tolerance,
            min_token_score=args.count_partial_min_token_score,
            score_shape_weight=args.count_partial_score_shape_weight,
            score_position_weight=args.count_partial_score_position_weight,
            score_scale_weight=args.count_partial_score_scale_weight,
            min_relative_token_area=args.count_partial_min_relative_token_area,
            scale_area_weight=args.count_partial_scale_area_weight,
            scale_pca_weight=args.count_partial_scale_pca_weight,
            save_path=step_path,
            result_key=result_key,
        )
        close_all("all")
        print(f"Count-partial step figure saved [{result_key}]: {step_path}")


def save_classnumber_figures(args, ref_gm, rows, log_dir: Path) -> None:
    ensure_mpl()
    import matplotlib.pyplot as plt

    from ..core.classnumber_matching import split_score
    from ..viz.classnumber_visualization import (
        plot_classnumber_splits,
        plot_classnumber_step,
        plot_classnumber_topk_splits,
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
    top_k = max(args.count_partial_review_top_k, 1)
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
    plot_classnumber_topk_splits(
        ref_gm,
        split_records,
        top_k=top_k,
        score_mode=rank_by,
        title=f"Global classnumber split top candidates ({rank_by})",
        save_path=topk_path,
    )
    plt.close("all")
    print(f"Classnumber TopK figure saved: {topk_path}")

    for rank, record in enumerate(split_records[:top_k], start=1):
        step_path = steps_dir / f"rank{rank:02d}_{_safe_name(record['file'])}_class{record['classnumber']}_steps.png"
        plot_classnumber_step(
            ref_gm,
            record["grid_maps"],
            score_mode=rank_by,
            title=f"Rank {rank}: {record['file']} class {record['classnumber']} ({rank_by})",
            min_area=args.count_partial_min_area,
            top_k=args.count_partial_top_k_proposals,
            proposal_mode=args.count_partial_proposal_mode,
            rotation_tolerance=args.count_partial_rotation_tolerance,
            min_token_score=args.count_partial_min_token_score,
            score_shape_weight=args.count_partial_score_shape_weight,
            score_position_weight=args.count_partial_score_position_weight,
            score_scale_weight=args.count_partial_score_scale_weight,
            min_relative_token_area=args.count_partial_min_relative_token_area,
            scale_area_weight=args.count_partial_scale_area_weight,
            scale_pca_weight=args.count_partial_scale_pca_weight,
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
        for rank, record in enumerate(mo_split_records[:top_k], start=1):
            step_path = mo_steps_dir / f"rank{rank:02d}_{_safe_name(record['file'])}_class{record['classnumber']}_mo_steps.png"
            plot_classnumber_step(
                ref_gm,
                record["grid_maps"],
                score_mode=rank_by,
                title=f"Rank {rank}: {record['file']} class {record['classnumber']} ({rank_by}, matched-only)",
                min_area=args.count_partial_min_area,
                top_k=args.count_partial_top_k_proposals,
                proposal_mode=args.count_partial_proposal_mode,
                rotation_tolerance=args.count_partial_rotation_tolerance,
                min_token_score=args.count_partial_min_token_score,
                score_shape_weight=args.count_partial_score_shape_weight,
                score_position_weight=args.count_partial_score_position_weight,
                score_scale_weight=args.count_partial_score_scale_weight,
                min_relative_token_area=args.count_partial_min_relative_token_area,
                scale_area_weight=args.count_partial_scale_area_weight,
                scale_pca_weight=args.count_partial_scale_pca_weight,
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
