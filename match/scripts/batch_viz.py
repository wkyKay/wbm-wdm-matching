from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

def ensure_mpl() -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")


def save_count_partial_figures(args, ref_gm, rows) -> None:
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
    out_dir = _review_output_dir(args.count_partial_fig_dir, getattr(args, "identifier", ""), "count_partial_review")
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
        save_path=topk_path,
    )
    plt.close("all")
    print(f"Count-partial TopK figure saved: {topk_path}")

    for rank, (name, gm, _) in enumerate(scored[: max(args.count_partial_step_max, 0)], start=1):
        step_path = steps_dir / f"rank{rank:02d}_{_safe_name(name)}_steps.png"
        plot_count_partial_steps(
            ref_gm,
            gm,
            title=f"Rank {rank}: {name}",
            min_area=args.count_partial_min_area,
            top_k=args.count_partial_top_k_proposals,
            proposal_mode=args.count_partial_proposal_mode,
            save_path=step_path,
        )
        plt.close("all")
        print(f"Count-partial step figure saved: {step_path}")


def save_classnumber_figures(args, ref_gm, rows) -> None:
    ensure_mpl()
    import matplotlib.pyplot as plt

    from ..core.classnumber_matching import split_score
    from ..viz.classnumber_visualization import (
        plot_classnumber_splits,
        plot_classnumber_step,
        plot_classnumber_topk_splits,
    )

    review_name = _classnumber_review_name(args)
    out_dir = _review_output_dir(args.classnumber_fig_dir, getattr(args, "identifier", ""), review_name)
    steps_dir = out_dir / "topk_steps"
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    split_records = []
    for fname, res in rows:
        grid_maps = res.get("_grid_maps")
        class_result = res.get("_classnumber_result")
        if grid_maps is None or class_result is None or not class_result.splits:
            continue
        path = out_dir / f"{_safe_name(Path(fname).stem)}_classnumber_splits.png"
        plot_classnumber_splits(
            ref_gm,
            grid_maps,
            class_result,
            title=f"{fname} classnumber split matching",
            save_path=path,
        )
        plt.close("all")
        print(f"Classnumber split figure saved: {path}")
        saved += 1
        for split in class_result.splits:
            split_records.append({
                "file": Path(fname).stem,
                "classnumber": split.classnumber,
                "split": split,
                "grid_maps": split.grid_maps,
                "partial": split.partial,
                "binary": split.binary,
                "rank_score": split.rank_score,
                "rank_mode": class_result.rank_by,
            })

    if saved == 0:
        print("Classnumber figures skipped: no valid classnumber split results available")
        return

    rank_by = _effective_classnumber_rank_by(args)
    split_records.sort(key=lambda item: split_score(item["split"], rank_by), reverse=True)
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

    top_k = max(args.count_partial_review_top_k, 1)
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
            save_path=step_path,
        )
        plt.close("all")
        print(f"Classnumber TopK step figure saved: {step_path}")


def _effective_classnumber_rank_by(args) -> str:
    if args.classnumber_match_mode == "binary":
        return "binary"
    if args.classnumber_match_mode == "count":
        return "count"
    return args.classnumber_rank_by


def _classnumber_review_name(args) -> str:
    mode = getattr(args, "classnumber_match_mode", "count")
    if mode == "both":
        rank_by = _effective_classnumber_rank_by(args)
        return f"classnumber_review_both_rank_{_safe_name(rank_by)}"
    return f"classnumber_review_{_safe_name(mode)}"


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)[:80]


def _review_output_dir(base_dir: str, identifier: str, review_name: str) -> Path:
    base = Path(base_dir)
    run_id = _safe_name(str(identifier).strip())
    if not run_id:
        return base
    if base.name.startswith(review_name) or base.name.startswith("classnumber_review"):
        base = base.parent
    return base / run_id / review_name
