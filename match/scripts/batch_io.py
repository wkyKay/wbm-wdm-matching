from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from ..core.local_matching import LocalMatchResult
from ..core.similarity import SimilarityResult


def write_result_log(
    log_path: str | Path,
    rows: List[Tuple[str, dict]],
    result_columns: List[str],
    format_score,
) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["file"] + result_columns + ["mapped_defects"]
    with open(log_path, "w") as f:
        f.write("\t".join(header) + "\n")
        for fname, res in rows:
            scores = [format_score(res.get(col, "")) for col in result_columns]
            scores.append(f"{res.get('_mapped', '')}/{res.get('_input', '')}")
            f.write(f"{fname}\t" + "\t".join(scores) + "\n")


def write_topk_log(
    log_path: str | Path,
    rows: List[Tuple[str, dict]],
    ranking_columns: List[str],
    topk: int,
) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write("metric\trank\tfile\tscore\n")
        for col in ranking_columns:
            scored = collect_numeric_scores(rows, col)
            scored.sort(key=lambda item: item[1], reverse=True)
            limit = topk if topk > 0 else len(scored)
            for rank, (fname, score) in enumerate(scored[:limit], 1):
                f.write(f"{col}\t{rank}\t{fname}\t{score:.6f}\n")


def write_token_match_log(log_path: str | Path, rows: List[Tuple[str, dict]]) -> None:
    """Write per-WBM-token top-K WDM token evidence rows."""
    if not log_path:
        return
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "file",
        "wbm_token_id",
        "rank",
        "wdm_token_id",
        "score",
        "shape_sim",
        "position_affinity",
        "scale_affinity",
        "support_area_affinity",
        "pca_extent_affinity",
        "type_affinity",
        "wbm_area",
        "wdm_area",
        "wbm_mass",
        "wdm_mass",
        "wbm_type",
        "wdm_type",
        "wbm_centroid_row",
        "wbm_centroid_col",
        "wdm_centroid_row",
        "wdm_centroid_col",
    ]
    with open(log_path, "w") as f:
        f.write("\t".join(header) + "\n")
        for fname, res in rows:
            for token_matches in res.get("_token_topk_matches", []):
                for match in token_matches:
                    f.write("\t".join(_match_row(fname, match, per_token=True)) + "\n")


def write_map_match_log(log_path: str | Path, rows: List[Tuple[str, dict]]) -> None:
    """Write highest-scoring token pairs per WDM map, independent of final aggregation."""
    if not log_path:
        return
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "file",
        "rank",
        "wbm_token_id",
        "wdm_token_id",
        "score",
        "shape_sim",
        "position_affinity",
        "scale_affinity",
        "support_area_affinity",
        "pca_extent_affinity",
        "type_affinity",
        "wbm_area",
        "wdm_area",
        "wbm_mass",
        "wdm_mass",
        "wbm_type",
        "wdm_type",
        "wbm_centroid_row",
        "wbm_centroid_col",
        "wdm_centroid_row",
        "wdm_centroid_col",
    ]
    with open(log_path, "w") as f:
        f.write("\t".join(header) + "\n")
        for fname, res in rows:
            for match in res.get("_map_topk_matches", []):
                f.write("\t".join(_match_row(fname, match, per_token=False)) + "\n")


def collect_numeric_scores(rows: List[Tuple[str, dict]], column: str) -> list[tuple[str, float]]:
    scored: list[tuple[str, float]] = []
    for fname, res in rows:
        val = res.get(column)
        if isinstance(val, SimilarityResult):
            scored.append((fname, val.score))
        elif isinstance(val, (float, int)):
            scored.append((fname, float(val)))
    return scored


def format_score(val) -> str:
    if isinstance(val, SimilarityResult):
        return f"{val.score:.6f}"
    if isinstance(val, LocalMatchResult):
        return f"{val.matched_tokens}/{val.wbm_tokens}/{val.wdm_tokens}"
    if isinstance(val, (float, int)):
        return f"{val:.6f}"
    if isinstance(val, str):
        return val
    return str(val)


def _match_row(fname: str, match: dict, per_token: bool) -> list[str]:
    query = match.get("query_token", {})
    candidate = match.get("candidate_token", {})
    common = [
        fname,
        str(match.get("query_token_id", "")),
        str(match.get("rank", "")),
        str(match.get("candidate_token_id", "")),
        _fmt_float(match.get("score", "")),
        _fmt_float(match.get("shape_sim", "")),
        _fmt_float(match.get("position_affinity", "")),
        _fmt_float(match.get("scale_affinity", "")),
        _fmt_float(match.get("support_area_affinity", "")),
        _fmt_float(match.get("pca_extent_affinity", "")),
        _fmt_float(match.get("type_affinity", "")),
        str(query.get("area", "")),
        str(candidate.get("area", "")),
        _fmt_float(query.get("mass", "")),
        _fmt_float(candidate.get("mass", "")),
        str(query.get("geometry_type", "")),
        str(candidate.get("geometry_type", "")),
        _fmt_float(query.get("centroid_row", "")),
        _fmt_float(query.get("centroid_col", "")),
        _fmt_float(candidate.get("centroid_row", "")),
        _fmt_float(candidate.get("centroid_col", "")),
    ]
    if per_token:
        return common
    return [common[0], common[2], common[1], *common[3:]]


def _fmt_float(value) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.6f}"
    return str(value)
