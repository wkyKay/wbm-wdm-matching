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

