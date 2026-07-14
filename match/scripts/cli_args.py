from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..core.mappers import MAPPERS
from ..core.representations import REPRESENTATIONS

MODES = ("count-partial", "classnumber")

SIMILARITY_COLUMNS: List[str] = ["coverage-leakage"]

PARTIAL_MATCH_COLUMNS: List[str] = [
    "count-partial",
    "count-partial-shape",
    "count-partial-position",
    "count-partial-scale",
    "count-partial-type",
    "count-partial-tokens",
]

PARTIAL_MATCH_MO_COLUMNS: List[str] = [
    "count-partial-mo",
    "count-partial-mo-shape",
    "count-partial-mo-position",
    "count-partial-mo-scale",
    "count-partial-mo-type",
    "count-partial-mo-tokens",
]

CLASSNUMBER_COLUMNS: List[str] = [
    "classnumber-count",
    "best-classnumber",
    "best-classnumber-partial",
    "best-classnumber-tokens",
    "best-classnumber-binary",
    "best-classnumber-binary-shape",
    "best-classnumber-binary-position",
    "best-classnumber-binary-scale",
    "best-classnumber-binary-type",
    "best-classnumber-binary-tokens",
    "best-classnumber-binary-coverage",
    "best-classnumber-binary-leakage",
    "best-classnumber-rank-mode",
    "best-classnumber-rank-score",
    "best-classnumber-mo-partial",
    "best-classnumber-mo-tokens",
    "best-classnumber-mo-binary",
    "best-classnumber-mo-binary-shape",
    "best-classnumber-mo-binary-position",
    "best-classnumber-mo-binary-scale",
    "best-classnumber-mo-binary-type",
    "best-classnumber-mo-binary-tokens",
    "best-classnumber-mo-rank-score",
]

RESULT_COLUMNS: List[str] = SIMILARITY_COLUMNS + PARTIAL_MATCH_COLUMNS + PARTIAL_MATCH_MO_COLUMNS


def parse_args(argv: list[str] | None = None, *, validate: bool = True) -> argparse.Namespace:
    prelim = argparse.ArgumentParser(add_help=False)
    prelim.add_argument("--config", default=None, help="JSON file with parameter defaults. CLI args override file values.")
    prelim_args, remaining = prelim.parse_known_args(argv)

    parser = argparse.ArgumentParser(
        description="Batch WBM-WDM matching: process a directory of KLARF files against a reference WBM."
    )
    _add_config_arg(parser)
    _add_input_args(parser)
    _add_mapping_args(parser)
    _add_output_args(parser)
    _add_count_partial_args(parser)
    _add_classnumber_args(parser)

    if prelim_args.config:
        config_dict = _load_config_file(prelim_args.config)
        parser.set_defaults(**config_dict)

    args = parser.parse_args(remaining)
    if validate:
        _validate_required_args(args)
    return args


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a JSON file containing parameter defaults (dest names as keys). CLI args take precedence.",
    )


def _load_config_file(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        print(f"ERROR: --config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid JSON in --config file {config_path}: {exc}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(data, dict):
        print(f"ERROR: --config file must contain a JSON object, got {type(data).__name__}", file=sys.stderr)
        sys.exit(1)
    return data


def _add_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--klarf-dir",
        help="Directory containing KLARF files to process.",
    )
    parser.add_argument(
        "--klarf-glob",
        default="*.*",
        help="Glob pattern to match KLARF files (default: *.klarf).",
    )
    parser.add_argument(
        "--reference",
        help="Path to the reference WBM PNG.",
    )
    parser.add_argument(
        "--wbm",
        help="Path to the target WBM PNG for shape reference. Defaults to --reference.",
    )


def _add_mapping_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--height", type=int,
        help="Target WBM grid height. Overrides --wbm shape.",
    )
    parser.add_argument(
        "--width", type=int,
        help="Target WBM grid width. Overrides --wbm shape.",
    )
    parser.add_argument(
        "--mapper",
        choices=sorted(MAPPERS),
        default="die-index",
        help="Pluggable coordinate mapping strategy.",
    )
    parser.add_argument(
        "--representation",
        choices=sorted(REPRESENTATIONS),
        default="density",
        help="Pluggable grid map representation.",
    )
    parser.add_argument(
        "--die-x-range", nargs=2, type=int, metavar=("MIN", "MAX"),
        help="Fixed die index range along X (e.g. -20 20).",
    )
    parser.add_argument(
        "--die-y-range", nargs=2, type=int, metavar=("MIN", "MAX"),
        help="Fixed die index range along Y (e.g. -20 20).",
    )
    parser.add_argument(
        "--defect-table-index", type=int, default=0,
        help="Which DefectList to use when a KLARF contains multiple wafers.",
    )
    parser.add_argument(
        "--die-defect-threshold",
        type=int,
        default=1,
        help="Minimum defects in one mapped die/cell required to mark binary_map=1. 1 keeps the legacy count>0 behavior.",
    )


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="count-partial",
        help="Matching mode: count-partial (global + token matching), classnumber (+ classnumber splitting).",
    )
    parser.add_argument(
        "--identifier",
        default="",
        help="Optional run identifier. All outputs saved under <output-dir>/<identifier>/<mode>/.",
    )
    parser.add_argument(
        "--output-dir",
        default="match/output",
        help="Root output directory. Logs and figures saved under <output-dir>/<identifier>/<mode>/.",
    )
    parser.add_argument(
        "--defect-threshold",
        "--min-defects",
        dest="defect_threshold",
        type=int,
        default=5,
        help="Skip KLARF files with fewer than this many defects.",
    )
    parser.add_argument(
        "--topk", type=int, default=10,
        help="Number of top-K files to show per metric in the ranking log (0 = all).",
    )


def _add_count_partial_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--review-top-k",
        type=int,
        default=3,
        help="Number of top candidates shown in WDM raw review figures (shared across modes).",
    )
    parser.add_argument(
        "--step-max",
        type=int,
        default=3,
        help="Number of top candidates rendered as individual proposal-step figures (shared across modes).",
    )
    parser.add_argument(
        "--proposal-min-area",
        type=int,
        default=5,
        help="Minimum support area for WBM/WDM tokens.",
    )
    parser.add_argument(
        "--proposal-top-k",
        type=int,
        default=6,
        help="Maximum WBM/WDM proposal tokens retained for matching.",
    )
    parser.add_argument(
        "--token-match-top-k",
        type=int,
        default=3,
        help="Number of WDM token candidates saved for each WBM token.",
    )
    parser.add_argument(
        "--map-match-top-k",
        type=int,
        default=20,
        help="Number of highest-scoring token pairs saved for each WDM map.",
    )
    parser.add_argument(
        "--token-min-score",
        type=float,
        default=0.45,
        help="Minimum final token-pair score for token matching.",
    )
    parser.add_argument(
        "--token-score-shape-weight",
        type=float,
        default=0.60,
        help="Shape similarity weight in token score.",
    )
    parser.add_argument(
        "--token-score-position-weight",
        type=float,
        default=0.25,
        help="Position affinity weight in token score.",
    )
    parser.add_argument(
        "--token-score-scale-weight",
        type=float,
        default=0.15,
        help="Scale affinity weight in token score.",
    )
    parser.add_argument(
        "--proposal-min-relative-token-area",
        type=float,
        default=0.10,
        help="Minimum token area relative to the largest same-side token.",
    )
    parser.add_argument(
        "--token-scale-area-weight",
        type=float,
        default=0.50,
        help="Support-area component weight inside scale affinity.",
    )
    parser.add_argument(
        "--token-scale-pca-weight",
        type=float,
        default=0.50,
        help="PCA-extent component weight inside scale affinity.",
    )
    parser.add_argument(
        "--proposal-mode",
        choices=("cc", "compact", "sparse-density", "auto"),
        default="cc",
        help="Proposal mode for token extraction. 'cc' preserves legacy components; 'sparse-density' forces multi-scale KDE; 'auto' enables it for fragmented sparse pairs.",
    )
    parser.add_argument(
        "--proposal-rotation-tolerance",
        action="store_true",
        help="Use a rotation-tolerant shape descriptor for token matching.",
    )
    parser.add_argument(
        "--density-sigmas",
        nargs="+",
        type=float,
        default=(0.8, 1.6, 3.2),
        metavar="SIGMA",
        help="Grid-cell Gaussian scales used by sparse-density proposal mode.",
    )
    parser.add_argument(
        "--density-threshold",
        type=float,
        default=0.20,
        help="Relative-to-peak KDE support threshold used by sparse-density proposal mode.",
    )
    parser.add_argument(
        "--density-min-raw-points",
        type=int,
        default=3,
        help="Minimum original occupied grid cells required for one sparse-density token.",
    )
    parser.add_argument(
        "--density-min-raw-mass",
        type=float,
        default=3.0,
        help="Minimum pre-KDE weight mass required for one sparse-density token.",
    )
    parser.add_argument(
        "--density-merge-iou",
        type=float,
        default=0.60,
        help="IoU threshold for deduplicating overlapping sparse-density tokens across scales.",
    )
    parser.add_argument(
        "--density-weight-transform",
        choices=("count", "sqrt", "log1p"),
        default="sqrt",
        help="WDM count transform used as sparse-density KDE point weights.",
    )
    parser.add_argument("--ring-min-area", type=int, default=None, help="Override the adaptive minimum ring-band cell count.")
    parser.add_argument("--ring-edge-r-min", type=float, default=None, help="Override the minimum normalized radius considered part of an edge ring.")
    parser.add_argument("--ring-band-width", type=float, default=None, help="Override the normalized radial half-width around the selected ring band.")
    parser.add_argument("--ring-min-angular-coverage", type=float, default=None, help="Override the adaptive occupied angular-bin ratio required for a ring.")
    parser.add_argument("--ring-angular-bins", type=int, default=None, help="Override the adaptive angular-bin count for compact-ring coverage.")
    parser.add_argument("--ring-max-radial-std", type=float, default=None, help="Override the maximum normalized radial spread for a ring.")
    parser.add_argument("--ring-max-defect-ratio", type=float, default=None, help="Override the maximum valid-area defect ratio permitted for a ring.")
    parser.add_argument("--ring-min-edge-defect-fraction", type=float, default=None, help="Override the minimum defect fraction required in the outer radial region.")


def _add_classnumber_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--classnumber-binary-dilation",
        type=int,
        default=1,
        help="Deprecated compatibility option.",
    )
    parser.add_argument(
        "--classnumber-binary-beta",
        type=float,
        default=0.5,
        help="Deprecated compatibility option.",
    )


def derive_classnumber_match_mode(representation: str) -> str:
    """Map representation to classnumber match mode: 'count'→'count', 'binary'→'binary', others→'count'."""
    return representation if representation in ("count", "binary") else "count"


def _validate_required_args(args: argparse.Namespace) -> None:
    missing = []
    if not args.klarf_dir:
        missing.append("--klarf-dir")
    if not args.reference:
        missing.append("--reference")
    if missing:
        print(f"ERROR: the following arguments are required: {', '.join(missing)}", file=sys.stderr)
        sys.exit(2)
