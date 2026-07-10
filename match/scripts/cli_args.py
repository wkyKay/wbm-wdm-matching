from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..core.mappers import MAPPERS
from ..core.representations import REPRESENTATIONS


SIMILARITY_COLUMNS: List[str] = [
    "dice", "iou", "ncc", "cosine",
    "coverage", "leakage", "coverage-leakage", "chamfer",
]

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


def parse_args() -> argparse.Namespace:
    # Preliminary parse to detect --config before building the full parser.
    prelim = argparse.ArgumentParser(add_help=False)
    prelim.add_argument("--config", default=None, help="JSON file with parameter defaults. CLI args override file values.")
    prelim_args, remaining = prelim.parse_known_args()

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
        "--identifier",
        default="",
        help="Optional run identifier. Review figures are saved under <fig-dir>/<identifier>/<review-name> when set.",
    )
    parser.add_argument(
        "--output-dir",
        default="match/output",
        help="Directory for per-file .npz outputs.",
    )
    parser.add_argument(
        "--defect-threshold",
        "--min-defects",
        dest="defect_threshold",
        type=int,
        default=5,
        help="Skip KLARF files with fewer than this many defects before any similarity or visualization work.",
    )
    parser.add_argument(
        "--topk", type=int, default=10,
        help="Number of top-K files to show per metric in the ranking log (0 = all).",
    )


def _add_count_partial_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--save-count-partial-figures",
        action="store_true",
        help="Save count-map partial matching TopK and proposal-step review figures.",
    )
    parser.add_argument(
        "--count-partial-fig-dir",
        default="match/output/count_partial_review",
        help="Directory for count-partial review figures when --save-count-partial-figures is set.",
    )
    parser.add_argument(
        "--count-partial-review-top-k",
        type=int,
        default=3,
        help="Number of candidates shown in the count-partial TopK figure.",
    )
    parser.add_argument(
        "--count-partial-step-max",
        type=int,
        default=3,
        help="Number of top count-partial candidates rendered as proposal-step figures.",
    )
    parser.add_argument(
        "--count-partial-min-area",
        type=int,
        default=5,
        help="Minimum support area for count-partial WBM/WDM tokens.",
    )
    parser.add_argument(
        "--count-partial-top-k-proposals",
        type=int,
        default=6,
        help="Maximum WBM/WDM proposal tokens retained for count-partial matching.",
    )
    parser.add_argument(
        "--count-partial-token-match-top-k",
        type=int,
        default=3,
        help="Number of WDM token candidates saved for each WBM token.",
    )
    parser.add_argument(
        "--count-partial-map-match-top-k",
        type=int,
        default=20,
        help="Number of highest-scoring token pairs saved for each WDM map.",
    )
    parser.add_argument(
        "--count-partial-min-token-score",
        type=float,
        default=0.45,
        help="Minimum final token-pair score required before a pair can enter count-partial matching.",
    )
    parser.add_argument(
        "--count-partial-score-shape-weight",
        type=float,
        default=0.60,
        help="Shape similarity weight in count-partial token score.",
    )
    parser.add_argument(
        "--count-partial-score-position-weight",
        type=float,
        default=0.25,
        help="Position affinity weight in count-partial token score.",
    )
    parser.add_argument(
        "--count-partial-score-scale-weight",
        type=float,
        default=0.15,
        help="Scale affinity weight in count-partial token score.",
    )
    parser.add_argument(
        "--count-partial-min-relative-token-area",
        type=float,
        default=0.10,
        help="Minimum token area relative to the largest same-side token before a token can enter count-partial matching.",
    )
    parser.add_argument(
        "--count-partial-scale-area-weight",
        type=float,
        default=0.50,
        help="Support-area component weight inside count-partial scale affinity.",
    )
    parser.add_argument(
        "--count-partial-scale-pca-weight",
        type=float,
        default=0.50,
        help="PCA-extent component weight inside count-partial scale affinity.",
    )
    parser.add_argument(
        "--count-partial-proposal-mode",
        choices=("cc", "compact"),
        default="cc",
        help="Proposal mode for count-partial/classnumber token extraction. 'cc' preserves legacy connected components.",
    )
    parser.add_argument(
        "--count-partial-rotation-tolerance",
        action="store_true",
        help="Use a rotation-tolerant shape descriptor for count-partial/classnumber token matching.",
    )


def _add_classnumber_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--use-classnumber",
        action="store_true",
        help="Split each KLARF by classnumber and run additional per-class WDM matching.",
    )
    parser.add_argument(
        "--save-classnumber-figures",
        action="store_true",
        help="Save WBM, full WDM, and classnumber-split WDM review figures. Requires --use-classnumber.",
    )
    parser.add_argument(
        "--classnumber-fig-dir",
        default="match/output/classnumber_review",
        help="Directory for classnumber split review figures.",
    )
    parser.add_argument(
        "--classnumber-match-mode",
        choices=("count", "binary"),
        default="count",
        help="Scoring mode for classnumber split matching: 'count' uses count-partial token matching, 'binary' uses binary-token partial matching.",
    )
    parser.add_argument(
        "--classnumber-binary-dilation",
        type=int,
        default=1,
        help="Deprecated compatibility option; binary matching now uses token descriptors without dilation.",
    )
    parser.add_argument(
        "--classnumber-binary-beta",
        type=float,
        default=0.5,
        help="Deprecated compatibility option; binary matching now uses token-descriptor partial matching.",
    )


def _validate_required_args(args: argparse.Namespace) -> None:
    missing = []
    if not args.klarf_dir:
        missing.append("--klarf-dir")
    if not args.reference:
        missing.append("--reference")
    if missing:
        print(f"ERROR: the following arguments are required: {', '.join(missing)}", file=sys.stderr)
        sys.exit(2)
