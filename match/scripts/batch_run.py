"""Batch experiment runner — reads a JSON experiments file and calls run() for each entry.

JSON format::

    {
      "common": {
        "mode": "count-partial",
        "mapper": "physical-coordinate",
        "representation": "density",
        "die_x_range": [-20, 20],
        "die_y_range": [-20, 20]
      },
      "experiments": [
        {"klarf_dir": "/data/klarf1", "reference": "/data/ref1.png", "identifier": "exp1"},
        {"klarf_dir": "/data/klarf2", "reference": "/data/ref2.png", "identifier": "exp2"}
      ]
    }

``common`` is optional — fallback keys shared by all experiments.
Each experiment **must** provide ``klarf_dir``, ``reference``, and ``identifier``.
Per-experiment keys override ``common`` keys.

Usage::

    PYTHONPATH=wbm-wdm-matching python3 -m match.scripts.batch_run \\
      --experiments batch.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from .main import make_args, run as run_main


def main() -> None:
    args = parse_args()
    spec = _load_spec(args.experiments)
    experiments = spec.get("experiments", [])
    if not experiments:
        print("ERROR: experiments list is empty.", file=sys.stderr)
        sys.exit(1)

    common = spec.get("common", {})
    _validate_experiments(experiments, common)

    print(f"Running {len(experiments)} experiment(s) …\n")
    t0 = time.monotonic()
    for i, exp in enumerate(experiments, 1):
        merged = {**common, **exp}
        identifier = merged["identifier"]
        print(f"[{i}/{len(experiments)}] {identifier}")
        run_main(make_args(**merged))
        print()
    elapsed = time.monotonic() - t0
    print(f"All {len(experiments)} experiment(s) finished in {elapsed:.1f}s.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch WBM-WDM matching experiment runner.")
    parser.add_argument(
        "--experiments",
        required=True,
        help="Path to a JSON file defining experiments (see docstring for format).",
    )
    return parser.parse_args()


def _load_spec(path: str) -> Dict[str, Any]:
    path_obj = Path(path)
    if not path_obj.is_file():
        print(f"ERROR: experiments file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with path_obj.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _validate_experiments(experiments: List[Dict[str, Any]], common: Dict[str, Any]) -> None:
    seen: set[str] = set()
    for i, exp in enumerate(experiments, 1):
        merged = {**common, **exp}
        for key in ("klarf_dir", "reference", "identifier"):
            if not merged.get(key):
                print(
                    f"ERROR: experiment #{i} is missing '{key}' "
                    f"(resolve with common or set per-experiment).",
                    file=sys.stderr,
                )
                sys.exit(1)
        identifier = str(merged["identifier"])
        if identifier in seen:
            print(f"ERROR: duplicate identifier '{identifier}' at experiment #{i}.", file=sys.stderr)
            sys.exit(1)
        seen.add(identifier)


if __name__ == "__main__":
    main()
