"""Private worker that measures one matrix case inside one Git worktree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.end_to_end.core import capabilities, case_from_json, write_and_fingerprint


def main() -> None:
    """Run one isolated case and emit its result as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capabilities", action="store_true")
    parser.add_argument("--case", type=case_from_json)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.capabilities:
        print(json.dumps(capabilities(), sort_keys=True))
        return
    if args.case is None or args.output is None:
        parser.error("--case and --output are required unless --capabilities is used")
    fingerprint, metrics = write_and_fingerprint(args.output, args.case, scale=args.scale)
    print(json.dumps({"case": args.case.name, "fingerprint": fingerprint, "metrics": metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
