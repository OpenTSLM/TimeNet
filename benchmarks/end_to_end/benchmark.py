"""Fail when end-to-end performance regresses between two revisions."""

from __future__ import annotations

import argparse
from collections import defaultdict
from statistics import median

from benchmarks.end_to_end.core import MATRIX
from benchmarks.end_to_end.orchestrator import measure, revision_capabilities, worktrees


_TIME_METRICS = ("convert_ns", "write_ns", "read_ns", "total_ns")


def regression_percent(before: float, after: float) -> float:
    """Return the percentage increase from ``before`` to ``after``."""
    return (after / before - 1.0) * 100.0


def main() -> None:
    """Measure repeated runs and reject median timing or stored-size regressions.

    Raises:
        SystemExit: If validation fails or a metric exceeds the regression budget.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before_revision")
    parser.add_argument("after_revision")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument(
        "--max-regression-percent",
        type=float,
        default=0.0,
        help="largest allowed median timing or stored-size increase (default: 0)",
    )
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        parser.error("--repeats must be >= 1 and --warmups must be >= 0")

    with worktrees(args.before_revision, args.after_revision, keep=args.keep) as (before, after, root):
        before_capabilities = revision_capabilities(before)
        after_capabilities = revision_capabilities(after)
        failures: list[str] = []
        for case in MATRIX:
            if case.values_backend == "zarr" and not (before_capabilities["zarr"] and after_capabilities["zarr"]):
                print(f"SKIP {case.name}: both revisions must support Zarr")
                continue
            if case.profile == "rich" and not (before_capabilities["rich"] and after_capabilities["rich"]):
                print(f"SKIP {case.name}: both revisions must support dtype/N-D specs")
                continue
            results: dict[str, list[dict[str, float]]] = defaultdict(list)
            for repetition in range(args.warmups + args.repeats):
                # Alternate order to reduce systematic hot-cache/thermal bias.
                order = (("before", before), ("after", after))
                if repetition % 2:
                    order = tuple(reversed(order))
                for label, worktree in order:
                    result = measure(
                        worktree,
                        root / "outputs" / f"{case.name}-{label}-{repetition}",
                        case,
                        scale=args.scale,
                    )
                    if repetition >= args.warmups:
                        results[label].append(result["metrics"])

            print(f"\n{case.name}")
            for metric in (*_TIME_METRICS, "disk_bytes"):
                before_value = median(run[metric] for run in results["before"])
                after_value = median(run[metric] for run in results["after"])
                regression = regression_percent(before_value, after_value)
                unit = "ns" if metric.endswith("_ns") else "bytes"
                print(f"  {metric:<12} {before_value:>12.0f} -> {after_value:>12.0f} {unit} ({regression:+.2f}%)")
                if regression > args.max_regression_percent:
                    failures.append(
                        f"{case.name}/{metric}: {regression:+.2f}% exceeds {args.max_regression_percent:+.2f}%"
                    )
        if failures:
            raise SystemExit("Performance regression check failed:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()
