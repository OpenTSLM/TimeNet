"""Fail when two revisions produce logically different TimeF datasets.

The comparison is backend-neutral: physical Parquet/Zarr artifacts may differ, but metadata, schema,
tasks, records, annotations, series contracts, dtype, shape, and exact value bytes must not.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from benchmarks.end_to_end.core import MATRIX
from benchmarks.end_to_end.orchestrator import measure, revision_capabilities, worktrees


def main() -> None:
    """Compare every matrix case across two isolated revisions.

    Raises:
        SystemExit: If any logical fingerprint differs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before_revision")
    parser.add_argument("after_revision")
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    with worktrees(args.before_revision, args.after_revision, keep=args.keep) as (before, after, root):
        before_capabilities = revision_capabilities(before)
        after_capabilities = revision_capabilities(after)
        failures: list[str] = []
        for case in MATRIX:
            if case.values_backend == "zarr" and not after_capabilities["zarr"]:
                print(f"SKIP {case.name}: target revision has no Zarr backend")
                continue
            if case.profile == "rich" and not (before_capabilities["rich"] and after_capabilities["rich"]):
                print(f"SKIP {case.name}: both revisions must support dtype/N-D specs")
                continue
            before_case = case
            if case.values_backend == "zarr" and not before_capabilities["zarr"]:
                before_case = replace(case, name=f"{case.name}-parquet-reference", values_backend="parquet")
            before_result = measure(before, root / "outputs" / "before" / case.name, before_case, scale=args.scale)
            after_result = measure(after, root / "outputs" / "after" / case.name, case, scale=args.scale)
            if before_result["fingerprint"] != after_result["fingerprint"]:
                failures.append(f"{case.name}: {before_result['fingerprint']} != {after_result['fingerprint']}")
            else:
                print(f"PASS {case.name}: {after_result['fingerprint']}")
        if failures:
            raise SystemExit("Logical no-op check failed:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()
