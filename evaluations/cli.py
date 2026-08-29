"""Run the whole comparison: write each format from the release, then measure each read."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from evaluations.errors import EvaluationError
from evaluations.formats.base import Format, FormatName
from evaluations.formats.pandas_ import PandasFormat
from evaluations.formats.timef import TimeFFormat
from evaluations.formats.torch_ import TorchFormat
from evaluations.harness import Artifact, Measurement, measure_read, measure_write
from evaluations.report import print_summary, write_json
from evaluations.result import EvaluationResult, collect_environment


# Every format under test, in the order a run measures them. The name is the key, so the loop
# names a format and never holds a class.
FORMATS: Mapping[FormatName, Callable[[], Format]] = {
    FormatName.PANDAS: PandasFormat,
    FormatName.TORCH: TorchFormat,
    FormatName.TIMEF: TimeFFormat,
}


def run_evaluation(
    source: Path,
    out_dir: Path,
    *,
    repeats: int = 5,
    warmups: int = 1,
) -> EvaluationResult:
    """Write the release in every format, and time both the write and the read of each.

    Every format reads the release itself, so the loop below is the whole run. Each format is
    written and then read before the next one starts, and both measurements are kept.

    Nothing is skipped. A precondition failure raises and stops the run, because a report that
    looks complete and is not is worse than no report.

    Args:
        source: The directory the release was extracted into.
        out_dir: Directory the run directory is created beneath.
        repeats: Recorded read repetitions per format.
        warmups: Discarded leading read repetitions per format.

    Returns:
        Everything the run produced, already written to JSON and summarized on standard output.

    Raises:
        EvaluationError: If the source is missing.
    """
    if not source.is_dir():
        raise EvaluationError(f"source directory does not exist: {source}")

    run_id = uuid4().hex[:12]
    started_at = datetime.now(UTC)
    environment = collect_environment()
    run_dir = out_dir / run_id

    artifacts: list[Artifact] = []
    measurements: list[Measurement] = []
    for name, build_format in FORMATS.items():
        fmt = build_format()
        artifact, write = measure_write(fmt, source, run_dir / name)
        read = measure_read(fmt, artifact.path, repeats=repeats, warmups=warmups)

        artifacts.append(artifact)
        measurements.append(write)
        measurements.append(read)

    result = EvaluationResult(
        run_id=run_id,
        started_at=started_at,
        source_path=source,
        environment=environment,
        artifacts=tuple(artifacts),
        measurements=tuple(measurements),
    )
    write_json(result, out_dir)
    print_summary(result)

    return result


def main() -> None:
    """Run the comparison from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="directory the Sleep-EDF release was extracted into")
    parser.add_argument("--out", type=Path, default=Path("evaluations/results"))
    parser.add_argument("--repeats", type=int, default=5, help="recorded read repetitions, at least 1")
    parser.add_argument("--warmups", type=int, default=1, help="discarded read repetitions, at least 1")
    args = parser.parse_args()

    run_evaluation(
        args.source,
        args.out,
        repeats=args.repeats,
        warmups=args.warmups,
    )


if __name__ == "__main__":
    main()
