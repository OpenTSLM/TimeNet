"""Run the whole comparison: write each format from the release, then measure each read."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path

from evaluations.errors import EvaluationError
from evaluations.formats.base import Format, FormatName
from evaluations.formats.pandas_ import PandasFormat
from evaluations.formats.timef import TimeFFormat
from evaluations.formats.torch_ import TorchFormat
from evaluations.harness import Artifact, Measurement, Operation, measure_read, measure_write


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
) -> tuple[tuple[Artifact, ...], tuple[Measurement, ...]]:
    """Write the release in every format, and time both the write and the read of each.

    Every format reads the release itself, so the loop below is the whole run. Each format is
    written and then read before the next one starts, and both measurements are kept.

    Nothing is skipped. A precondition failure raises and stops the run, because a report that
    looks complete and is not is worse than no report.

    Args:
        source: The directory the release was extracted into.
        out_dir: Directory the artifacts are written beneath.
        repeats: Recorded read repetitions per format.
        warmups: Discarded leading read repetitions per format.

    Returns:
        The artifacts written, and the measurements in the order they were taken. Each format
        contributes one write measurement and one read measurement.

    Raises:
        EvaluationError: If the source is missing.
    """
    if not source.is_dir():
        raise EvaluationError(f"source directory does not exist: {source}")

    artifacts: list[Artifact] = []
    measurements: list[Measurement] = []
    for name, build_format in FORMATS.items():
        fmt = build_format()
        artifact, write = measure_write(fmt, source, out_dir / name)
        read = measure_read(fmt, artifact.path, repeats=repeats, warmups=warmups)

        artifacts.append(artifact)
        measurements.append(write)
        measurements.append(read)

    return tuple(artifacts), tuple(measurements)


def main() -> None:
    """Run the comparison from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="directory the Sleep-EDF release was extracted into")
    parser.add_argument("--out", type=Path, default=Path("evaluations/results"))
    parser.add_argument("--repeats", type=int, default=5, help="recorded read repetitions, at least 1")
    parser.add_argument("--warmups", type=int, default=1, help="discarded read repetitions, at least 1")
    args = parser.parse_args()

    artifacts, measurements = run_evaluation(
        args.source,
        args.out,
        repeats=args.repeats,
        warmups=args.warmups,
    )

    elapsed = {(one.format, one.operation): one.elapsed_ns / 1e9 for one in measurements}
    for artifact in artifacts:
        write_s = elapsed[artifact.format, Operation.WRITE]
        read_s = elapsed[artifact.format, Operation.READ]
        megabytes = artifact.size_bytes / 1e6
        print(f"{artifact.format:<8} write {write_s:7.3f} s   read {read_s:7.3f} s   size {megabytes:9.1f} MB")


if __name__ == "__main__":
    main()
