"""Measure one format at one operation, and reduce the repetitions to one number."""

from __future__ import annotations

from enum import StrEnum, unique
from pathlib import Path
from statistics import median
import time

from pydantic import BaseModel

from evaluations.errors import EvaluationError
from evaluations.formats.base import Format, FormatName


@unique
class Operation(StrEnum):
    """The operations a run measures."""

    # Read the release and store it in one format.
    WRITE = "write"
    # Materialize every value of a stored artifact.
    READ = "read"


class Artifact(BaseModel):
    """What a format wrote, and what it cost on disk. The harness builds it after the write."""

    # The format that wrote it.
    format: FormatName
    # The file or directory the format wrote.
    path: Path
    # Total bytes on disk, summed over every file when the artifact is a directory.
    size_bytes: int


class Measurement(BaseModel):
    """How long one format took at one operation."""

    # The format measured, and which of its operations was measured.
    format: FormatName
    operation: Operation
    # Median of the recorded repetitions.
    elapsed_ns: int
    # How many repetitions the median was taken over. Counted, not copied from the request.
    repetitions: int
    # How many leading repetitions were discarded. Recorded so a reader of the result can tell
    # that one was, rather than having to trust the run.
    warmups: int


def measure_write(fmt: Format, source: Path, out: Path) -> tuple[Artifact, Measurement]:
    """Write the release once, then measure the time it took and the bytes on disk.

    The write runs one time. There is no median and no warm-up, because the first write is the
    one a user pays for. A repeat also lands on the artifact that the first one committed.

    Args:
        fmt: The format to measure.
        source: The directory the release was extracted into.
        out: Directory the format writes into.

    Returns:
        The artifact written, and how long the write took.
    """
    started = time.perf_counter_ns()
    path = fmt.write(source, out)
    elapsed = time.perf_counter_ns() - started

    artifact = Artifact(format=fmt.name, path=path, size_bytes=directory_size(path))
    measurement = Measurement(
        format=fmt.name,
        operation=Operation.WRITE,
        elapsed_ns=elapsed,
        repetitions=1,
        warmups=0,
    )

    return artifact, measurement


def measure_read(
    fmt: Format,
    path: Path,
    *,
    repeats: int = 5,
    warmups: int = 1,
) -> Measurement:
    """Time ``read_all`` and reduce the repetitions to a median.

    This reports the median, not the mean. One thermal excursion or one slow page fault moves
    a mean. It does not move a median.

    The harness discards the warm-ups instead of averaging them in. The first read pays for the
    format's import and a cold page cache, and this measures neither. The harness requires at
    least one warm-up, so a run cannot produce a number that skipped it.

    Every measurement runs in this process. Peak memory is out of scope, and it is the only
    metric that needs a subprocess. Importing torch costs hundreds of megabytes, which lands in
    the next measurement otherwise.

    Args:
        fmt: The format to measure.
        path: The artifact that format wrote.
        repeats: How many repetitions to record. At least one.
        warmups: How many leading repetitions to discard. At least one.

    Returns:
        The median elapsed time over the recorded repetitions.

    Raises:
        EvaluationError: If ``repeats`` or ``warmups`` is below one.
    """
    if repeats < 1:
        raise EvaluationError(
            f"repeats must be at least 1, because a median needs at least one sample. "
            f"Got {repeats} for format {fmt.name} at {path}"
        )

    if warmups < 1:
        raise EvaluationError(
            f"warmups must be at least 1. The first read pays for the format's import and a "
            f"cold page cache, which is not what this measures. Got {warmups} for format "
            f"{fmt.name} at {path}"
        )

    for _ in range(warmups):
        fmt.read_all(path)

    samples: list[int] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        fmt.read_all(path)
        samples.append(time.perf_counter_ns() - started)

    return Measurement(
        format=fmt.name,
        operation=Operation.READ,
        elapsed_ns=int(median(samples)),
        repetitions=len(samples),
        warmups=warmups,
    )


def directory_size(path: Path) -> int:
    """Total the bytes an artifact occupies.

    A directory reports the sum of every file beneath it, not the size of its own directory
    entry, which says nothing about the data.

    Args:
        path: A file or a directory.

    Returns:
        The size in bytes.
    """
    if path.is_file():
        return path.stat().st_size

    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
