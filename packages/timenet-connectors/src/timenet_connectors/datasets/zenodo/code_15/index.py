"""Join ``exams.csv`` to the HDF5 parts by exam id: each exam's row, where its trace is, and its padding.

The parts' rows aren't in ``exams.csv``'s order, so each trace is joined by its ``exam_id``, as the author
says to in ``automatic-ecg-diagnosis`` issue 32. A trace that is zero everywhere is set aside, not joined.
"""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from timenet.errors import TimeFFormatError
from timenet_connectors.datasets.zenodo.code_15.exams import ExamRow
from timenet_connectors.datasets.zenodo.code_15.release import Part
from timenet_connectors.datasets.zenodo.code_15.tracings import TraceScan, part_path, scan_part


@dataclass(frozen=True)
class Exam:
    """One exam with a trace: its ``exams.csv`` row, and its trace as scanned."""

    row: ExamRow
    scan: TraceScan


@dataclass(frozen=True)
class ExamIndex:
    """Every exam of the release, joined to its trace."""

    exams: tuple[Exam, ...]
    """The exams with a trace, by part, then by row in the part."""
    all_zero_exam_ids: frozenset[int]
    """The exams whose trace is zero everywhere, which have no Record."""


def _row_of(scan: TraceScan, part: Part, rows_by_id: Mapping[int, ExamRow]) -> ExamRow:
    """Return the ``exams.csv`` row of a scanned trace.

    Returns:
        The row.

    Raises:
        TimeFFormatError: If ``exams.csv`` doesn't list the exam, or lists it in another part.
    """
    row = rows_by_id.get(scan.exam_id)
    if row is None or row.trace_file != part.hdf5_name:
        raise TimeFFormatError(
            f"{scan.location.path}: row {scan.location.row} holds exam {scan.exam_id}, which exams.csv doesn't "
            f"list in {part.hdf5_name}"
        )
    return row


def _scan_exams(root: Path, parts: Iterable[Part], rows_by_id: Mapping[int, ExamRow]) -> Iterator[Exam]:
    """Scan every part in turn, and join each trace to its ``exams.csv`` row as it is scanned.

    Yields:
        Each exam with its trace, all-zero ones included, by part, then by row in the part.

    Raises:
        TimeFFormatError: If a part is missing, holds an exam that ``exams.csv`` doesn't list in that part, or
            holds an exam that already has a trace.
    """
    seen: set[int] = set()
    for part in parts:
        for scan in scan_part(part_path(root, part)):
            row = _row_of(scan, part, rows_by_id)
            if scan.exam_id in seen:
                raise TimeFFormatError(f"{scan.location.path}: exam {scan.exam_id} has two traces")
            seen.add(scan.exam_id)
            yield Exam(row=row, scan=scan)


def _check_every_exam_has_a_trace(root: Path, rows_by_id: Mapping[int, ExamRow], exams: Sequence[Exam]) -> None:
    """Check that every exam of ``exams.csv`` was found in a part.

    :func:`_scan_exams` yields only exams that ``exams.csv`` lists, each once, so equal counts mean every exam
    has a trace; the missing ones are looked up only for the error.

    Raises:
        TimeFFormatError: If an exam has no trace.
    """
    if len(exams) == len(rows_by_id):
        return
    missing = sorted(set(rows_by_id) - {exam.row.exam_id for exam in exams})
    raise TimeFFormatError(f"{root}: {len(missing)} exams of exams.csv have no trace, such as {missing[:5]}")


def index_exams(root: Path, rows: Iterable[ExamRow], parts: Iterable[Part]) -> ExamIndex:
    """Scan every part once and join each trace to its ``exams.csv`` row.

    Args:
        root: The directory that holds the extracted parts.
        rows: Every row of ``exams.csv``.
        parts: The release's parts.

    Returns:
        The joined exams, and the exams whose trace is zero everywhere.

    Raises:
        TimeFFormatError: If a part is missing, holds an exam that ``exams.csv`` doesn't list in that part, or
            holds an exam twice, or if an exam of ``exams.csv`` has no trace.
    """  # noqa: DOC502 - raised by the helpers
    rows_by_id = {row.exam_id: row for row in rows}
    scanned = tuple(_scan_exams(root, parts, rows_by_id))
    _check_every_exam_has_a_trace(root, rows_by_id, scanned)
    return ExamIndex(
        exams=tuple(exam for exam in scanned if not exam.scan.all_zero),
        all_zero_exam_ids=frozenset(exam.row.exam_id for exam in scanned if exam.scan.all_zero),
    )
