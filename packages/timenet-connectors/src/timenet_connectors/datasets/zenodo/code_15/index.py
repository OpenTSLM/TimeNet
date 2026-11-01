"""Join ``exams.csv`` to the HDF5 parts by exam id: each exam's row, where its trace is, and its padding.

The parts' rows aren't in ``exams.csv``'s order, so each trace is joined by its ``exam_id``, as the author
says to in ``automatic-ecg-diagnosis`` issue 32. A trace that is zero everywhere is set aside, not joined.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from timenet.errors import TimeFFormatError
from timenet_connectors.datasets.zenodo.code_15.exams import ExamRow
from timenet_connectors.datasets.zenodo.code_15.preprocessing import Padding, centred_padding
from timenet_connectors.datasets.zenodo.code_15.release import Part
from timenet_connectors.datasets.zenodo.code_15.tracings import TraceScan, part_path, scan_part


@dataclass(frozen=True)
class Exam:
    """One exam with a trace: its ``exams.csv`` row, and its trace as scanned."""

    row: ExamRow
    scan: TraceScan

    @property
    def padding(self) -> Padding:
        """Return the zeros the authors added around the trace."""
        return centred_padding(self.scan.leading_zeros, self.scan.trailing_zeros)


@dataclass(frozen=True)
class ExamIndex:
    """Every exam of the release, joined to its trace."""

    exams: tuple[Exam, ...]
    """The exams with a trace, by part, then by row in the part."""
    all_zero_exam_ids: frozenset[int]
    """The exams whose trace is zero everywhere, which have no Record."""


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
    """
    rows_by_id = {row.exam_id: row for row in rows}
    seen: set[int] = set()
    exams: list[Exam] = []
    all_zero: set[int] = set()
    for part in parts:
        path = part_path(root, part)
        for scan in scan_part(path):
            row = rows_by_id.get(scan.exam_id)
            if row is None or row.trace_file != part.hdf5_name:
                raise TimeFFormatError(
                    f"{path}: row {scan.location.row} holds exam {scan.exam_id}, which exams.csv doesn't list "
                    f"in {part.hdf5_name}"
                )
            if scan.exam_id in seen:
                raise TimeFFormatError(f"{path}: exam {scan.exam_id} has two traces")
            seen.add(scan.exam_id)
            if scan.all_zero:
                all_zero.add(scan.exam_id)
                continue
            exams.append(Exam(row=row, scan=scan))
    missing = sorted(set(rows_by_id) - seen)
    if missing:
        raise TimeFFormatError(f"{root}: {len(missing)} exams of exams.csv have no trace, such as {missing[:5]}")
    return ExamIndex(exams=tuple(exams), all_zero_exam_ids=frozenset(all_zero))
