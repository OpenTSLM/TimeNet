"""Load CODE-15% into one lazy Record per exam with a trace, and its two tasks."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from timenet.dataset import Record
from timenet.errors import TimeFValidationError
from timenet.types import Annotation
from timenet_connectors.bases.loader import DatasetLoader
from timenet_connectors.datasets.zenodo.code_15.exams import ExamRow, read_exams
from timenet_connectors.datasets.zenodo.code_15.index import index_exams
from timenet_connectors.datasets.zenodo.code_15.keys import AnnotationKey
from timenet_connectors.datasets.zenodo.code_15.records import LeadLoader, build_record
from timenet_connectors.datasets.zenodo.code_15.release import CSV_FILE, PARTS, PINNED_COUNTS, Part, PinnedCounts
from timenet_connectors.datasets.zenodo.code_15.tasks import Code15AbnormalityTask, Code15AgeTask
from timenet_connectors.datasets.zenodo.code_15.vocabulary import Abnormality


_ABNORMALITY_VOCABULARY = "The six abnormalities, in the authors' output order: " + "; ".join(
    f"{abnormality.value}, {abnormality.full_name}" for abnormality in Abnormality
)


@dataclass(frozen=True)
class _ParsedExam:
    """One exam after parsing: its ``exams.csv`` row and its lazy Record."""

    row: ExamRow
    record: Record


class Code15DatasetLoader(DatasetLoader[Code15AgeTask | Code15AbnormalityTask]):
    """CODE-15%'s exams, read and checked: one lazy Record per exam with a trace, and its two tasks.

    CODE-15% has two tasks per exam: its patient's age (:class:`Code15AgeTask`), and its abnormalities among
    the six (:class:`Code15AbnormalityTask`). Select one kind with ``tasks_of``.

    The loader keeps each exam's row, the location of its trace and its padding, not its samples. The signals
    reread their part, so the files must stay available until writing ends.
    """

    TASK_KINDS = (Code15AgeTask, Code15AbnormalityTask)

    def __init__(self, exams: tuple[_ParsedExam, ...]) -> None:
        self._exams = exams

    @classmethod
    def parse_from(
        cls, refs: Iterable[Path], *, parts: tuple[Part, ...] = PARTS, counts: PinnedCounts = PINNED_COUNTS
    ) -> Self:
        """Read ``exams.csv``, scan every part once, join them, and build one lazy Record per exam with a trace.

        Args:
            refs: The one directory that holds ``exams.csv`` and the extracted parts.
            parts: The parts the directory holds.
            counts: What the directory must hold.

        Returns:
            The loaded release.

        Raises:
            TimeFValidationError: If ``refs`` isn't one directory, or its counts differ from ``counts``.
        """
        roots = list(refs)
        if len(roots) != 1:
            raise TimeFValidationError(f"CODE-15% loads from one directory, got {roots}")
        (root,) = roots
        rows = read_exams(root / CSV_FILE)
        index = index_exams(root, rows, parts)
        found = PinnedCounts(
            exams=len(rows),
            patients=len({row.patient_id for row in rows}),
            all_zero_exam_ids=index.all_zero_exam_ids,
        )
        counts.check(root, found)
        # One loader for the Records of this load, so an exam's leads share one read.
        leads = LeadLoader()
        return cls(tuple(_ParsedExam(row=exam.row, record=build_record(exam, leads)) for exam in index.exams))

    def dataset_annotations(self) -> tuple[Annotation, ...]:  # noqa: PLR6301 - the loader interface
        """Return the abnormality vocabulary, which the abnormality tasks name as their ``target_schema``.

        Returns:
            The vocabulary, in the authors' output order.
        """
        return (
            Annotation(
                key=AnnotationKey.ABNORMALITY_VOCABULARY,
                value=[abnormality.value for abnormality in Abnormality],
                description=_ABNORMALITY_VOCABULARY,
            ),
        )

    def records(self) -> Iterator[Record]:
        """Return one Record per exam with a trace, by part, then by row in the part."""
        return (exam.record for exam in self._exams)

    def tasks(self) -> Iterator[Code15AgeTask | Code15AbnormalityTask]:
        """Return each exam's age task, then its abnormality task, in the order of :meth:`records`.

        Yields:
            The tasks.
        """
        for exam in self._exams:
            yield Code15AgeTask(record=exam.record, row=exam.row)
            yield Code15AbnormalityTask(record=exam.record, row=exam.row)
