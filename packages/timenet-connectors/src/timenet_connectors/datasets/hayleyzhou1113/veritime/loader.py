"""Load VeriTime's four files into one lazy Record per distinct series and one task per row."""

from collections import Counter
from collections.abc import Iterable, Iterator
from typing import Self

from timenet.dataset import Record
from timenet.errors import TimeFValidationError
from timenet.types import Annotation
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.bases.loader import DatasetLoader
from timenet_connectors.datasets.hayleyzhou1113.veritime.keys import class_vocabulary
from timenet_connectors.datasets.hayleyzhou1113.veritime.records import SeriesLoader, build_record
from timenet_connectors.datasets.hayleyzhou1113.veritime.release import PINNED_INVENTORY, Family, Part, PinnedInventory
from timenet_connectors.datasets.hayleyzhou1113.veritime.rows import Row, read_rows
from timenet_connectors.datasets.hayleyzhou1113.veritime.series import group_series
from timenet_connectors.datasets.hayleyzhou1113.veritime.tasks import (
    AnomalyDetectionTask,
    InferentialCalculationTask,
    KnowledgeClassificationTask,
    ScenarioAttributionTask,
    VeriTimeTask,
    task_of,
)
from timenet_connectors.datasets.hayleyzhou1113.veritime.vocabulary import CLASS_VOCABULARIES


class VeriTimeDatasetLoader(DatasetLoader[VeriTimeTask]):
    """VeriTime's files, read and checked: one lazy Record per distinct series, and one task per row.

    VeriTime asks one question per row, under the authors' split. Its seven tasks make four kinds:

    - Anomaly detection (:class:`AnomalyDetectionTask`): answer Yes or No to the question's criterion.
    - Scenario attribution (:class:`ScenarioAttributionTask`): choose the option that best explains the
      series.
    - Inferential calculation (:class:`InferentialCalculationTask`): count the events the question
      names.
    - CTU, ECG, EMG and RCW (:class:`KnowledgeClassificationTask`): name the class of a real-world signal.

    Select one kind, or one kind of one split, with ``tasks_of``, such as
    ``tasks_of(KnowledgeClassificationTask, split=Split.TEST)``.

    The loader keeps each row's text fields, not its values, so tasks stream from memory and a signal
    rereads only its own row. The source files must stay available until writing ends.
    """

    TASK_KINDS = (
        AnomalyDetectionTask,
        ScenarioAttributionTask,
        InferentialCalculationTask,
        KnowledgeClassificationTask,
    )

    def __init__(self, rows: tuple[tuple[Row, Record], ...], records: tuple[Record, ...]) -> None:
        self._rows = rows
        self._records = records

    @classmethod
    def parse_from(cls, refs: Iterable[HubFileRef], *, inventory: PinnedInventory = PINNED_INVENTORY) -> Self:
        """Read and check each file once, in order, then build one lazy Record per distinct series.

        ``refs`` are the references the connector's base already checked: each part at most once. A
        malformed row, or a file whose rows per task differ from ``inventory``, raises
        ``TimeFValidationError``. So does a family whose two files hold another number of distinct
        series than ``inventory`` pins; a family with one file loaded is not checked.

        Args:
            refs: The downloaded files, one per part.
            inventory: The pinned rows per file and task, and distinct series per family.

        Returns:
            The loaded release.

        Raises:
            TimeFValidationError: If a reference names no VeriTime part.
        """
        rows: list[Row] = []
        parts: list[Part] = []
        for file in refs:
            if file.part not in set(Part):
                raise TimeFValidationError(f"{file.path}: {file.part!r} is not a VeriTime part")
            part = Part(file.part)
            file_rows = list(read_rows(file.path, part))
            inventory.check_rows(file.path, part, Counter(row.task for row in file_rows))
            rows.extend(file_rows)
            parts.append(part)
        all_series = group_series(rows)
        for family in Family:
            if all(part in parts for part in Part if part.family is family):
                inventory.check_series(family, sum(series.kind.family is family for series in all_series))
        # One loader for the Records of this load, which caches the one row it last reread.
        loader = SeriesLoader()
        records = tuple(build_record(series, loader) for series in all_series)
        record_of = {row.ref: record for series, record in zip(all_series, records, strict=True) for row in series.rows}
        return cls(tuple((row, record_of[row.ref]) for row in rows), records)

    def dataset_annotations(self) -> tuple[Annotation, ...]:
        """Return the class vocabulary of each classification task the loaded rows hold.

        A task names its vocabulary's key as its ``target_schema``. The connector adds the provenance.

        Returns:
            One annotation per classification task present, in the release's task order.
        """
        present = {row.task for row, _ in self._rows}
        return tuple(
            Annotation(key=class_vocabulary(task), value=list(classes))
            for task, classes in CLASS_VOCABULARIES.items()
            if task in present
        )

    def records(self) -> Iterator[Record]:
        """Return one Record per distinct series, in the order its first row appears."""
        return iter(self._records)

    def tasks(self) -> Iterator[VeriTimeTask]:
        """Return one task per row, in file order."""
        return (task_of(row, record) for row, record in self._rows)
