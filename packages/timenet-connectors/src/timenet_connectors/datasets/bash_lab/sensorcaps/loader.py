"""Parse the downloaded SensorCaps files into Records and a re-iterable caption stream."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Self

from timenet.dataset import Record
from timenet.types import Annotation
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.bases.loader import DatasetLoader
from timenet_connectors.datasets.bash_lab.sensorcaps.parse import iter_rows
from timenet_connectors.datasets.bash_lab.sensorcaps.records import (
    DECLARED_SAMPLING_RATE_HZ,
    SOURCE_SAMPLING_RATE_HZ,
    SOURCE_WINDOW_SAMPLES,
    TIME_AXIS_RECONSTRUCTION,
    ValueLoader,
    make_record,
)
from timenet_connectors.datasets.bash_lab.sensorcaps.release import HF_REPO, SOURCE_REVISION
from timenet_connectors.datasets.bash_lab.sensorcaps.tasks import SensorCaptionTask


class SensorCapsLoader(DatasetLoader[SensorCaptionTask]):
    """SensorCaps' Records and tasks, one of each per source row.

    Task kinds:
        SensorCaptionTask: the row's caption of its Record.
    """

    TASK_KINDS = (SensorCaptionTask,)

    def __init__(
        self,
        files: tuple[HubFileRef, ...],
        records: Mapping[tuple[str, int], Record],
        row_counts: Mapping[HubFileRef, int],
    ) -> None:
        self._files = files
        self._records = records
        self.row_counts = row_counts
        """Rows read from each file, for checking against the pinned inventory."""

    @classmethod
    def parse_from(cls, refs: Iterable[Any]) -> Self:
        """Index every row and build its Record, keeping only byte offsets and small annotations.

        Args:
            refs: The :class:`HubFileRef` list the connector downloaded.

        Returns:
            The loader.
        """
        files: tuple[HubFileRef, ...] = tuple(refs)
        loader = ValueLoader()
        records: dict[tuple[str, int], Record] = {}
        row_counts: dict[HubFileRef, int] = {}
        for file in files:
            count = 0
            for ref, row in iter_rows(file):
                records[ref.key] = make_record(ref, row, loader)
                count += 1
            row_counts[file] = count
        return cls(files, records, row_counts)

    def records(self) -> Iterator[Record]:
        """Return every Record, in file and row order."""
        return iter(self._records.values())

    def tasks(self) -> Iterator[SensorCaptionTask]:
        """Re-read each row's prompt and caption, without retaining the corpus.

        Yields:
            One caption task per row, in file and row order.
        """
        for file in self._files:
            for ref, row in iter_rows(file):
                yield SensorCaptionTask(self._records[ref.key], row)

    def dataset_annotations(self) -> Sequence[Annotation]:  # noqa: PLR6301 - the loader interface
        """Return the source pin and the timing interpretation.

        Returns:
            The dataset's annotations.
        """
        return tuple(
            Annotation(key=key, value=value)
            for key, value in (
                ("source_repository", HF_REPO),
                ("source_revision", SOURCE_REVISION),
                ("declared_sampling_rate_hz", DECLARED_SAMPLING_RATE_HZ),
                ("source_sampling_rate_hz", SOURCE_SAMPLING_RATE_HZ),
                ("source_window_samples", SOURCE_WINDOW_SAMPLES),
                ("time_axis_reconstruction", TIME_AXIS_RECONSTRUCTION),
            )
        )
