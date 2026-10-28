"""Parse the downloaded SensorCaps files into Records and a re-iterable caption stream."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Self

from timenet.dataset import Record
from timenet.errors import TimeFValidationError
from timenet.types import Annotation
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.bases.loader import DatasetLoader
from timenet_connectors.datasets.bash_lab.sensorcaps.parse import iter_rows
from timenet_connectors.datasets.bash_lab.sensorcaps.records import RowCache, make_record
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

        Raises:
            TimeFValidationError: If a row's system message differs from its file's first row.
        """
        files: tuple[HubFileRef, ...] = tuple(refs)
        cache = RowCache()
        records: dict[tuple[str, int], Record] = {}
        row_counts: dict[HubFileRef, int] = {}
        for file in files:
            count = 0
            # The instruction GPT-4o-mini received, which becomes every task's prompt.
            system: str | None = None
            for ref, row in iter_rows(file):
                if system is None:
                    system = row.system
                elif row.system != system:
                    raise TimeFValidationError(f"{ref.context}: system message differs from the file's first row")
                records[ref.key] = make_record(ref, row, cache)
                count += 1
            row_counts[file] = count
        return cls(files, records, row_counts)

    def records(self) -> Iterator[Record]:
        """Return every Record, in file and row order."""
        return iter(self._records.values())

    def tasks(self) -> Iterator[SensorCaptionTask]:
        """Re-read each row's instruction and caption, without retaining the corpus.

        Yields:
            One caption task per row, in file and row order.
        """
        for file in self._files:
            for ref, row in iter_rows(file):
                yield SensorCaptionTask(self._records[ref.key], row.system, row.caption)

    def dataset_annotations(self) -> Sequence[Annotation]:  # noqa: PLR6301 - the loader interface
        """Return the pinned source.

        Returns:
            The repository and revision the dataset was built from.
        """
        return (
            Annotation(key="source_repository", value=HF_REPO),
            Annotation(key="source_revision", value=SOURCE_REVISION),
        )
