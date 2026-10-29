"""Parse the downloaded SensorCaps files into Records and a re-iterable caption stream."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Self

from timenet.dataset import Record
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, AnswerTask
from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.bases.loader import DatasetLoader
from timenet_connectors.datasets.bash_lab.sensorcaps.parse import iter_rows
from timenet_connectors.datasets.bash_lab.sensorcaps.records import RowCache, make_record
from timenet_connectors.datasets.bash_lab.sensorcaps.release import HF_REPO, SOURCE_REVISION


class SensorCapsLoader(DatasetLoader):
    """SensorCaps' Records and tasks, one of each per source row.

    SensorCaps defines one task: the row's caption of its Record, as an :class:`AnswerTask` that
    repeats how GPT-4o-mini generated the caption. The source publishes no split of its rows, so the
    tasks have none.
    """

    TASK_TYPES = (AnswerTask,)

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

    def tasks(self) -> Iterator[AnswerTask]:
        """Re-read each row's instruction and caption, without retaining the corpus.

        Yields:
            One caption task per row, in file and row order: the system message, verbatim, as prompt,
            the assistant message, verbatim, as target, and the rest of the user message (the summary,
            the derived features, and the location) as input annotations.
        """
        for file in self._files:
            for ref, row in iter_rows(file):
                record = self._records[ref.key]
                yield AnswerTask(
                    inputs=(record,),
                    prompt=row.system,
                    targets=(row.caption,),
                    input_annotations=record.annotations,
                    split=None,
                )

    def dataset_annotations(self) -> Sequence[Annotation]:  # noqa: PLR6301 - the loader interface
        """Return the pinned source.

        Returns:
            The repository and revision the dataset was built from.
        """
        return (
            Annotation(key="source_repository", value=HF_REPO),
            Annotation(key="source_revision", value=SOURCE_REVISION),
        )
