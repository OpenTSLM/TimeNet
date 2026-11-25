"""Derive TSAQA-clean from the published TimeNet parent without ordering tasks."""

from pathlib import Path
from typing import ClassVar

from pydantic import ValidationError

from timenet.connectors import BaseConnector
from timenet.connectors.context import BuildContext
from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.types.ids import EntityId
from timenet.types.reference_keys import ParentAlias
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.source import QuestionFormat, TSAQAMetadata


class TSAQACleanConnector(BaseConnector[None]):
    """Keep TSAQA's records and non-ordering tasks, with lineage to the complete parent.

    Ordering can be unidentifiable: test row 18000 has identical B and D patches, so
    swapping those labels reconstructs exactly the same sequence. This variant excludes
    the whole ordering format; it does not claim that every remaining question is valid.
    """

    parent_alias: ClassVar[ParentAlias] = ParentAlias("tsaqa")

    def download(self, cache_dir: Path) -> None:
        """Use the TimeNet parent resolved by the build engine; no source download is needed."""

    def compose(self, raw: None, context: BuildContext) -> TimeFDataset:
        """Inherit every raw record and only the tasks enabled in this variant."""
        del raw
        parent = context.parents[self.parent_alias]
        enabled_tasks = self.enabled_task_ids(parent)
        dataset = TimeFDataset(metadata=self.metadata())
        dataset.import_from(parent, records=parent.record_ids(), tasks=enabled_tasks)
        return dataset

    @staticmethod
    def enabled_task_ids(parent: TimeFReader) -> list[EntityId]:
        """Select tasks by their resolved format, preserving mislabeled continuation questions.

        The source column labels 10,000 true-or-false or multiple-choice questions as
        ordering. The parent's validated metadata distinguishes these from actual puzzles.

        Returns:
            The parent IDs of all non-ordering tasks, in parent order.

        Raises:
            TimeNetConnectorError: If a parent task lacks valid TSAQA metadata.
        """
        enabled = []
        for task in parent.iter_tasks():
            try:
                metadata = TSAQAMetadata.model_validate(task.metadata)
            except ValidationError as exc:
                raise TimeNetConnectorError(f"TSAQA parent task {task.id}: invalid metadata: {exc}") from exc
            if metadata.question_format is not QuestionFormat.ORDERING:
                enabled.append(task.id)
        return enabled


CONNECTOR = TSAQACleanConnector
