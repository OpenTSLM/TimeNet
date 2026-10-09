"""Compose the complete ChatTS training corpus from its configurations."""

from pathlib import Path

from timenet.composition import BuildContext
from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset


class ChatTSTrainingDatasetConnector(BaseConnector[Path]):
    """Compose the complete ChatTS corpus from its five configuration datasets."""

    def download(self, cache_dir: Path) -> list[Path]:  # noqa: ARG002, PLR6301
        """Return no raw files because this dataset is composed from its parents.

        Returns:
            An empty list: this layer has no external raw files.
        """
        return []

    def compose(self, raw_refs: list[Path], context: BuildContext) -> TimeFDataset:  # noqa: ARG002, PLR6301
        """Import each configuration's records and retain its tasks.

        Returns:
            The complete ChatTS corpus with parent-owned signal values.
        """
        dataset = TimeFDataset(metadata=context.metadata)
        for dataset_id, parent in context.parents.items():
            records = tuple(parent.iter_records())
            for record in records:
                dataset.import_record(record, parent=dataset_id)
            dataset.add_tasks(tasks=parent.iter_tasks())
        return dataset


CONNECTOR = ChatTSTrainingDatasetConnector
