"""Combine the five HEARTS child datasets without copying their signal values."""

from pathlib import Path

from timenet.composition import BuildContext
from timenet.connectors import BaseConnector
from timenet.dataset import Record, TimeFDataset
from timenet.types import TSCorrespondenceTask
from timenet_connectors.datasets.yang_ai_lab.hearts_cgmacros.release import VOCABULARIES as CGMACROS_VOCABULARIES
from timenet_connectors.datasets.yang_ai_lab.hearts_coswara.release import VOCABULARIES as COSWARA_VOCABULARIES
from timenet_connectors.datasets.yang_ai_lab.hearts_coughvid.release import VOCABULARIES as COUGHVID_VOCABULARIES
from timenet_connectors.datasets.yang_ai_lab.hearts_vctk.release import VOCABULARIES as VCTK_VOCABULARIES


class HeartsConnector(BaseConnector[Path]):
    """An aggregate of the five task-bearing HEARTS children."""

    def download(self, cache_dir: Path) -> list[Path]:  # noqa: ARG002, PLR6301
        """Use only the datasets declared as parents.

        Returns:
            An empty list: this layer has no external raw files.
        """
        return []

    def compose(self, raw_refs: list[Path], context: BuildContext) -> TimeFDataset:  # noqa: ARG002, PLR6301
        """Import child records and explicitly carry over their tasks.

        Returns:
            The complete benchmark with no newly owned signal values.

        """
        dataset = TimeFDataset(metadata=context.metadata)
        dataset.register_annotations(
            (
                *CGMACROS_VOCABULARIES.values(),
                *COSWARA_VOCABULARIES.values(),
                *COUGHVID_VOCABULARIES.values(),
                *VCTK_VOCABULARIES.values(),
            )
        )
        for dataset_id, parent in context.parents.items():
            imported = set()
            tasks = tuple(parent.iter_tasks())
            for task in tasks:
                candidates = task.candidate_records if isinstance(task, TSCorrespondenceTask) else ()
                targets = tuple(value for value in task.targets or () if isinstance(value, Record))
                for record in (*task.inputs, *candidates, *targets):
                    if record.id not in imported:
                        record.task_ids = ()
                        dataset.import_record(record, parent=dataset_id)
                        imported.add(record.id)
            # Preserve any records that do not participate in a task as well.
            for record in parent.iter_records(identity for identity in parent.record_ids() if identity not in imported):
                dataset.import_record(record, parent=dataset_id)
            dataset.add_tasks(tasks=tasks)
        return dataset


CONNECTOR = HeartsConnector
