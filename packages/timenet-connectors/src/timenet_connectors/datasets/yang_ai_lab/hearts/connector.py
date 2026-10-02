"""Compose HEARTS tasks over five pinned, taskless corpus record layers."""

from pathlib import Path

from timenet.composition import BuildContext, DatasetBuilder
from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.errors import TimeFValidationError, TimeNetDownloadError
from timenet_connectors.datasets.yang_ai_lab._hearts.cases import Case, case_record_refs, iter_cases
from timenet_connectors.datasets.yang_ai_lab._hearts.release import REPO, REVISION, TASKS, VOCABULARIES
from timenet_connectors.datasets.yang_ai_lab.hearts.tasks import BoundRecord, CaseRecords, convert_case
from timenet_connectors.sources.huggingface_hub import hub_snapshot


class HeartsConnector(BaseConnector[Path]):
    """Connector for the HEARTS released test cases (Hub repo ``yang-ai-lab/HEARTS``)."""

    def download(self, cache_dir: Path) -> list[Path]:  # noqa: PLR6301 - BaseConnector override
        """Fetch every task directory of the pinned release.

        Args:
            cache_dir: Directory where the connector caches Hub files.

        Returns:
            The root of the tree, as its one handle.

        Raises:
            TimeNetDownloadError: If the fetch returned no files for a task directory.
        """
        root = hub_snapshot(REPO, REVISION, cache_dir, tuple(f"{directory}/*.pkl" for directory in TASKS))
        missing = [directory for directory in TASKS if not (root / directory).is_dir()]
        if missing:
            raise TimeNetDownloadError(f"{REPO!r} at {REVISION!r} returned no test cases for {missing} under {root}")
        return [root]

    def convert(  # noqa: PLR6301 - BaseConnector override
        self, raw_refs: list[Path], context: BuildContext | None = None
    ) -> TimeFDataset:
        """Import parent records and build the task-owned parts of every test case.

        Args:
            raw_refs: The single handle :meth:`download` gave back.
            context: Build context resolving all five exact corpus parents.

        Returns:
            The composed HEARTS dataset.

        Raises:
            TimeFValidationError: If conversion has no composition context or a parent lacks a
                required case record.
        """
        if context is None:
            raise TimeFValidationError("HEARTS requires a build context with its five corpus parents")
        dataset = context.dataset()
        dataset.register_annotations(VOCABULARIES.values())
        dataset.add_tasks(
            tasks=[
                convert_case(dataset, case, _import_case_records(dataset, case, context))
                for case in iter_cases(raw_refs[0])
            ]
        )
        return dataset


def _import_case_records(dataset: DatasetBuilder, case: Case, context: BuildContext) -> CaseRecords:
    """Resolve and import one case's records from its matching corpus parent.

    Returns:
        Imported records paired with the source origins needed to place task annotations.

    Raises:
        TimeFValidationError: If the corpus parent lacks a required record.
    """
    refs = case_record_refs(case)
    requested = tuple(dict.fromkeys(ref.record_id for ref in (*refs.inputs, *refs.candidates)))
    parent_records = tuple(context.parent(case.corpus).iter_records(requested)) if requested else ()
    by_id = {record.id: record for record in parent_records}
    missing = sorted(set(requested) - by_id.keys())
    if missing:
        raise TimeFValidationError(f"HEARTS parent {case.corpus!r} lacks case records {missing}")
    imported = {record_id: dataset.import_record(by_id[record_id], parent=case.corpus) for record_id in requested}
    return CaseRecords(
        tuple(BoundRecord(imported[item.record_id], item.origin_us) for item in refs.inputs),
        tuple(BoundRecord(imported[item.record_id], item.origin_us) for item in refs.candidates),
    )


CONNECTOR = HeartsConnector
