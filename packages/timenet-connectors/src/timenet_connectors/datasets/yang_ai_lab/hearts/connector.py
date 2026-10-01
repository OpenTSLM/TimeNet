"""Download the pinned HEARTS release and convert its test cases."""

from collections.abc import Iterator
from pathlib import Path

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetDownloadError
from timenet_connectors.datasets.yang_ai_lab.hearts.pickles import load_payload
from timenet_connectors.datasets.yang_ai_lab.hearts.records import Case
from timenet_connectors.datasets.yang_ai_lab.hearts.release import REPO, REVISION, TASKS, VOCABULARIES
from timenet_connectors.datasets.yang_ai_lab.hearts.tasks import convert_case
from timenet_connectors.sources.huggingface_hub import hub_snapshot


class HeartsConnector(BaseConnector[Path]):
    """Connector for the HEARTS released test cases (Hub repo ``yang-ai-lab/HEARTS``)."""

    values_backend = "zarr"  # the meal photographs are image tensors

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

    def convert(self, raw_refs: list[Path]) -> TimeFDataset:
        """Build the records and the task of every test case under the tree.

        Args:
            raw_refs: The single handle :meth:`download` gave back.

        Returns:
            The populated dataset.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        dataset.register_annotations(VOCABULARIES.values())
        dataset.add_tasks(tasks=[convert_case(dataset, case) for case in _cases(raw_refs[0])])
        return dataset


def _cases(root: Path) -> Iterator[Case]:
    """Yield the loaded test cases in task-directory and numeric-file order.

    Yields:
        Each test case of every task directory the tree holds.
    """
    for directory, definition in TASKS.items():
        folder = root / directory
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.pkl"), key=lambda path: int(path.stem)):
            yield Case(directory, definition, int(path.stem), path, load_payload(str(path)))


CONNECTOR = HeartsConnector
