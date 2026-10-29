"""Build taskless corpus layers from the pinned HEARTS test-case release."""

from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

from timenet.composition import BuildContext
from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.errors import TimeNetDownloadError
from timenet_connectors.datasets.yang_ai_lab.hearts.pickles import load_payload
from timenet_connectors.datasets.yang_ai_lab.hearts.records import Case, case_records
from timenet_connectors.datasets.yang_ai_lab.hearts.release import REPO, REVISION, TASKS
from timenet_connectors.sources.huggingface_hub import hub_snapshot


CORPORA = ("cgmacros", "coswara", "coughvid", "harespod", "vctk")
"""Corpus names represented by taskless HEARTS record layers."""


def iter_cases(root: Path, corpus: str | None = None) -> Iterator[Case]:
    """Yield loaded test cases in task-directory and numeric-file order.

    Args:
        root: Root of the pinned HEARTS snapshot.
        corpus: Optional corpus name. When omitted, yield all available cases.

    Yields:
        Each matching HEARTS test case.
    """
    for directory, definition in TASKS.items():
        if corpus is not None and directory.partition("/")[0] != corpus:
            continue
        folder = root / directory
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.pkl"), key=lambda path: int(path.stem)):
            yield Case(directory, definition, int(path.stem), path, load_payload(str(path)))


class HeartsCorpusConnector(BaseConnector[Path]):
    """Base connector for one taskless, frozen corpus layer distributed by HEARTS."""

    CORPUS: ClassVar[str]
    """Top-level HEARTS corpus directory owned by the concrete connector."""

    def download(self, cache_dir: Path) -> list[Path]:
        """Fetch this corpus's task directories from the pinned HEARTS release.

        Args:
            cache_dir: Directory where the connector caches Hub files.

        Returns:
            The snapshot root as its one handle.

        Raises:
            TimeNetDownloadError: If the snapshot lacks a declared task directory.
        """
        directories = tuple(directory for directory in TASKS if directory.startswith(f"{self.CORPUS}/"))
        root = hub_snapshot(REPO, REVISION, cache_dir, tuple(f"{directory}/*.pkl" for directory in directories))
        missing = [directory for directory in directories if not (root / directory).is_dir()]
        if missing:
            raise TimeNetDownloadError(f"{REPO!r} at {REVISION!r} returned no test cases for {missing} under {root}")
        return [root]

    def convert(self, raw_refs: list[Path], context: BuildContext | None = None) -> TimeFDataset:
        """Build reusable input and candidate records, without benchmark tasks.

        Args:
            raw_refs: The single snapshot root from :meth:`download`.
            context: Unused because these are root layers.

        Returns:
            A taskless dataset containing this corpus's task-specific records.
        """
        _ = context
        dataset = TimeFDataset(metadata=self.metadata())
        for case in iter_cases(raw_refs[0], self.CORPUS):
            records = case_records(case)
            for built in (*records.inputs, *records.candidates):
                dataset.add_record(record=built.record)
        return dataset
