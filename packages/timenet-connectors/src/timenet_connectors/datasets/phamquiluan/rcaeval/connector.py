"""Convert the nine pinned RCAEval zips into lazy Records and each case's root cause tasks.

Each failure case becomes a Record whose metrics, logs, traces and derived series reread their zip on
demand, and a ClassificationTask per level the authors score: the root cause service, and, where the
fault has one, the root cause indicator. README.md explains how this follows the authors.

Metadata is provenance and never model input. An annotation is model input only for a task that names
it in ``input_annotations``.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.types import Annotation
from timenet_connectors.datasets.phamquiluan.rcaeval import release
from timenet_connectors.datasets.phamquiluan.rcaeval.keys import AnnotationKey
from timenet_connectors.datasets.phamquiluan.rcaeval.loader import RCAEvalDatasetLoader
from timenet_connectors.datasets.phamquiluan.rcaeval.release import PinnedArchive, Suite, UntimedRows
from timenet_connectors.download import Artifact, download_files


class RCAEvalConnector(BaseConnector[Path]):
    """Build RCAEval from the zips of its Zenodo record, the files the authors' ``main.py`` reads."""

    SOURCE_ARCHIVES: ClassVar[Mapping[Suite, PinnedArchive]] = release.PINNED_ARCHIVES
    """Each zip's digest and inventory, which every download and build checks."""
    SOURCE_UNTIMED_ROWS: ClassVar[Mapping[tuple[str, str], UntimedRows]] = release.PINNED_UNTIMED_ROWS
    """The metric rows without a time, which every build checks and leaves out."""

    async def download_async(self, cache_dir: Path) -> list[Path]:
        """Fetch the nine zips and check their digests. They stay zipped: the build reads their files in place.

        Returns:
            The zips, in suite order.
        """
        return await download_files(
            Artifact(suite.archive_url, cache_dir / suite.archive_filename, sha256=pinned.sha256)
            for suite, pinned in self.SOURCE_ARCHIVES.items()
        )

    def convert(self, raw_refs: list[Path]) -> TimeFDataset:
        """Convert each case into one lazy Record, and stream each case's tasks.

        The connector adds the pinned source's provenance. ``RCAEvalDatasetLoader`` reads each case's files
        once, checks them and the pinned inventory, and builds the Records; its ``populate`` adds them and
        streams the tasks, not holding them. A malformed file raises ``TimeFFormatError``, and an inventory
        that differs from :attr:`SOURCE_ARCHIVES` raises ``TimeFValidationError``.

        Returns:
            A dataset with the provenance annotations, the Records, and a stream of every task.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        for key, value in (
            (AnnotationKey.SOURCE_URL, release.RECORD_URL),
            (AnnotationKey.SOURCE_DOI, release.RECORD_DOI),
        ):
            dataset.annotate(Annotation(key=key, value=value))
        return RCAEvalDatasetLoader.parse_from(
            raw_refs, archives=self.SOURCE_ARCHIVES, untimed=self.SOURCE_UNTIMED_ROWS
        ).populate(dataset)


CONNECTOR = RCAEvalConnector
