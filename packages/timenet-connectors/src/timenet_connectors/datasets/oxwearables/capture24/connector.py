"""Convert CAPTURE-24's participant recordings into lazy Records and the benchmark's window tasks.

Each participant becomes a Record whose x, y and z signals reread their recording on demand, and each of
the authors' 10 s windows becomes one ClassificationTask per benchmarked label scheme. README.md explains
how this follows the authors.

Metadata is provenance and never model input.
"""

from pathlib import Path
from typing import ClassVar

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.types import Annotation
from timenet_connectors.datasets.oxwearables.capture24 import release
from timenet_connectors.datasets.oxwearables.capture24.keys import AnnotationKey
from timenet_connectors.datasets.oxwearables.capture24.loader import Capture24DatasetLoader
from timenet_connectors.datasets.oxwearables.capture24.release import PinnedCounts
from timenet_connectors.download import ensure_archive, find_dir_containing


class Capture24Connector(BaseConnector[Path]):
    """Build CAPTURE-24 from the pinned zip of its Oxford Research Archive record."""

    SOURCE_COUNTS: ClassVar[PinnedCounts] = release.PINNED_COUNTS
    """What the pinned archive holds, which every build checks."""

    async def download_async(self, cache_dir: Path) -> list[Path]:  # noqa: PLR6301 - the connector interface
        """Fetch the pinned archive, check its digest, and extract it once.

        Returns:
            The extracted ``capture24`` directory.
        """
        root = await ensure_archive(
            release.ARCHIVE_URL, cache_dir, filename=release.ARCHIVE_FILENAME, sha256=release.ARCHIVE_SHA256
        )
        return [find_dir_containing(root, f"{release.ARCHIVE_ROOT}/{release.METADATA_FILE}")]

    def convert(self, raw_refs: list[Path]) -> TimeFDataset:
        """Convert each recording into one lazy Record, and stream one task per window and scheme.

        The connector adds the pinned source's provenance. ``Capture24DatasetLoader`` reads each recording's
        clock and labels once, checks the pinned counts, and builds the Records; its ``populate`` adds them
        and streams the tasks, not holding them. A malformed file raises ``TimeFFormatError``, and counts
        that differ from the pinned archive's raise ``TimeFValidationError``.

        Returns:
            A dataset with the provenance and vocabulary annotations, the Records, and a stream of every task.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        for key, value in (
            (AnnotationKey.SOURCE_URL, release.ARCHIVE_URL),
            (AnnotationKey.SOURCE_SHA256, release.ARCHIVE_SHA256),
        ):
            dataset.annotate(Annotation(key=key, value=value))
        return Capture24DatasetLoader.parse_from(raw_refs, counts=self.SOURCE_COUNTS).populate(dataset)


CONNECTOR = Capture24Connector
