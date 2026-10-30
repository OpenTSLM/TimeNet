"""Convert the MIT-BIH Arrhythmia Database into lazy Records with every reference label, and their tasks.

Each of the 48 records becomes a Record whose two leads reread the ``.dat`` on demand, annotated with the
cardiologists' beat, rhythm, signal quality and other labels. Each record has a task for the beats and one for
the rhythms, the outputs the authors' WFDB comparators score, and each beat a classification task. README.md
explains how this follows the authors.

Metadata is provenance and never model input.
"""

from pathlib import Path
from typing import ClassVar

from timenet.dataset import TimeFDataset
from timenet.types import Annotation
from timenet_connectors.bases.physionet import BasePhysioNetConnector
from timenet_connectors.datasets.physionet.mitdb import release
from timenet_connectors.datasets.physionet.mitdb.keys import AnnotationKey
from timenet_connectors.datasets.physionet.mitdb.loader import MitdbDatasetLoader, WfdbAccess
from timenet_connectors.datasets.physionet.mitdb.release import PinnedCounts
from timenet_connectors.download import ensure_archive, find_dir_containing


class MitdbConnector(BasePhysioNetConnector[Path]):
    """Build the MIT-BIH Arrhythmia Database from the pinned zip of PhysioNet's ``mitdb`` 1.0.0."""

    SOURCE_COUNTS: ClassVar[PinnedCounts] = release.PINNED_COUNTS
    """What the pinned archive holds, which every build checks."""

    async def download_async(self, cache_dir: Path) -> list[Path]:  # noqa: PLR6301 - the connector interface
        """Fetch the pinned archive, check its digest, and extract it once.

        Returns:
            The extracted release directory, which holds ``RECORDS``.
        """
        root = await ensure_archive(
            release.ARCHIVE_URL, cache_dir, filename=release.ARCHIVE_FILENAME, sha256=release.ARCHIVE_SHA256
        )
        return [find_dir_containing(root, f"{release.ARCHIVE_ROOT}/{release.RECORDS_FILE}")]

    def convert(self, raw_refs: list[Path]) -> TimeFDataset:
        """Convert each record into one lazy Record, and stream its tasks.

        The connector adds the pinned source's provenance. ``MitdbDatasetLoader`` reads each record's header
        and reference annotations once, checks the pinned counts, and builds the Records; its ``populate``
        adds them and streams the tasks, not holding them. A malformed file raises ``TimeFFormatError``, and
        counts that differ from the pinned archive's raise ``TimeFValidationError``.

        Returns:
            A dataset with the provenance and vocabulary annotations, the Records, and a stream of every task.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        for key, value in (
            (AnnotationKey.SOURCE_URL, release.ARCHIVE_URL),
            (AnnotationKey.SOURCE_SHA256, release.ARCHIVE_SHA256),
        ):
            dataset.annotate(Annotation(key=key, value=value))
        loader = MitdbDatasetLoader.parse_from(raw_refs, wfdb=self.wfdb_access(), counts=self.SOURCE_COUNTS)
        return loader.populate(dataset)

    def wfdb_access(self) -> WfdbAccess:
        """Return how the loader reads WFDB files: through this connector's PhysioNet base.

        Returns:
            The header parser, the lead loader, and the lazy ``wfdb`` import.
        """
        return WfdbAccess(read_header=self._read_header, lead_loader=self._lead_loader, module=self._wfdb)


CONNECTOR = MitdbConnector
