"""Convert CODE-15%'s exams into lazy 12-lead Records, each with an age task and an abnormality task.

Each exam with a trace becomes a Record whose leads reread its HDF5 part on demand, in millivolts and without
the authors' zero padding. README.md explains how this follows the authors.

Metadata is provenance and never model input. Task annotations are never model input either.
"""

import asyncio
from pathlib import Path
from typing import ClassVar

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.types import Annotation
from timenet_connectors.datasets.zenodo.code_15 import release
from timenet_connectors.datasets.zenodo.code_15.keys import AnnotationKey
from timenet_connectors.datasets.zenodo.code_15.loader import Code15DatasetLoader
from timenet_connectors.datasets.zenodo.code_15.release import Part, PinnedCounts
from timenet_connectors.download import Artifact, download_files, ensure_archive


class Code15Connector(BaseConnector[Path]):
    """Build CODE-15% from ``exams.csv`` and the 18 zips of its Zenodo record."""

    SOURCE_PARTS: ClassVar[tuple[Part, ...]] = release.PARTS
    """The zips, each pinned by its sha256."""
    SOURCE_COUNTS: ClassVar[PinnedCounts] = release.PINNED_COUNTS
    """What the pinned release holds, which every build checks."""

    async def download_async(self, cache_dir: Path) -> list[Path]:
        """Fetch ``exams.csv`` and every zip, check their digests, and extract each zip once.

        Each zip is deleted once extracted, so the cache holds about 68 GB of HDF5.

        Returns:
            The one directory that holds ``exams.csv`` and the parts.
        """
        root = cache_dir / "code15"
        await download_files(
            [Artifact(release.file_url(release.CSV_FILE), root / release.CSV_FILE, sha256=release.CSV_SHA256)]
        )
        await asyncio.gather(
            *(ensure_archive(part.url, root, filename=part.zip_name, sha256=part.sha256) for part in self.SOURCE_PARTS)
        )
        return [root]

    def convert(self, raw_refs: list[Path]) -> TimeFDataset:
        """Convert each exam with a trace into one lazy Record, and stream its age and abnormality tasks.

        The connector adds the record's provenance. ``Code15DatasetLoader`` reads ``exams.csv``, scans every part
        once, checks the pinned counts, and builds the Records; its ``populate`` adds them and streams the tasks,
        not holding them. A malformed file raises ``TimeFFormatError``, and counts that differ from the pinned
        release's raise ``TimeFValidationError``.

        Returns:
            A dataset with the provenance and vocabulary annotations, the Records, and a stream of every task.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        dataset.annotate(Annotation(key=AnnotationKey.SOURCE_URL, value=release.RECORD_URL))
        dataset.annotate(Annotation(key=AnnotationKey.SOURCE_DOI, value=release.RECORD_DOI))
        loader = Code15DatasetLoader.parse_from(raw_refs, parts=self.SOURCE_PARTS, counts=self.SOURCE_COUNTS)
        return loader.populate(dataset)


CONNECTOR = Code15Connector
