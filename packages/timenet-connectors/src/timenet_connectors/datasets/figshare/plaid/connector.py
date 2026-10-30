"""Convert PLAID's CSV files into lazy Records, and its two uses into tasks.

Each submetered and aggregated CSV file becomes a Record whose current and voltage signals reread the file
on demand. Each submetered file has one appliance-type task, and each aggregated file one task to find its
labelled switching events. README.md explains how this follows the paper and the authors' code.

Metadata is provenance and never model input.
"""

import asyncio
from pathlib import Path
from typing import ClassVar

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.types import Annotation
from timenet_connectors.datasets.figshare.plaid.keys import AnnotationKey
from timenet_connectors.datasets.figshare.plaid.loader import PlaidDatasetLoader
from timenet_connectors.datasets.figshare.plaid.release import (
    ARTICLE_URL,
    KNOWN_DEFECTS,
    PINNED_COUNTS,
    KnownDefects,
    PinnedCounts,
    Subset,
)
from timenet_connectors.download import Artifact, download_files, ensure_archive


class PlaidConnector(BaseConnector[Path]):
    """Build PLAID from the pinned files of version 2 of its figshare article."""

    SOURCE_COUNTS: ClassVar[PinnedCounts] = PINNED_COUNTS
    """What the pinned files hold, which every build checks."""
    SOURCE_DEFECTS: ClassVar[KnownDefects] = KNOWN_DEFECTS
    """The pinned files' known defects, which every build checks are exactly the ones it finds."""

    async def download_async(self, cache_dir: Path) -> list[Path]:  # noqa: PLR6301 - the connector interface
        """Fetch both metadata files and both zips, check their digests, and extract the zips once.

        The HDF5 zip holds the same measurements again, so it is not fetched.

        Returns:
            ``cache_dir``, which then holds both metadata files and both extracted subset directories.
        """
        metadata = [
            Artifact(subset.metadata.url, cache_dir / subset.metadata.name, sha256=subset.metadata.sha256)
            for subset in Subset
        ]
        await asyncio.gather(
            download_files(metadata),
            *(
                ensure_archive(
                    subset.archive.url, cache_dir, filename=subset.archive.name, sha256=subset.archive.sha256
                )
                for subset in Subset
            ),
        )
        return [cache_dir]

    def convert(self, raw_refs: list[Path]) -> TimeFDataset:
        """Convert each CSV file into one lazy Record, and stream one task per Record.

        The connector adds the pinned files' provenance. ``PlaidDatasetLoader`` reads both metadata files and
        every file's row count once, checks the pinned counts, and builds the Records; its ``populate`` adds
        them and streams the tasks, not holding them. A malformed file raises ``TimeFFormatError``, and counts
        that differ from the pinned files' raise ``TimeFValidationError``.

        Returns:
            A dataset with the provenance and vocabulary annotations, the Records, and a stream of every task.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        files = [file for subset in Subset for file in (subset.metadata, subset.archive)]
        for key, value in (
            (AnnotationKey.SOURCE_URL, ARTICLE_URL),
            (AnnotationKey.SOURCE_FILES, [f"{file.name} sha256:{file.sha256}" for file in files]),
        ):
            dataset.annotate(Annotation(key=key, value=value))
        return PlaidDatasetLoader.parse_from(raw_refs, counts=self.SOURCE_COUNTS, defects=self.SOURCE_DEFECTS).populate(
            dataset
        )


CONNECTOR = PlaidConnector
