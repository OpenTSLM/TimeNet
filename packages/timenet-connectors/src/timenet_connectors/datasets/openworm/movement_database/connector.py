"""Convert the Open Worm Movement Database's single-worm videos into lazy Records and the authors' strain tasks.

Each video of the authors' benchmark becomes a Record whose 98 skeleton signals reread its features file on
demand, and one ClassificationTask names its strain. README.md explains how this follows the authors.

Metadata is provenance and never model input.
"""

from pathlib import Path
from typing import ClassVar

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.types import Annotation
from timenet_connectors.datasets.openworm.movement_database import benchmark, release
from timenet_connectors.datasets.openworm.movement_database.index import INDEX_FILE, read_index
from timenet_connectors.datasets.openworm.movement_database.keys import AnnotationKey
from timenet_connectors.datasets.openworm.movement_database.loader import OpenWormDatasetLoader
from timenet_connectors.datasets.openworm.movement_database.release import DownloadedRelease, PinnedCounts
from timenet_connectors.download import Artifact, download_files


# Zenodo limits how fast a guest may download; more parallel downloads only meet HTTP 429 sooner.
_MAX_CONCURRENCY = 4


class OpenWormMovementConnector(BaseConnector[DownloadedRelease]):
    """Build the authors' strain benchmark from its videos' Zenodo records and their repository's folds."""

    # Each timestep is a (49, 2) skeleton, and the Parquet values backend holds only scalars.
    values_backend = "zarr"

    INDEX: ClassVar[Path] = INDEX_FILE
    """The pinned index of the videos' Zenodo records."""
    SOURCE_COUNTS: ClassVar[PinnedCounts] = release.PINNED_COUNTS
    """What the pinned benchmark holds, which every build checks."""

    async def download_async(self, cache_dir: Path) -> list[DownloadedRelease]:
        """Fetch the authors' folds and classes, and every indexed video's features file, and check each digest.

        A run that stops part way, for example on Zenodo's rate limit, keeps the files it finished, and the next
        run downloads only the rest.

        Returns:
            The one downloaded release.
        """
        benchmark_dir = cache_dir / "benchmark"
        videos_dir = cache_dir / "videos"
        folds_path, classes_path = await download_files(
            [
                Artifact(benchmark.FOLDS_URL, benchmark_dir / "SWDB_fold_dict.p", sha256=benchmark.FOLDS_SHA256),
                Artifact(benchmark.CLASSES_URL, benchmark_dir / "SWDB_straindict.p", sha256=benchmark.CLASSES_SHA256),
            ]
        )
        videos = read_index(self.INDEX)
        paths = await download_files(
            [Artifact(video.url, videos_dir / video.file_name) for video in videos], max_concurrency=_MAX_CONCURRENCY
        )
        for video, path in zip(videos, paths, strict=True):
            release.check_md5(path, video.md5)
        return [DownloadedRelease(folds_path=folds_path, classes_path=classes_path, videos_dir=videos_dir)]

    def convert(self, raw_refs: list[DownloadedRelease]) -> TimeFDataset:
        """Convert each video into one lazy Record, and stream one strain task per video.

        The connector adds the sources' provenance. ``OpenWormDatasetLoader`` reads each video's header once,
        checks the pinned counts, and builds the Records; its ``populate`` adds them and streams the tasks.
        A malformed file raises ``TimeFFormatError``, and counts that differ from the pinned benchmark's raise
        ``TimeFValidationError``.

        Returns:
            A dataset with the provenance and vocabulary annotations, the Records, and a stream of every task.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        for key, value in (
            (AnnotationKey.SOURCE_URL, release.COMMUNITY_URL),
            (AnnotationKey.BENCHMARK_URL, f"{benchmark.AUTHORS_REPOSITORY}/tree/{benchmark.AUTHORS_COMMIT}"),
        ):
            dataset.annotate(Annotation(key=key, value=value))
        loader = OpenWormDatasetLoader.parse_from(raw_refs, index=read_index(self.INDEX), counts=self.SOURCE_COUNTS)
        return loader.populate(dataset)


CONNECTOR = OpenWormMovementConnector
