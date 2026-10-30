"""Parse the downloaded benchmark once: its folds and classes, each video's header, its Records and tasks."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Self

from timenet.dataset import Record
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.types import Annotation, ClassificationTask
from timenet_connectors.bases.loader import DatasetLoader
from timenet_connectors.datasets.openworm.movement_database.benchmark import (
    TEST_FOLD,
    AuthorsVideo,
    class_of,
    read_classes,
    read_folds,
    read_video_table,
)
from timenet_connectors.datasets.openworm.movement_database.index import IndexedVideo, read_index
from timenet_connectors.datasets.openworm.movement_database.keys import AnnotationKey
from timenet_connectors.datasets.openworm.movement_database.records import build_record
from timenet_connectors.datasets.openworm.movement_database.release import (
    PINNED_COUNTS,
    DownloadedRelease,
    PinnedCounts,
)
from timenet_connectors.datasets.openworm.movement_database.tasks import strain_task
from timenet_connectors.datasets.openworm.movement_database.videos import Video, read_video


@dataclass(frozen=True)
class _ParsedVideo:
    """One video after parsing: its lazy Record, its fold, and its class."""

    record: Record
    fold: int
    target: str


def _check_video(indexed: IndexedVideo, video: Video) -> None:
    """Check that a features file is the indexed video's: same base name, strain and sex.

    Raises:
        TimeFFormatError: If the file's ``experiment_info`` differs from its Zenodo record.
    """
    experiment = video.experiment
    found = (experiment.base_name, experiment.strain, experiment.sex)
    expected = (indexed.base_name, indexed.strain, indexed.sex)
    if found != expected:
        raise TimeFFormatError(f"{video.path}: the file states {found}, its Zenodo record {expected}")


def _check_join(
    indexed_videos: tuple[IndexedVideo, ...], folds: dict[str, int], table: dict[str, AuthorsVideo]
) -> None:
    """Check that the index, the authors' folds and the authors' table name the same videos.

    Raises:
        TimeFValidationError: If a folded video isn't indexed or listed in the table, or one is that isn't
            folded.
    """
    indexed_names = {video.base_name for video in indexed_videos}
    for source, names in (("index", indexed_names), ("authors' table", set(table))):
        if names != set(folds):
            missing = sorted(set(folds) - names)[:5]
            extra = sorted(names - set(folds))[:5]
            raise TimeFValidationError(f"the {source} lacks the folded videos {missing}, and adds {extra}")


def _authors_class(indexed: IndexedVideo, fold: int, listed: AuthorsVideo) -> str:
    """Return a video's class as the authors' table states it, once its sources agree on the video.

    Returns:
        The class, such as ``N2_XO``.

    Raises:
        TimeFValidationError: If the table's fold differs from ``SWDB_fold_dict.p``, or its sex or class
            differs from the video's Zenodo record.
    """
    found = (listed.fold, listed.sex, listed.strain)
    expected = (fold, indexed.sex, class_of(indexed.strain, indexed.sex))
    if found != expected:
        raise TimeFValidationError(
            f"{indexed.base_name!r}: the authors' table states fold, sex and class {found}, "
            f"their folds and the Zenodo record {expected}"
        )
    return listed.strain


class OpenWormDatasetLoader(DatasetLoader):
    """The authors' single-worm strain benchmark, read and checked: one lazy Record per video, and its task.

    The benchmark defines one task (:func:`strain_task`): classify a video's strain, with N2 hermaphrodites
    and males as two classes. After ``populate``, ``TimeFDataset.get_split`` selects one split.

    The loader keeps each video's header, not its skeletons. The signals reread their file, so the files must
    stay available until writing ends.
    """

    TASK_TYPES = (ClassificationTask,)

    def __init__(self, classes: tuple[str, ...], videos: tuple[_ParsedVideo, ...]) -> None:
        self._classes = classes
        self._videos = videos

    @classmethod
    def parse_from(
        cls,
        refs: Iterable[DownloadedRelease],
        *,
        index: tuple[IndexedVideo, ...] | None = None,
        counts: PinnedCounts = PINNED_COUNTS,
    ) -> Self:
        """Read the authors' benchmark files and every video's header, and build one lazy Record per video.

        Args:
            refs: The one downloaded release.
            index: The videos' Zenodo records, by default the pinned index.
            counts: What the benchmark must hold.

        Returns:
            The loaded benchmark.

        Raises:
            TimeFValidationError: If ``refs`` isn't one release, the index, the folds and the authors' table
                name different videos or disagree on one, the videos' classes aren't the authors' classes, or
                the counts differ from ``counts``.
        """
        releases = list(refs)
        if len(releases) != 1:
            raise TimeFValidationError(f"the movement database loads from one release, got {releases}")
        (release,) = releases
        folds = read_folds(release.folds_path)
        classes = read_classes(release.classes_path)
        table = read_video_table(release.video_table_path)
        indexed_videos = read_index() if index is None else index
        _check_join(indexed_videos, folds, table)
        parsed = []
        for indexed in indexed_videos:
            video = read_video(release.videos_dir / indexed.file_name)
            _check_video(indexed, video)
            fold = folds[indexed.base_name]
            record = build_record(indexed, video, fold)
            target = _authors_class(indexed, fold, table[indexed.base_name])
            parsed.append(_ParsedVideo(record=record, fold=fold, target=target))
        targets = {video.target for video in parsed}
        if targets != set(classes):
            raise TimeFValidationError(
                f"the videos' classes differ from the authors': {sorted(targets ^ set(classes))[:5]}"
            )
        found = PinnedCounts(
            videos=len(parsed),
            classes=len(classes),
            test_videos=sum(video.fold == TEST_FOLD for video in parsed),
        )
        counts.check(found)
        return cls(classes, tuple(parsed))

    def dataset_annotations(self) -> tuple[Annotation, ...]:
        """Return the authors' classes: all a task needs besides its Record.

        Returns:
            The class vocabulary, in the authors' index order, which every task names as its ``target_schema``.
        """
        return (Annotation(key=AnnotationKey.STRAIN_VOCABULARY, value=list(self._classes)),)

    def records(self) -> Iterator[Record]:
        """Return one Record per video, by base name, whose values stay in the files until written."""
        return (video.record for video in self._videos)

    def tasks(self) -> Iterator[ClassificationTask]:
        """Return one task per video, by base name.

        Yields:
            The tasks.
        """
        for video in self._videos:
            yield strain_task(record=video.record, fold=video.fold, target=video.target)
