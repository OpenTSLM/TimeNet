import csv
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta, timezone
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import pickle
from typing import ClassVar

import h5py
import numpy as np
import pytest

from timenet.dataset.axis import RegularAxis
from timenet.engine import store_dataset
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.types import ClassificationTask, Split, unix_us
from timenet.types.text import read_value
from timenet_connectors.datasets.openworm.movement_database import OpenWormMovementConnector
from timenet_connectors.datasets.openworm.movement_database.benchmark import (
    TEST_FOLD,
    class_of,
    read_classes,
    read_folds,
    split_of,
)
from timenet_connectors.datasets.openworm.movement_database.index import INDEX_COLUMNS, read_index
from timenet_connectors.datasets.openworm.movement_database.keys import AnnotationKey
from timenet_connectors.datasets.openworm.movement_database.loader import OpenWormDatasetLoader
from timenet_connectors.datasets.openworm.movement_database.records import record_metadata
from timenet_connectors.datasets.openworm.movement_database.release import (
    PINNED_COUNTS,
    DownloadedRelease,
    PinnedCounts,
    check_md5,
)
from timenet_connectors.datasets.openworm.movement_database.tasks import StrainTask
from timenet_connectors.datasets.openworm.movement_database.videos import period_us, read_video
from timenet_connectors.datasets.openworm.movement_database.vocabulary import Sex


_BST = timezone(timedelta(hours=1))


def _skeletons(n_frames: int, seed: int, missing: tuple[int, ...] = ()) -> np.ndarray:
    skeletons = np.random.default_rng(seed).normal(size=(n_frames, 49, 2)).astype(np.float32) * 1000
    skeletons[list(missing)] = np.nan
    return skeletons


@dataclass(frozen=True)
class _Video:
    """One fixture video: its Zenodo record, its experiment, its frames, and the authors' fold."""

    record_id: int
    strain: str
    sex: str
    ventral_side: str
    timestamp: str
    fps: float
    skeletons: np.ndarray
    fold: int
    frames: tuple[int, ...] | None = None
    """The per-frame table's ``timestamp`` column, by default 0 to n - 1."""
    xy_units: bytes = b"micrometers"


# Three videos as the release writes them: a hermaphrodite mutant, an N2 male, and an N2 hermaphrodite.
# Their frame rates are two that the release uses: 1 / 33.3 ms and 1 / 38.9 ms.
_VIDEOS = {
    "unc-1 (e580) on food L_2010_01_01__10_00_00___1___1": _Video(
        record_id=101,
        strain="CB580",
        sex="hermaphrodite",
        ventral_side="clockwise",
        timestamp="2010-06-01T10:00:00+01:00",
        fps=1 / 0.0333,
        skeletons=_skeletons(7, seed=1, missing=(0, 3)),
        fold=0,
    ),
    "N2 male on food R_2011_02_03__12_00_00___2___2": _Video(
        record_id=102,
        strain="N2",
        sex="male",
        ventral_side="anticlockwise",
        timestamp="2011-02-03T12:00:00+00:00",
        fps=1 / 0.0389,
        skeletons=_skeletons(5, seed=2),
        fold=1,
    ),
    "N2 on food L_2011_02_03__13_00_00___3___3": _Video(
        record_id=103,
        strain="N2",
        sex="hermaphrodite",
        ventral_side="clockwise",
        timestamp="2011-02-03T13:00:00+00:00",
        fps=1 / 0.0333,
        skeletons=_skeletons(6, seed=3, missing=(5,)),
        fold=2,
    ),
}
_CLASSES = ("CB580_XX", "N2_XO", "N2_XX")
_FIXTURE_COUNTS = PinnedCounts(videos=3, classes=3, test_videos=1)


def _write_features(path: Path, base_name: str, video: _Video) -> None:
    """Write a features file as Tierpsy Tracker does, with only what the connector reads."""
    skeletons = video.skeletons
    info = {
        "base_name": base_name,
        "strain": video.strain,
        "sex": video.sex,
        "ventral_side": video.ventral_side,
        "timestamp": video.timestamp,
        "lab": {"name": "William R Schafer"},
        "food": "OP50",
    }
    n_frames = len(skeletons)
    table = np.zeros(n_frames, dtype=[("worm_index", "<f4"), ("timestamp", "<f4"), ("skeleton_id", "<f4")])
    table["worm_index"] = 1
    table["timestamp"] = np.arange(n_frames) if video.frames is None else video.frames
    with h5py.File(path, "w") as handle:
        handle["experiment_info"] = json.dumps(info).encode()
        handle["coordinates/skeletons"] = skeletons
        dataset = handle.create_dataset("features_timeseries", data=table)
        dataset.attrs["fps"] = video.fps
        dataset.attrs["xy_units"] = video.xy_units
        dataset.attrs["time_units"] = b"seconds"


def _write_index(path: Path, videos_dir: Path) -> None:
    rows = []
    for base_name, video in sorted(_VIDEOS.items()):
        file = videos_dir / f"{video.record_id}.hdf5"
        md5 = hashlib.md5(file.read_bytes(), usedforsecurity=False).hexdigest()
        rows.append([base_name, video.record_id, md5, file.stat().st_size, video.strain, video.sex])
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(INDEX_COLUMNS)
        writer.writerows(rows)


def _release(tmp_path: Path, *, folds=None, classes=None) -> DownloadedRelease:
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir(exist_ok=True)
    for base_name, video in _VIDEOS.items():
        _write_features(videos_dir / f"{video.record_id}.hdf5", base_name, video)
    folds_path = tmp_path / "SWDB_fold_dict.p"
    classes_path = tmp_path / "SWDB_straindict.p"
    folds = {base_name: video.fold for base_name, video in _VIDEOS.items()} if folds is None else folds
    classes = {name: index for index, name in enumerate(_CLASSES)} if classes is None else classes
    folds_path.write_bytes(pickle.dumps(folds))
    classes_path.write_bytes(pickle.dumps(classes))
    _write_index(tmp_path / "index.csv.gz", videos_dir)
    return DownloadedRelease(folds_path=folds_path, classes_path=classes_path, videos_dir=videos_dir)


def _fixture_connector(tmp_path: Path) -> type[OpenWormMovementConnector]:
    class _FixtureConnector(OpenWormMovementConnector):
        """The connector, pinned to the fixtures' index and counts instead of the release's."""

        CARD = Path(__file__).parents[1] / "dataset.yaml"
        INDEX: ClassVar[Path] = tmp_path / "index.csv.gz"
        SOURCE_COUNTS: ClassVar[PinnedCounts] = _FIXTURE_COUNTS

    return _FixtureConnector


def _convert(tmp_path: Path):
    release = _release(tmp_path)
    return _fixture_connector(tmp_path)().convert([release])


def _parse(tmp_path: Path, release: DownloadedRelease) -> OpenWormDatasetLoader:
    return OpenWormDatasetLoader.parse_from(
        [release], index=read_index(tmp_path / "index.csv.gz"), counts=_FIXTURE_COUNTS
    )


def _skeleton_values(signal) -> tuple[np.ndarray, np.ndarray]:
    """Return a skeleton signal's frames as NaN-filled ``(n, 49, 2)`` floats, and which frames are null."""
    values = signal.to_arrow()
    null = np.asarray(values.is_null())
    dense = np.full((len(values), 49, 2), np.nan, dtype=np.float32)
    dense[~null] = values.storage.flatten().to_numpy().reshape(-1, 49, 2)
    return dense, null


def test_convert_builds_one_record_per_video_by_base_name(tmp_path):
    dataset = _convert(tmp_path)
    names = [record_metadata(record).base_name for record in dataset.records]
    assert names == sorted(_VIDEOS)
    for record in dataset.records:
        metadata = record_metadata(record)
        video = _VIDEOS[metadata.base_name]
        assert metadata.zenodo_record == video.record_id
        assert metadata.fold == video.fold
        assert metadata.subject_id is None


def test_each_record_holds_one_skeleton_tensor_signal_as_the_file_stores_it(tmp_path):
    dataset = _convert(tmp_path)
    for record in dataset.records:
        video = _VIDEOS[record_metadata(record).base_name]
        (source,) = record.sources
        (signal,) = source.signals
        assert (source.name, signal.name) == ("Tierpsy Tracker", "skeleton")
        assert signal.spec.value_shape == (49, 2)
        assert signal.spec.dimension_names == ("point", "coordinate")
        assert str(signal.spec.unit_value) == "micrometer"
        assert signal.spec.nullable
        dense, null = _skeleton_values(signal)
        expected = video.skeletons
        np.testing.assert_array_equal(null, np.isnan(expected).all(axis=(1, 2)))
        np.testing.assert_array_equal(dense, expected)


def test_a_frame_without_skeleton_is_a_null_timestep(tmp_path):
    dataset = _convert(tmp_path)
    by_name = {record_metadata(record).base_name: record for record in dataset.records}
    (signal,) = by_name["unc-1 (e580) on food L_2010_01_01__10_00_00___1___1"].signals
    assert signal.to_arrow().is_null().to_pylist() == [True, False, False, True, False, False, False]


def test_the_axis_is_regular_at_the_files_frame_period(tmp_path):
    dataset = _convert(tmp_path)
    periods = {record_metadata(record).base_name: record.signals[0].time_axis for record in dataset.records}
    assert periods["unc-1 (e580) on food L_2010_01_01__10_00_00___1___1"] == RegularAxis(period_us=Fraction(33_300))
    assert periods["N2 male on food R_2011_02_03__12_00_00___2___2"] == RegularAxis(period_us=Fraction(38_900))


def test_record_start_time_and_ventral_side_come_from_the_experiment(tmp_path):
    dataset = _convert(tmp_path)
    for record in dataset.records:
        video = _VIDEOS[record_metadata(record).base_name]
        assert record.start_time.timestamp == unix_us(datetime.fromisoformat(video.timestamp))
        annotations = {annotation.key: read_value(annotation.value) for annotation in record.annotations}
        assert annotations == {AnnotationKey.VENTRAL_SIDE: video.ventral_side}
    first = dataset.records[2]
    assert record_metadata(first).base_name.startswith("unc-1")
    assert first.start_time.timestamp == unix_us(datetime(2010, 6, 1, 9, 0, tzinfo=UTC))
    assert first.start_time.timestamp == unix_us(datetime(2010, 6, 1, 10, 0, tzinfo=_BST))


def test_the_class_is_never_a_record_annotation(tmp_path):
    dataset = _convert(tmp_path)
    for record in dataset.records:
        values = {read_value(annotation.value) for annotation in record.annotations}
        assert not values & {"CB580", "N2", "hermaphrodite", "male", *_CLASSES}


def test_one_strain_task_per_video_with_the_authors_class_and_split(tmp_path):
    dataset = _convert(tmp_path)
    tasks = list(dataset.iter_tasks())
    assert len(tasks) == len(_VIDEOS)
    for task in tasks:
        assert isinstance(task, ClassificationTask)
        (record,) = task.inputs
        video = _VIDEOS[record_metadata(record).base_name]
        assert [read_value(target) for target in task.targets] == [class_of(video.strain, Sex(video.sex))]
        assert task.target_schema == AnnotationKey.STRAIN_VOCABULARY
        assert task.split is (Split.TEST if video.fold == TEST_FOLD else Split.TRAIN)
        assert task.scope is None
        assert task.prompt is None


def test_dataset_annotations_hold_the_sources_and_the_class_vocabulary(tmp_path):
    dataset = _convert(tmp_path)
    annotations = {annotation.key: read_value(annotation.value) for annotation in dataset.annotations}
    assert annotations[AnnotationKey.STRAIN_VOCABULARY] == list(_CLASSES)
    assert annotations[AnnotationKey.SOURCE_URL] == "https://zenodo.org/communities/open-worm-movement-database"
    assert "6d4d6b6f04e05a93baea7a5a7550a7a180b60c94" in annotations[AnnotationKey.BENCHMARK_URL]


def test_tasks_of_selects_the_test_fold(tmp_path):
    loader = _parse(tmp_path, _release(tmp_path))
    test = list(loader.tasks_of(StrainTask, split=Split.TEST))
    assert [task.target for task in test] == ["CB580_XX"]


def test_write_and_read_back_keeps_every_value_null_and_task(tmp_path):
    dataset = _convert(tmp_path)
    dataset.derive_schema()
    version_dir = store_dataset(dataset, tmp_path / "out", values_backend=OpenWormMovementConnector.values_backend)
    with TimeFReader(open_local_version(version_dir)) as reader:
        restored = reader.read()
        assert len(restored.records) == len(_VIDEOS)
        for record in restored.records:
            video = _VIDEOS[record_metadata(record).base_name]
            dense, _ = _skeleton_values(record.signals[0])
            np.testing.assert_array_equal(dense, video.skeletons)
        targets = sorted(str(read_value(target)) for task in restored.tasks for target in task.targets or ())
        assert targets == sorted(_CLASSES)


def test_a_partly_missing_frame_is_rejected_when_read(tmp_path):
    release = _release(tmp_path)
    base_name, video = next(iter(_VIDEOS.items()))
    skeletons = video.skeletons.copy()
    skeletons[1, 4, 0] = np.nan
    _write_features(release.videos_dir / f"{video.record_id}.hdf5", base_name, replace(video, skeletons=skeletons))
    loader = _parse(tmp_path, release)
    record = next(record for record in loader.records() if record_metadata(record).base_name == base_name)
    with pytest.raises(TimeFFormatError, match="partly NaN"):
        record.signals[0].to_arrow()


def test_a_file_that_disagrees_with_its_zenodo_record_is_rejected(tmp_path):
    release = _release(tmp_path)
    base_name, video = next(iter(_VIDEOS.items()))
    _write_features(release.videos_dir / f"{video.record_id}.hdf5", base_name, replace(video, strain="CB4856"))
    with pytest.raises(TimeFFormatError, match="Zenodo record"):
        _parse(tmp_path, release)


def test_folds_and_index_must_name_the_same_videos(tmp_path):
    folds = {base_name: video.fold for base_name, video in _VIDEOS.items()}
    folds["another video"] = 1
    with pytest.raises(TimeFValidationError, match="lacks the folded videos"):
        _parse(tmp_path, _release(tmp_path, folds=folds))


def test_every_class_must_have_videos(tmp_path):
    classes = {name: index for index, name in enumerate((*_CLASSES, "ZZ15_XX"))}
    with pytest.raises(TimeFValidationError, match="classes differ"):
        _parse(tmp_path, _release(tmp_path, classes=classes))


def test_counts_must_match_the_pinned_ones(tmp_path):
    release = _release(tmp_path)
    with pytest.raises(TimeFValidationError, match="pinned benchmark"):
        OpenWormDatasetLoader.parse_from([release], index=read_index(tmp_path / "index.csv.gz"), counts=PINNED_COUNTS)


def test_frames_must_be_numbered_without_gaps(tmp_path):
    release = _release(tmp_path)
    base_name, video = next(iter(_VIDEOS.items()))
    path = release.videos_dir / f"{video.record_id}.hdf5"
    _write_features(path, base_name, replace(video, frames=(0, 1, 2, 4, 5, 6, 7)))
    with pytest.raises(TimeFFormatError, match="frames"):
        read_video(path)


def test_units_must_be_micrometers(tmp_path):
    base_name, video = next(iter(_VIDEOS.items()))
    path = tmp_path / "video.hdf5"
    _write_features(path, base_name, replace(video, xy_units=b"pixels"))
    with pytest.raises(TimeFFormatError, match="xy_units"):
        read_video(path)


@pytest.mark.parametrize(("fps", "period"), [(1 / 0.0333, 33_300), (1 / 0.0392, 39_200), (25.0, 40_000)])
def test_a_frame_rate_gives_a_whole_microsecond_period(fps, period):
    assert period_us(fps, "video") == period


@pytest.mark.parametrize("fps", [30.0, 0.0, -25.0])
def test_a_period_that_is_not_whole_is_rejected(fps):
    with pytest.raises(TimeFFormatError, match="frame rate"):
        period_us(fps, "video")


def test_the_authors_pickles_may_hold_no_class(tmp_path):
    path = tmp_path / "folds.p"
    path.write_bytes(pickle.dumps({"video": np.int64(1)}))
    with pytest.raises(TimeFFormatError, match="numpy"):
        read_folds(path)


def test_class_indices_must_follow_sorted_order(tmp_path):
    path = tmp_path / "classes.p"
    path.write_bytes(pickle.dumps({"N2_XX": 0, "CB580_XX": 1}))
    with pytest.raises(TimeFFormatError, match="sorted order"):
        read_classes(path)


def test_split_and_class_follow_the_authors_code():
    assert split_of(0) is Split.TEST
    assert split_of(1) is split_of(2) is Split.TRAIN
    assert class_of("N2", Sex.MALE) == "N2_XO"
    assert class_of("AQ1031", Sex.HERMAPHRODITE) == "AQ1031_XX"


def test_check_md5_rejects_changed_bytes(tmp_path):
    path = tmp_path / "file"
    path.write_bytes(b"abc")
    check_md5(path, hashlib.md5(b"abc", usedforsecurity=False).hexdigest())
    with pytest.raises(TimeFFormatError, match="MD5"):
        check_md5(path, hashlib.md5(b"abd", usedforsecurity=False).hexdigest())


def test_the_pinned_index_lists_every_video_of_the_benchmark_once():
    videos = read_index()
    assert len(videos) == PINNED_COUNTS.videos
    assert [video.base_name for video in videos] == sorted(video.base_name for video in videos)
    assert len({class_of(video.strain, video.sex) for video in videos}) == PINNED_COUNTS.classes


def test_the_pinned_counts_are_the_papers():
    assert PinnedCounts(videos=10_476, classes=365, test_videos=3_609) == PINNED_COUNTS
