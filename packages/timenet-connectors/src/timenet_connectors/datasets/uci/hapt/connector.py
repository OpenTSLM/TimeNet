"""The UCI HAPT connector: continuous smartphone IMU trials with activity labels.

UCI HAPT v2.1 records 50 Hz accelerometer and gyroscope streams for 61 trials across 30 subjects.
Each trial becomes one Record with six signals. Each contiguous labeled segment in
``RawData/labels.txt`` becomes one scoped ClassificationTask over the trial. The connector assigns
the official UCI subject split and carves a deterministic validation set out of the official train
subjects.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from timenet.connectors import BaseConnector
from timenet.dataset import Record, RegularAxis, Signal, Source, TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet.types import (
    Annotation,
    ClassificationTask,
    Label,
    LabelSet,
    Split,
    TimeInterval,
    TimeSeriesSpec,
    ureg,
)
from timenet_connectors.download import ensure_archive, find_dir_containing


HAPT_URL = "https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip"
HAPT_SHA256 = "4ac4ae064227c07045a99551876b54d204837e995e298b6488559e209b3deb09"
SAMPLE_RATE_HZ = 50
US_PER_SAMPLE = 1_000_000 // SAMPLE_RATE_HZ
AXIS = RegularAxis.from_rate_hz(SAMPLE_RATE_HZ)
# ``RawData/labels.txt`` holds five columns per segment: experiment, subject, activity, start, end.
_LABEL_COLUMNS = 5
# Each IMU file is a two-dimensional array of ``_IMU_AXES`` triaxial samples.
_IMU_NDIM = 2
_IMU_AXES = 3

ACTIVITY_NAMES = {
    1: "walking",
    2: "walking_upstairs",
    3: "walking_downstairs",
    4: "sitting",
    5: "standing",
    6: "laying",
    7: "stand_to_sit",
    8: "sit_to_stand",
    9: "sit_to_lie",
    10: "lie_to_sit",
    11: "stand_to_lie",
    12: "lie_to_stand",
}
ACTIVITY = LabelSet(
    [
        Label(name=ACTIVITY_NAMES[activity_id], description=description)
        for activity_id, description in (
            (1, "Walking."),
            (2, "Walking upstairs."),
            (3, "Walking downstairs."),
            (4, "Sitting."),
            (5, "Standing."),
            (6, "Lying down."),
            (7, "Postural transition from standing to sitting."),
            (8, "Postural transition from sitting to standing."),
            (9, "Postural transition from sitting to lying down."),
            (10, "Postural transition from lying down to sitting."),
            (11, "Postural transition from standing to lying down."),
            (12, "Postural transition from lying down to standing."),
        )
    ],
    annotations=[Annotation("name", value="activity")],
)
# The release states accelerations in g and angular velocities in rad/s. A resting accelerometer
# reads about 1 g, so the accelerometer spec is dimensionless rather than m/s^2.
ACCELEROMETER = TimeSeriesSpec(
    spec_type="accelerometer",
    name="Accelerometer",
    unit_value=ureg.dimensionless,
    dtype="float64",
)
GYROSCOPE = TimeSeriesSpec(
    spec_type="gyroscope",
    name="Gyroscope",
    unit_value=ureg.radian / ureg.second,
    dtype="float64",
)


@dataclass(frozen=True)
class HaptLabel:
    """One contiguous labeled segment of a trial."""

    experiment_id: int
    subject_id: int
    activity_id: int
    start: int
    end: int


@dataclass(frozen=True)
class HaptSource:
    """The extracted HAPT archive: the extraction root and the RawData directory."""

    root: Path
    raw_dir: Path


def _read_activity_labels(path: Path) -> None:
    """Check that ``activity_labels.txt`` matches the connector's label names.

    Args:
        path: The ``activity_labels.txt`` file of the release.

    Raises:
        TimeNetConnectorError: If the file is missing, malformed, or disagrees with the label set.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TimeNetConnectorError(f"cannot read {path}: {exc}") from exc
    seen: dict[int, str] = {}
    for lineno, line in enumerate(lines, start=1):
        parts = line.split()
        if not parts:
            continue
        try:
            activity_id = int(parts[0])
            name = parts[1].lower()
        except (IndexError, ValueError) as exc:
            raise TimeNetConnectorError(f"{path.name} line {lineno}: expected '<id> <name>'") from exc
        seen[activity_id] = name
    if set(seen) != set(ACTIVITY_NAMES) or any(
        seen[activity_id] != ACTIVITY_NAMES[activity_id] for activity_id in ACTIVITY_NAMES
    ):
        raise TimeNetConnectorError(f"{path.name} does not match the connector's twelve activity labels")


def _read_subject_ids(path: Path) -> set[int]:
    """Read a subject-id file with one integer per line.

    Args:
        path: The subject-id file.

    Returns:
        The subject ids.

    Raises:
        TimeNetConnectorError: If the file is missing or holds a non-integer line.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TimeNetConnectorError(f"cannot read {path}: {exc}") from exc
    subjects: set[int] = set()
    for lineno, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            continue
        try:
            subjects.add(int(text))
        except ValueError as exc:
            raise TimeNetConnectorError(f"{path.name} line {lineno}: expected an integer subject id") from exc
    return subjects


def _read_labels(path: Path) -> list[HaptLabel]:
    """Read ``RawData/labels.txt`` into typed segments.

    The file has five columns: experiment id, user id, activity id, label start point, and label end
    point. Start and end are 1-indexed inclusive sample positions at 50 Hz.

    Args:
        path: The labels file.

    Returns:
        The labeled segments in file order.

    Raises:
        TimeNetConnectorError: If the file is missing, malformed, or names an unknown activity.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TimeNetConnectorError(f"cannot read {path}: {exc}") from exc
    labels: list[HaptLabel] = []
    for lineno, line in enumerate(lines, start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != _LABEL_COLUMNS:
            raise TimeNetConnectorError(
                f"{path.name} line {lineno}: expected {_LABEL_COLUMNS} columns, got {len(parts)}"
            )
        try:
            experiment_id, subject_id, activity_id, start, end = (int(part) for part in parts)
        except ValueError as exc:
            raise TimeNetConnectorError(f"{path.name} line {lineno}: non-integer column") from exc
        if activity_id not in ACTIVITY_NAMES:
            raise TimeNetConnectorError(f"{path.name} line {lineno}: unknown activity id {activity_id}")
        labels.append(
            HaptLabel(
                experiment_id=experiment_id,
                subject_id=subject_id,
                activity_id=activity_id,
                start=start,
                end=end,
            )
        )
    return labels


def _read_imu(path: Path, *, kind: str, experiment_id: int, subject_id: int) -> np.ndarray:
    """Read one triaxial IMU text file into a float64 array.

    Args:
        path: The ``acc_expXX_userYY.txt`` or ``gyro_expXX_userYY.txt`` file.
        kind: ``"accelerometer"`` or ``"gyroscope"``, for error messages.
        experiment_id: The experiment id, for error messages.
        subject_id: The subject id, for error messages.

    Returns:
        The samples as an ``(n_samples, 3)`` array.

    Raises:
        TimeNetConnectorError: If the file is unreadable, not three-columnar, or holds non-finite
            values.
    """
    try:
        data = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except (OSError, ValueError) as exc:
        raise TimeNetConnectorError(
            f"{kind} file {path.name} for experiment {experiment_id} subject {subject_id} is unreadable: {exc}"
        ) from exc
    if data.ndim != _IMU_NDIM or data.shape[1] != _IMU_AXES or data.shape[0] == 0:
        raise TimeNetConnectorError(
            f"{kind} file {path.name} for experiment {experiment_id} subject {subject_id} must have 3 columns"
        )
    if not np.isfinite(data).all():
        raise TimeNetConnectorError(
            f"{kind} file {path.name} for experiment {experiment_id} subject {subject_id} holds non-finite values"
        )
    return data


def _trial_paths(raw_dir: Path, experiment_id: int, subject_id: int) -> tuple[Path, Path]:
    """Return the accelerometer and gyroscope paths of one trial.

    Args:
        raw_dir: The ``RawData`` directory.
        experiment_id: The experiment id.
        subject_id: The subject id.

    Returns:
        The ``(accelerometer, gyroscope)`` paths.
    """
    stem = f"exp{experiment_id:02d}_user{subject_id:02d}"
    return raw_dir / f"acc_{stem}.txt", raw_dir / f"gyro_{stem}.txt"


def _split_for_subject(subject_id: int, train_subjects: set[int], test_subjects: set[int]) -> Split:
    """Assign a task split from the official UCI subject partition.

    The official test subjects are ``Split.TEST``. Among the official train subjects, ids divisible
    by five form ``Split.VALIDATION``; the rest are ``Split.TRAIN``.

    Args:
        subject_id: The subject id.
        train_subjects: The official train subject ids.
        test_subjects: The official test subject ids.

    Returns:
        The split for the subject's tasks.

    Raises:
        TimeNetConnectorError: If the subject is in neither official split.
    """
    if subject_id in test_subjects:
        return Split.TEST
    if subject_id in train_subjects:
        return Split.VALIDATION if subject_id % 5 == 0 else Split.TRAIN
    raise TimeNetConnectorError(f"subject {subject_id} is in neither the official train nor test split")


def _segment_task(record: Record, segment: HaptLabel, split: Split, *, n_samples: int) -> ClassificationTask:
    """Build the scoped classification task of one labeled segment.

    Args:
        record: The trial record the segment belongs to.
        segment: The labeled segment.
        split: The split of the segment's subject.
        n_samples: The number of samples in the trial.

    Returns:
        The classification task.

    Raises:
        TimeNetConnectorError: If the segment bounds leave the trial.
    """
    if segment.start < 1 or segment.end < segment.start or segment.end > n_samples:
        raise TimeNetConnectorError(
            f"labels.txt segment exp{segment.experiment_id} user{segment.subject_id} "
            f"activity{segment.activity_id} [{segment.start}, {segment.end}] leaves the {n_samples}-sample trial"
        )
    return ClassificationTask(
        inputs=(record,),
        targets=ACTIVITY_NAMES[segment.activity_id],
        label_set=ACTIVITY,
        scope=TimeInterval.micros((segment.start - 1) * US_PER_SAMPLE, segment.end * US_PER_SAMPLE),
        split=split,
    )


class HaptConnector(BaseConnector[HaptSource]):
    """Connector for the UCI HAPT v2.1 release."""

    async def download_async(self, cache_dir: Path) -> HaptSource:  # ruff: ignore[no-self-use]
        """Download and extract the HAPT archive once.

        Args:
            cache_dir: The directory that holds the extracted files.

        Returns:
            The extraction root and its ``RawData`` directory.
        """
        root = await ensure_archive(HAPT_URL, cache_dir, filename="hapt.zip", sha256=HAPT_SHA256)
        return HaptSource(root=root, raw_dir=find_dir_containing(root, "labels.txt"))

    def convert(self, source: HaptSource) -> TimeFDataset:
        """Create one Record per trial and one scoped ClassificationTask per labeled segment.

        Args:
            source: The extracted archive from :meth:`download_async`.

        Returns:
            The populated dataset.

        Raises:
            TimeNetConnectorError: If a required file is missing or malformed, or a segment leaves its
                trial.
        """
        required = (
            source.root / "activity_labels.txt",
            source.root / "Train" / "subject_id_train.txt",
            source.root / "Test" / "subject_id_test.txt",
            source.raw_dir / "labels.txt",
        )
        for path in required:
            if not path.is_file():
                raise TimeNetConnectorError(f"missing required file {path}")
        _read_activity_labels(source.root / "activity_labels.txt")
        train_subjects = _read_subject_ids(source.root / "Train" / "subject_id_train.txt")
        test_subjects = _read_subject_ids(source.root / "Test" / "subject_id_test.txt")
        labels = _read_labels(source.raw_dir / "labels.txt")

        experiments: dict[tuple[int, int], list[HaptLabel]] = {}
        for label in labels:
            experiments.setdefault((label.experiment_id, label.subject_id), []).append(label)
        for segments in experiments.values():
            segments.sort(key=lambda segment: segment.start)

        dataset = TimeFDataset(metadata=self.metadata())
        for (experiment_id, subject_id), segments in sorted(experiments.items()):
            accelerometer_path, gyroscope_path = _trial_paths(source.raw_dir, experiment_id, subject_id)
            accelerometer = _read_imu(
                accelerometer_path, kind="accelerometer", experiment_id=experiment_id, subject_id=subject_id
            )
            gyroscope = _read_imu(gyroscope_path, kind="gyroscope", experiment_id=experiment_id, subject_id=subject_id)
            if accelerometer.shape != gyroscope.shape:
                raise TimeNetConnectorError(
                    f"experiment {experiment_id} subject {subject_id}: accelerometer and gyroscope lengths differ"
                )
            record = Record(
                sources=[
                    Source(
                        "Accelerometer",
                        signals=[
                            Signal.from_values(
                                accelerometer[:, index],
                                spec=ACCELEROMETER,
                                name=f"accelerometer_{axis}",
                                time_axis=AXIS,
                            )
                            for index, axis in enumerate(("x", "y", "z"))
                        ],
                    ),
                    Source(
                        "Gyroscope",
                        signals=[
                            Signal.from_values(
                                gyroscope[:, index],
                                spec=GYROSCOPE,
                                name=f"gyroscope_{axis}",
                                time_axis=AXIS,
                            )
                            for index, axis in enumerate(("x", "y", "z"))
                        ],
                    ),
                ],
                metadata={"experiment_id": experiment_id, "subject_id": subject_id},
            )
            record = dataset.add_record(record)
            split = _split_for_subject(subject_id, train_subjects, test_subjects)
            dataset.add_tasks(
                [_segment_task(record, segment, split, n_samples=accelerometer.shape[0]) for segment in segments]
            )
        return dataset


CONNECTOR = HaptConnector
