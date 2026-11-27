"""Convert official MHEALTH subject logs into lazy, interval-labeled TimeF records."""
# ruff: noqa: ASYNC240, DOC201, DOC501, PLR1702, PLR2004, PLR6301

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile
import zlib

import pyarrow as pa

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError
from timenet.types import Annotation, ClassificationTask, DataSource, TimeInterval, TimeSeriesSpec, ureg
from timenet_connectors.download import Artifact, download_files


MHEALTH_URL = "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip"
"""Official UCI archive URL."""
ARCHIVE_FILENAME = "mhealth-319.zip"
"""Stable cache filename for the official archive."""
ARCHIVE_SHA256 = "16ad0ce709f3f00df18f348610d15bce0884b79e2143f57f446493673f02b8e0"
"""Pinned SHA-256 of the official archive."""
ROOT = "MHEALTHDATASET"
SAMPLE_RATE_HZ = 50
VALUES_PER_ROW = 24
ACTIVITY_VOCABULARY_ID = "mhealth-activity-vocabulary"
_LOG_PATTERN = re.compile(r"mHealth_subject([1-9]|10)\.log$")
_ACTIVITY_LABELS = {
    0: "null_class",
    1: "standing_still",
    2: "sitting_and_relaxing",
    3: "lying_down",
    4: "walking",
    5: "climbing_stairs",
    6: "waist_bends_forward",
    7: "frontal_arm_elevation",
    8: "knees_bending",
    9: "cycling",
    10: "jogging",
    11: "running",
    12: "jump_front_back",
}
_SOURCE = DataSource(data_source_type="wearable", name="MHEALTH Shimmer2 sensors", provider="UCI")
_SPECS = {
    "acceleration": TimeSeriesSpec(
        spec_type="acceleration",
        name="Acceleration",
        unit_value=ureg.meter / ureg.second**2,
        data_source=_SOURCE,
        dtype="float64",
    ),
    "ecg": TimeSeriesSpec(
        spec_type="ecg", name="Electrocardiogram", unit_value=ureg.millivolt, data_source=_SOURCE, dtype="float64"
    ),
    "gyroscope": TimeSeriesSpec(
        spec_type="gyroscope",
        name="Gyroscope angular velocity",
        unit_value=ureg.degree / ureg.second,
        data_source=_SOURCE,
        dtype="float64",
    ),
    "magnetometer_local": TimeSeriesSpec(
        spec_type="magnetometer_local",
        name="Magnetometer (source-local scale)",
        unit_value=ureg.dimensionless,
        data_source=_SOURCE,
        dtype="float64",
    ),
}
_SIGNALS = (
    ("chest_acceleration_x", "acceleration"),
    ("chest_acceleration_y", "acceleration"),
    ("chest_acceleration_z", "acceleration"),
    ("chest_ecg_lead_1", "ecg"),
    ("chest_ecg_lead_2", "ecg"),
    ("left_ankle_acceleration_x", "acceleration"),
    ("left_ankle_acceleration_y", "acceleration"),
    ("left_ankle_acceleration_z", "acceleration"),
    ("left_ankle_gyroscope_x", "gyroscope"),
    ("left_ankle_gyroscope_y", "gyroscope"),
    ("left_ankle_gyroscope_z", "gyroscope"),
    ("left_ankle_magnetometer_x", "magnetometer_local"),
    ("left_ankle_magnetometer_y", "magnetometer_local"),
    ("left_ankle_magnetometer_z", "magnetometer_local"),
    ("right_lower_arm_acceleration_x", "acceleration"),
    ("right_lower_arm_acceleration_y", "acceleration"),
    ("right_lower_arm_acceleration_z", "acceleration"),
    ("right_lower_arm_gyroscope_x", "gyroscope"),
    ("right_lower_arm_gyroscope_y", "gyroscope"),
    ("right_lower_arm_gyroscope_z", "gyroscope"),
    ("right_lower_arm_magnetometer_x", "magnetometer_local"),
    ("right_lower_arm_magnetometer_y", "magnetometer_local"),
    ("right_lower_arm_magnetometer_z", "magnetometer_local"),
)


@dataclass(frozen=True)
class MHealthSource:
    """The safely extracted official MHEALTH source directory."""

    dataset_root: Path


@dataclass(frozen=True)
class _LabelRun:
    """One contiguous run of an exact integer source label."""

    label: int
    start: int
    stop: int


@dataclass(frozen=True)
class _LogInfo:
    """Validated lightweight metadata for one subject-native log file."""

    path: Path
    subject: int
    samples: int
    labels: tuple[_LabelRun, ...]


def _sha256(path: Path) -> str:
    """Return a file SHA-256 without loading the archive into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(name: str) -> bool:
    """Return whether a ZIP member name is a safe relative POSIX path."""
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and "\\" not in name


def _required_members(archive: zipfile.ZipFile) -> tuple[zipfile.ZipInfo, ...]:
    """Return exactly the official README and ten subject logs, rejecting unsafe layouts."""
    expected = {f"{ROOT}/README.txt", *(f"{ROOT}/mHealth_subject{subject}.log" for subject in range(1, 11))}
    found = {info.filename: info for info in archive.infolist() if info.filename in expected}
    if len(found) != len(expected) or any(not _safe_member(name) for name in found):
        missing = sorted(expected - found.keys())
        raise TimeFFormatError(f"MHEALTH archive is missing required members: {missing}")
    if any(info.is_dir() or info.flag_bits & 1 for info in found.values()):
        raise TimeFFormatError("MHEALTH archive has an encrypted or directory required member")
    return tuple(found[name] for name in sorted(expected))


def _crc(path: Path) -> tuple[int, int]:
    """Return the streaming CRC32 and byte count of an extracted source member."""
    checksum = count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            checksum = zlib.crc32(chunk, checksum)
            count += len(chunk)
    return checksum & 0xFFFFFFFF, count


def _validate_cached_members(root: Path, members: tuple[zipfile.ZipInfo, ...]) -> None:
    """Reject cached members that differ from the exact pinned ZIP source bytes."""
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise TimeFFormatError(f"MHEALTH cache root is unsafe: {root}")
    actual = {path.relative_to(root.parent).as_posix(): path for path in root.rglob("*") if path.is_file()}
    expected = {info.filename for info in members}
    if set(actual) != expected:
        raise TimeFFormatError(f"MHEALTH cache root differs from official member set: {root}")
    for info in members:
        if _crc(actual[info.filename]) != (info.CRC, info.file_size):
            raise TimeFFormatError(f"MHEALTH cached member differs from official bytes: {info.filename}")


def _extract_members(archive: Path, cache_dir: Path) -> Path:
    """Safely and atomically extract only the official README and subject log files."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir.is_symlink() or not cache_dir.is_dir():
        raise TimeFFormatError(f"MHEALTH cache directory must be a real directory: {cache_dir}")
    root = cache_dir / ROOT
    try:
        with zipfile.ZipFile(archive) as source:
            members = _required_members(source)
            if root.exists():
                if not root.is_dir():
                    raise TimeFFormatError(f"MHEALTH cache root is incomplete or unsafe: {root}")
                _validate_cached_members(root, members)
                return root
            staging = Path(tempfile.mkdtemp(prefix=".mhealth-", dir=cache_dir)) / ROOT
            for info in members:
                destination = staging / Path(info.filename).relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as target:
                    temporary = Path(target.name)
                    with source.open(info) as origin:
                        while chunk := origin.read(1 << 20):
                            target.write(chunk)
                temporary.replace(destination)
            staging.replace(root)
    except zipfile.BadZipFile as exc:
        raise TimeFFormatError(f"invalid MHEALTH archive: {archive}") from exc
    return root


def _parse_row(path: Path, row: int, line: bytes) -> tuple[list[float], int]:
    """Parse one exact 23-signal MHEALTH row and its integer activity label."""
    try:
        fields = line.decode("ascii").split()
    except UnicodeDecodeError as exc:
        raise TimeFFormatError(f"{path}: row {row} is not ASCII text") from exc
    if len(fields) != VALUES_PER_ROW:
        raise TimeFFormatError(f"{path}: row {row} has {len(fields)} fields; expected {VALUES_PER_ROW}")
    try:
        values = [float(field) for field in fields[:23]]
        label = int(fields[23])
    except ValueError as exc:
        raise TimeFFormatError(f"{path}: row {row} has non-numeric source content") from exc
    if label < 0 or label > 12:
        raise TimeFFormatError(f"{path}: row {row} has unsupported activity label {label}")
    return values, label


def _scan_log(path: Path, subject: int) -> _LogInfo:
    """Validate one source file and collect only its label run boundaries."""
    runs: list[_LabelRun] = []
    previous: int | None = None
    start = 0
    samples = 0
    with path.open("rb") as handle:
        for samples, line in enumerate(handle, start=1):
            _, label = _parse_row(path, samples, line)
            if previous is None:
                previous, start = label, samples - 1
            elif label != previous:
                runs.append(_LabelRun(previous, start, samples - 1))
                previous, start = label, samples - 1
    if samples == 0 or previous is None:
        raise TimeFFormatError(f"{path}: source log is empty")
    runs.append(_LabelRun(previous, start, samples))
    return _LogInfo(path=path, subject=subject, samples=samples, labels=tuple(runs))


def _column_loader(info: _LogInfo, column: int) -> Callable[[], pa.Array]:
    """Return a lazy loader for one source column, preserving float64 and NaN values."""

    def load() -> pa.Array:
        values = [row[column] for row in _cached_record(info.path)]
        if len(values) != info.samples:
            raise TimeFFormatError(f"{info.path}: source length changed after conversion")
        return pa.array(values, type=pa.float64())

    return load


@lru_cache(maxsize=1)
def _cached_record(path: Path) -> tuple[tuple[float, ...], ...]:
    """Parse at most one native source file at a time for all channel loaders."""
    with path.open("rb") as handle:
        return tuple(tuple(_parse_row(path, row, line)[0]) for row, line in enumerate(handle, start=1))


class MHealthConnector(BaseConnector[MHealthSource]):
    """Connector for MHEALTH's full subject-native multimodal sensor recordings."""

    async def download_async(self, cache_dir: Path) -> list[MHealthSource]:
        """Cache the pinned official archive and safely extract the fixed source member set."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        archive = cache_dir / ARCHIVE_FILENAME
        if archive.exists() and _sha256(archive) != ARCHIVE_SHA256:
            raise TimeFFormatError(f"cached MHEALTH archive does not match pinned SHA-256: {archive}")
        if not archive.exists():
            await download_files([Artifact(MHEALTH_URL, archive, sha256=ARCHIVE_SHA256)], skip_existing=False)
        if _sha256(archive) != ARCHIVE_SHA256:
            raise TimeFFormatError(f"downloaded MHEALTH archive does not match pinned SHA-256: {archive}")
        return [MHealthSource(dataset_root=_extract_members(archive, cache_dir))]

    def convert(self, raw_refs: list[MHealthSource]) -> TimeFDataset:
        """Build one lazy 23-signal record and scoped exact-label tasks per subject log."""
        if len(raw_refs) != 1:
            raise TimeFFormatError(f"MHEALTH conversion needs one source root, got {len(raw_refs)}")
        root = raw_refs[0].dataset_root
        infos = []
        for path in sorted(
            root.glob("mHealth_subject*.log"),
            key=lambda item: int(match.group(1)) if (match := _LOG_PATTERN.search(item.name)) else 99,
        ):
            match = _LOG_PATTERN.fullmatch(path.name)
            if match:
                infos.append(_scan_log(path, int(match.group(1))))
        if not infos:
            raise TimeFFormatError(f"{root}: no MHEALTH subject logs found")
        dataset = TimeFDataset(metadata=self.metadata())
        dataset.register_annotations(
            [Annotation(key="activity_vocabulary", value=_ACTIVITY_LABELS, id=ACTIVITY_VOCABULARY_ID)]
        )
        axis = RegularAxis.from_rate_hz(SAMPLE_RATE_HZ)
        for info in infos:
            self._add_record(dataset, info, axis)
        return dataset

    @staticmethod
    def _add_record(dataset: TimeFDataset, info: _LogInfo, axis: RegularAxis) -> None:
        """Attach a native subject record, exact source-label intervals, and scoped tasks."""
        record_id = f"mhealth-subject-{info.subject}"
        series = tuple(
            TimeSeries(
                spec=_SPECS[spec_name],
                signal=signal,
                time_axis=axis,
                loader=_column_loader(info, column),
                source_id=info.path.name,
                time_series_id=f"{record_id}-{signal}",
                n_values=info.samples,
            )
            for column, (signal, spec_name) in enumerate(_SIGNALS)
        )
        record = dataset.add_record(time_series=series, record_id=record_id, subject_ids=(str(info.subject),))
        for run_number, run in enumerate(info.labels, start=1):
            span = TimeInterval.micros(run.start * 20_000, run.stop * 20_000)
            annotation = Annotation(
                key="activity_id", value=run.label, span=span, id=f"{record_id}-activity-{run_number}"
            )
            record.add_annotation(annotation)
            record.add_annotation(
                Annotation(
                    key="activity_name",
                    value=_ACTIVITY_LABELS[run.label],
                    span=span,
                    id=f"{record_id}-activity-name-{run_number}",
                )
            )
            dataset.add_task(
                record,
                ClassificationTask(
                    target=_ACTIVITY_LABELS[run.label],
                    target_schema=ACTIVITY_VOCABULARY_ID,
                    scope=span,
                    id=f"{record_id}-activity-task-{run_number}",
                ),
            )


CONNECTOR = MHealthConnector
