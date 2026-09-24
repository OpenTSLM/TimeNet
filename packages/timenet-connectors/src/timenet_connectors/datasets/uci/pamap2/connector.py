"""Convert the official nested PAMAP2 archive into native-file TimeF records."""

# ruff: noqa: ASYNC240, DOC201, DOC501, PLR2004, PLR6301
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile
import zlib

import pyarrow as pa

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import IrregularAxis
from timenet.errors import TimeFFormatError
from timenet.types import Annotation, ClassificationTask, DataSource, TimeInterval, TimeSeriesSpec, ureg
from timenet_connectors.download import Artifact, download_files


PAMAP2_URL = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
ARCHIVE_FILENAME = "pamap2-231.zip"
ARCHIVE_SHA256 = "76b3580bd5a804121f507717cb498c66a312ef5c774f4a78ac470e6558add352"
ROOT = "PAMAP2_Dataset"
NESTED_ARCHIVE = f"{ROOT}.zip"
ACTIVITY_VOCABULARY_ID = "pamap2-activity-vocabulary"
_ACTIVITY_LABELS = {
    0: "other_transient",
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    9: "watching_tv",
    10: "computer_work",
    11: "car_driving",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    18: "folding_laundry",
    19: "house_cleaning",
    20: "playing_soccer",
    24: "rope_jumping",
}
_SOURCE = DataSource(data_source_type="wearable", name="PAMAP2 Colibri IMU and heart-rate monitor", provider="UCI")
_SPECS = {
    "heart_rate": TimeSeriesSpec("heart_rate", "Heart rate", ureg.bpm, _SOURCE, "float64", nullable=True),
    "temperature": TimeSeriesSpec("temperature", "IMU temperature", ureg.degC, _SOURCE, "float64", nullable=True),
    "acceleration_16g": TimeSeriesSpec(
        "acceleration_16g",
        "Acceleration (±16 g range)",
        ureg.meter / ureg.second**2,
        _SOURCE,
        "float64",
        nullable=True,
    ),
    "acceleration_6g": TimeSeriesSpec(
        "acceleration_6g",
        "Acceleration (±6 g range)",
        ureg.meter / ureg.second**2,
        _SOURCE,
        "float64",
        nullable=True,
    ),
    "gyroscope": TimeSeriesSpec(
        "gyroscope", "Gyroscope angular velocity", ureg.radian / ureg.second, _SOURCE, "float64", nullable=True
    ),
    "magnetometer": TimeSeriesSpec("magnetometer", "Magnetometer", ureg.microtesla, _SOURCE, "float64", nullable=True),
    "orientation_invalid": TimeSeriesSpec(
        "orientation", "Orientation (invalid source field)", ureg.dimensionless, _SOURCE, "float64", nullable=True
    ),
}
_SIGNALS = (
    ("heart_rate", "heart_rate"),
    *tuple(
        (f"{location}_{kind}_{component}", spec)
        for location in ("hand", "chest", "ankle")
        for kind, spec, components in (
            ("temperature", "temperature", ("celsius",)),
            ("acceleration_16g", "acceleration_16g", ("x", "y", "z")),
            ("acceleration_6g", "acceleration_6g", ("x", "y", "z")),
            ("gyroscope", "gyroscope", ("x", "y", "z")),
            ("magnetometer", "magnetometer", ("x", "y", "z")),
            ("orientation_invalid", "orientation_invalid", ("q1", "q2", "q3", "q4")),
        )
        for component in components
    ),
)


@dataclass(frozen=True)
class Pamap2Source:
    """The safely extracted official PAMAP2 source directory."""

    dataset_root: Path


@dataclass(frozen=True)
class _Info:
    """Validated metadata and exact activity runs for one native recording."""

    path: Path
    subject: str
    session: str
    rows: int
    first_us: int
    last_us: int
    runs: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class _CachedRecord:
    """One bounded native recording parsed into shared Arrow value and time arrays."""

    timestamps: pa.Array
    columns: tuple[pa.Array, ...]


def _sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(name: str) -> bool:
    """Return whether a ZIP name is a safe relative POSIX path."""
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and "\\" not in name


def _not_symlink(info: zipfile.ZipInfo) -> bool:
    """Return whether a ZIP member does not encode a symbolic link."""
    return (info.external_attr >> 16) & 0o170000 != 0o120000


def _outer_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Validate the fixed two-member official outer archive."""
    members = [info for info in archive.infolist() if not info.is_dir()]
    names = [info.filename for info in members]
    if set(names) != {NESTED_ARCHIVE, "readme.pdf"} or len(names) != 2:
        raise TimeFFormatError("PAMAP2 outer archive does not match the official two-member layout")
    if any(not _safe_member(info.filename) or info.flag_bits & 1 or not _not_symlink(info) for info in members):
        raise TimeFFormatError("PAMAP2 outer archive contains an unsafe member")
    return {info.filename: info for info in members}


def _inner_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Validate and return the nested source archive's ordinary members."""
    members = [info for info in archive.infolist() if not info.is_dir()]
    names = [info.filename for info in members]
    required = {f"{ROOT}/Protocol/subject{subject}.dat" for subject in range(101, 110)}
    if not members or len(set(names)) != len(names) or not required <= set(names):
        raise TimeFFormatError("PAMAP2 nested archive is incomplete or has duplicate members")
    if any(
        not _safe_member(info.filename)
        or not info.filename.startswith(f"{ROOT}/")
        or info.flag_bits & 1
        or not _not_symlink(info)
        for info in members
    ):
        raise TimeFFormatError("PAMAP2 nested archive contains an unsafe member")
    return {info.filename: info for info in members}


def _crc(path: Path) -> tuple[int, int]:
    """Return a file's streaming CRC32 and byte count."""
    checksum = count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            checksum = zlib.crc32(block, checksum)
            count += len(block)
    return checksum & 0xFFFFFFFF, count


def _validate_existing(root: Path, members: dict[str, zipfile.ZipInfo]) -> None:
    """Reject a changed, incomplete, extra, or symlinked cached extraction."""
    if root.is_symlink() or not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise TimeFFormatError(f"PAMAP2 cache root is unsafe: {root}")
    actual = {path.relative_to(root.parent).as_posix(): path for path in root.rglob("*") if path.is_file()}
    if set(actual) != set(members):
        raise TimeFFormatError("PAMAP2 cached extraction differs from the pinned nested archive")
    for name, info in members.items():
        if _crc(actual[name]) != (info.CRC, info.file_size):
            raise TimeFFormatError(f"PAMAP2 cached extraction has changed: {name}")


def _copy_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, destination: Path) -> None:
    """Copy and CRC-validate one already-validated ZIP member."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    checksum = count = 0
    with archive.open(info) as source, destination.open("xb") as target:
        while block := source.read(1 << 20):
            target.write(block)
            checksum = zlib.crc32(block, checksum)
            count += len(block)
    if (checksum & 0xFFFFFFFF, count) != (info.CRC, info.file_size):
        raise TimeFFormatError(f"PAMAP2 member changed while extracting: {info.filename}")


def _extract_members(archive_path: Path, cache_dir: Path) -> Path:
    """Safely and atomically extract the validated official nested source ZIP."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir.is_symlink() or not cache_dir.is_dir():
        raise TimeFFormatError(f"PAMAP2 cache directory must be a real directory: {cache_dir}")
    staging = Path(tempfile.mkdtemp(prefix=".pamap2-", dir=cache_dir))
    try:
        with zipfile.ZipFile(archive_path) as outer:
            nested_path = staging / NESTED_ARCHIVE
            _copy_member(outer, _outer_members(outer)[NESTED_ARCHIVE], nested_path)
        with zipfile.ZipFile(nested_path) as nested:
            members = _inner_members(nested)
            root = cache_dir / ROOT
            if root.exists():
                _validate_existing(root, members)
                return root
            for info in members.values():
                _copy_member(nested, info, staging / info.filename)
        (staging / ROOT).replace(root)
        return root
    except zipfile.BadZipFile as exc:
        raise TimeFFormatError(f"invalid PAMAP2 archive: {archive_path}") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _row(path: Path, row: int, line: bytes) -> tuple[int, int, tuple[float, ...]]:
    """Parse one 54-column row, retaining native float64 NaN values."""
    try:
        fields = line.decode("ascii").split()
    except UnicodeDecodeError as exc:
        raise TimeFFormatError(f"{path}: row {row} is not ASCII text") from exc
    if len(fields) != 54:
        raise TimeFFormatError(f"{path}: row {row} has {len(fields)} fields; expected 54")
    try:
        timestamp_decimal = Decimal(fields[0]) * 1_000_000
        if timestamp_decimal != timestamp_decimal.to_integral_value():
            raise InvalidOperation
        timestamp, label = int(timestamp_decimal), int(fields[1])
        values = tuple(float(value) for value in fields[2:])
    except (InvalidOperation, ValueError) as exc:
        raise TimeFFormatError(f"{path}: invalid source row {row}") from exc
    if label not in _ACTIVITY_LABELS:
        raise TimeFFormatError(f"{path}: row {row} has unsupported activity label {label}")
    return timestamp, label, values


def _scan(path: Path, subject: str, session: str) -> _Info:
    """Validate a recording and collect only exact activity-run boundaries."""
    runs: list[tuple[int, int, int]] = []
    previous_label: int | None = None
    previous_time: int | None = None
    start_us = first_us = last_us = rows = 0
    with path.open("rb") as handle:
        for rows, line in enumerate(handle, start=1):
            timestamp, label, _ = _row(path, rows, line)
            if previous_time is not None and timestamp <= previous_time:
                raise TimeFFormatError(f"{path}: source timestamp is not strictly increasing at row {rows}")
            if rows == 1:
                first_us = start_us = timestamp
            if previous_label is not None and label != previous_label:
                runs.append((previous_label, start_us, timestamp))
                start_us = timestamp
            previous_label, previous_time, last_us = label, timestamp, timestamp
    if rows == 0 or previous_label is None:
        raise TimeFFormatError(f"{path}: empty PAMAP2 recording")
    runs.append((previous_label, start_us, last_us + 1))
    return _Info(path, subject, session, rows, first_us, last_us, tuple(runs))


@lru_cache(maxsize=1)
def _cached_record(path: Path) -> _CachedRecord:
    """Parse at most one native recording at a time into exact shared Arrow arrays."""
    timestamps: list[int] = []
    columns: list[list[float]] = [[] for _ in _SIGNALS]
    with path.open("rb") as handle:
        for row, line in enumerate(handle, start=1):
            timestamp, _, values = _row(path, row, line)
            timestamps.append(timestamp)
            for column, value in enumerate(values):
                columns[column].append(value)
    return _CachedRecord(
        timestamps=pa.array(timestamps, type=pa.int64()),
        columns=tuple(pa.array(values, type=pa.float64()) for values in columns),
    )


def _loader(info: _Info, column: int) -> Callable[[], pa.Array]:
    """Return one lazy loader backed by the bounded shared native-record cache."""

    def load() -> pa.Array:
        cached = _cached_record(info.path)
        if len(cached.timestamps) != info.rows:
            raise TimeFFormatError(f"{info.path}: source length changed after conversion")
        return cached.columns[column]

    return load


def _times(info: _Info) -> Callable[[], pa.Array]:
    """Return a lazy loader of exact source timestamps in microseconds."""

    def load() -> pa.Array:
        cached = _cached_record(info.path)
        if len(cached.timestamps) != info.rows:
            raise TimeFFormatError(f"{info.path}: source length changed after conversion")
        return cached.timestamps

    return load


class Pamap2Connector(BaseConnector[Pamap2Source]):
    """Convert complete PAMAP2 Protocol and Optional recordings without alteration."""

    async def download_async(self, cache_dir: Path) -> list[Pamap2Source]:
        """Fetch the pinned official outer ZIP and safely extract its nested source ZIP."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        archive = cache_dir / ARCHIVE_FILENAME
        if archive.exists() and _sha256(archive) != ARCHIVE_SHA256:
            raise TimeFFormatError(f"cached PAMAP2 archive does not match pinned SHA-256: {archive}")
        if not archive.exists():
            await download_files([Artifact(PAMAP2_URL, archive, sha256=ARCHIVE_SHA256)], skip_existing=False)
        if _sha256(archive) != ARCHIVE_SHA256:
            raise TimeFFormatError(f"downloaded PAMAP2 archive does not match pinned SHA-256: {archive}")
        return [Pamap2Source(dataset_root=_extract_members(archive, cache_dir))]

    def convert(self, raw_refs: list[Pamap2Source]) -> TimeFDataset:
        """Create one lazy 52-stream record per native source recording."""
        if len(raw_refs) != 1:
            raise TimeFFormatError(f"PAMAP2 conversion needs one source root, got {len(raw_refs)}")
        root, infos = raw_refs[0].dataset_root, self._recordings(raw_refs[0].dataset_root)
        dataset = TimeFDataset(metadata=self.metadata())
        dataset.register_annotations(
            [Annotation(key="activity_vocabulary", value=_ACTIVITY_LABELS, id=ACTIVITY_VOCABULARY_ID)]
        )
        for info in infos:
            self._add_record(dataset, info, root)
        return dataset

    @staticmethod
    def _recordings(root: Path) -> list[_Info]:
        """Return validated Protocol and Optional native recordings in source order."""
        infos: list[_Info] = []
        for session in ("Protocol", "Optional"):
            directory = root / session
            if directory.is_symlink() or not directory.is_dir():
                if session == "Protocol":
                    raise TimeFFormatError(f"{root}: missing PAMAP2 Protocol directory")
                continue
            for path in sorted(directory.glob("subject*.dat")):
                subject = path.stem.removeprefix("subject")
                if path.is_symlink() or not path.is_file() or not subject.isdigit():
                    raise TimeFFormatError(f"{path}: invalid PAMAP2 source filename")
                infos.append(_scan(path, subject, session.lower()))
        if not infos:
            raise TimeFFormatError(f"{root}: no PAMAP2 Protocol or Optional recordings")
        return infos

    @staticmethod
    def _add_record(dataset: TimeFDataset, info: _Info, root: Path) -> None:
        """Attach one record, exact source-label annotations, and scoped tasks."""
        record_id, axis = (
            f"pamap2-{info.session}-subject-{info.subject}",
            IrregularAxis(first_us=info.first_us, last_us=info.last_us),
        )
        series = tuple(
            TimeSeries(
                spec=_SPECS[spec_name],
                signal=signal,
                time_axis=axis,
                loader=_loader(info, column),
                time_offsets_loader=_times(info),
                source_id=info.path.relative_to(root).as_posix(),
                time_series_id=f"{record_id}-{signal}",
                n_values=info.rows,
            )
            for column, (signal, spec_name) in enumerate(_SIGNALS)
        )
        record = dataset.add_record(time_series=series, record_id=record_id, subject_ids=(info.subject,))
        record.add_annotation(Annotation(key="session", value=info.session, id=f"{record_id}-session"))
        for number, (label, start, stop) in enumerate(info.runs, start=1):
            span = TimeInterval.micros(start, stop)
            record.add_annotation(
                Annotation(key="activity_id", value=label, span=span, id=f"{record_id}-activity-{number}")
            )
            record.add_annotation(
                Annotation(
                    key="activity_name",
                    value=_ACTIVITY_LABELS[label],
                    span=span,
                    id=f"{record_id}-activity-name-{number}",
                )
            )
            dataset.add_task(
                record,
                ClassificationTask(
                    target=_ACTIVITY_LABELS[label],
                    target_schema=ACTIVITY_VOCABULARY_ID,
                    scope=span,
                    id=f"{record_id}-activity-task-{number}",
                ),
            )


CONNECTOR = Pamap2Connector
