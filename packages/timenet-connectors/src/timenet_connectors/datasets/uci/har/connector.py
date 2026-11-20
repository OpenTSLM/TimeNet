"""Convert UCI-HAR's fixed inertial-signal windows into TimeF records."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import shutil
import tempfile
import zipfile
import zlib

import pyarrow as pa

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError
from timenet.types import Annotation, ClassificationTask, DataSource, TimeSeriesSpec, ureg
from timenet_connectors.download import Artifact, download_files


UCI_HAR_URL = "https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip"
"""Official UCI archive URL for dataset 240."""

ARCHIVE_FILENAME = "uci-240.zip"
"""Stable cache name for the preserved official outer archive."""

ARCHIVE_SHA256 = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
"""SHA-256 of the official outer archive."""

ACTIVITY_VOCABULARY_ID = "uci-har-vocabulary-activity"
"""Registered annotation id used by every activity-classification task."""

_DATASET_DIR = "UCI HAR Dataset"
_INNER_ARCHIVE = f"{_DATASET_DIR}.zip"
_SPLITS = ("train", "test")
_WINDOW_VALUES = 128
_MAX_SUBJECT_ID = 30
_ACTIVITIES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}
_SIGNALS = (
    ("total_acc_x", "total_acceleration"),
    ("total_acc_y", "total_acceleration"),
    ("total_acc_z", "total_acceleration"),
    ("body_acc_x", "body_acceleration"),
    ("body_acc_y", "body_acceleration"),
    ("body_acc_z", "body_acceleration"),
    ("body_gyro_x", "body_gyroscope"),
    ("body_gyro_y", "body_gyroscope"),
    ("body_gyro_z", "body_gyroscope"),
)
_SOURCE = DataSource(data_source_type="smartphone", name="UCI HAR inertial signals", provider="UCI")
_SPECS = {
    "total_acceleration": TimeSeriesSpec(
        spec_type="total_acceleration",
        name="Total acceleration",
        unit_value=ureg.standard_gravity,
        data_source=_SOURCE,
        dtype="float64",
    ),
    "body_acceleration": TimeSeriesSpec(
        spec_type="body_acceleration",
        name="Body acceleration",
        unit_value=ureg.standard_gravity,
        data_source=_SOURCE,
        dtype="float64",
    ),
    "body_gyroscope": TimeSeriesSpec(
        spec_type="body_gyroscope",
        name="Body gyroscope angular velocity",
        unit_value=ureg.radian / ureg.second,
        data_source=_SOURCE,
        dtype="float64",
    ),
}


@dataclass(frozen=True)
class UciHarSource:
    """The extracted UCI-HAR directory passed from download to conversion."""

    dataset_root: Path


@dataclass(frozen=True)
class _SignalRows:
    """Binary line offsets for one source signal file, read lazily by record row."""

    path: Path
    offsets: tuple[int, ...]

    def loader_for(self, row: int) -> Callable[[], pa.Array]:
        """Return a lazy Arrow loader for one source row.

        Returns:
            A callable that reads the selected row only when TimeF requests it.
        """
        return lambda: self._load_row(row)

    def _load_row(self, row: int) -> pa.Array:
        """Seek, parse, and validate one 128-value source line as float64.

        Returns:
            The source row as a float64 Arrow array.

        Raises:
            TimeFFormatError: If the requested row is absent or malformed.
        """
        try:
            offset = self.offsets[row]
        except IndexError as exc:
            raise TimeFFormatError(f"{self.path}: source row {row + 1} is absent") from exc
        with self.path.open("rb") as handle:
            handle.seek(offset)
            line = handle.readline()
        values = _parse_signal_line(self.path, row + 1, line, keep_values=True)
        if values is None:  # keep_values above is a local contract, retained for type narrowing.
            raise TimeFFormatError(f"{self.path}: source row {row + 1} could not be loaded")
        return pa.array(values, type=pa.float64())


def _required_members() -> tuple[str, ...]:
    """Return the exact nested-archive members needed by this connector."""
    members = [f"{_DATASET_DIR}/activity_labels.txt"]
    for split in _SPLITS:
        members.extend((f"{_DATASET_DIR}/{split}/subject_{split}.txt", f"{_DATASET_DIR}/{split}/y_{split}.txt"))
        members.extend(f"{_DATASET_DIR}/{split}/Inertial Signals/{signal}_{split}.txt" for signal, _ in _SIGNALS)
    return tuple(members)


def _sha256(path: Path) -> str:
    """Hash a file without loading an archive into memory.

    Returns:
        The lowercase SHA-256 hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_archive(path: Path) -> None:
    """Reject an incomplete or stale cached outer archive.

    Raises:
        TimeFFormatError: If the archive does not match the pinned SHA-256.
    """
    actual = _sha256(path)
    if actual != ARCHIVE_SHA256:
        raise TimeFFormatError(
            f"SHA-256 mismatch for cached UCI-HAR archive {path}: expected {ARCHIVE_SHA256}, got {actual}"
        )


def _parse_signal_line(path: Path, row: int, line: bytes, *, keep_values: bool = False) -> list[float] | None:
    """Validate one source line and retain floats only when a lazy loader needs them.

    Returns:
        Parsed values for a lazy load, otherwise ``None`` after validation.

    Raises:
        TimeFFormatError: If the source row has the wrong shape or invalid numeric values.
    """
    try:
        tokens = line.decode("ascii").split()
    except UnicodeDecodeError as exc:
        raise TimeFFormatError(f"{path}: source row {row} is not ASCII numeric text") from exc
    if len(tokens) != _WINDOW_VALUES:
        raise TimeFFormatError(f"{path}: source row {row} has {len(tokens)} values; expected {_WINDOW_VALUES}")
    try:
        values = [float(token) for token in tokens] if keep_values else None
    except ValueError as exc:
        raise TimeFFormatError(f"{path}: source row {row} contains a non-numeric value") from exc
    if values is not None:
        if not all(math.isfinite(value) for value in values):
            raise TimeFFormatError(f"{path}: source row {row} contains a non-finite value")
    else:
        try:
            if not all(math.isfinite(float(token)) for token in tokens):
                raise TimeFFormatError(f"{path}: source row {row} contains a non-finite value")
        except ValueError as exc:
            raise TimeFFormatError(f"{path}: source row {row} contains a non-numeric value") from exc
    return values


def _scan_signal(path: Path) -> _SignalRows:
    """Validate each source line once and retain binary offsets for lazy loaders.

    Returns:
        The source path and its validated binary line offsets.

    Raises:
        TimeFFormatError: If the signal file is empty or a row is malformed.
    """
    offsets: list[int] = []
    with path.open("rb") as handle:
        while line := handle.readline():
            offsets.append(handle.tell() - len(line))
            _parse_signal_line(path, len(offsets), line)
    if not offsets:
        raise TimeFFormatError(f"{path}: signal file is empty")
    return _SignalRows(path=path, offsets=tuple(offsets))


def _read_integer_rows(path: Path, field: str) -> tuple[int, ...]:
    """Read non-empty, one-integer-per-row source metadata.

    Returns:
        Integer values in their source-row order.

    Raises:
        TimeFFormatError: If the file is empty or any row is not one integer.
    """
    values: list[int] = []
    with path.open(encoding="ascii") as handle:
        for row, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                raise TimeFFormatError(f"{path}: {field} row {row} is empty")
            try:
                values.append(int(text))
            except ValueError as exc:
                raise TimeFFormatError(f"{path}: {field} row {row} is not an integer") from exc
    if not values:
        raise TimeFFormatError(f"{path}: {field} file is empty")
    return tuple(values)


def _read_activities(path: Path) -> dict[int, str]:
    """Read the activity table and require its exact official six-label mapping.

    Returns:
        The official activity ids mapped to their exact source labels.

    Raises:
        TimeFFormatError: If the table differs from the official six-label mapping.
    """
    labels: dict[int, str] = {}
    with path.open(encoding="ascii") as handle:
        for row, line in enumerate(handle, start=1):
            parts = line.split()
            if len(parts) != 2:  # noqa: PLR2004 (the source table has exactly id and label columns)
                raise TimeFFormatError(f"{path}: activity row {row} must contain an id and label")
            try:
                activity_id = int(parts[0])
            except ValueError as exc:
                raise TimeFFormatError(f"{path}: activity row {row} has a non-integer id") from exc
            if activity_id in labels:
                raise TimeFFormatError(f"{path}: activity id {activity_id} is declared twice")
            labels[activity_id] = parts[1]
    if labels != _ACTIVITIES:
        raise TimeFFormatError(f"{path}: activity mapping differs from the official UCI-HAR labels")
    return labels


@contextmanager
def _nested_archive(outer: Path, cache_dir: Path) -> Iterator[zipfile.ZipFile]:
    """Yield the official nested archive from a streamed temporary file.

    Yields:
        The opened nested official release archive.

    Raises:
        TimeFFormatError: If either archive is malformed or lacks the nested release member.
    """
    temporary: Path | None = None
    try:
        with zipfile.ZipFile(outer) as archive:
            try:
                with (
                    archive.open(_INNER_ARCHIVE) as nested,
                    tempfile.NamedTemporaryFile("wb", dir=cache_dir, suffix=".zip", delete=False) as handle,
                ):
                    temporary = Path(handle.name)
                    shutil.copyfileobj(nested, handle, length=1 << 20)
            except KeyError as exc:
                raise TimeFFormatError(f"{outer}: missing nested member {_INNER_ARCHIVE!r}") from exc
        with zipfile.ZipFile(temporary) as archive:
            yield archive
    except zipfile.BadZipFile as exc:
        raise TimeFFormatError(f"{outer}: invalid UCI-HAR zip archive") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _expected_members(archive: zipfile.ZipFile) -> dict[str, tuple[int, int]]:
    """Return required member CRC32/size pairs, rejecting absent or duplicate source members.

    Returns:
        Required member names mapped to their CRC32 and uncompressed byte size.

    Raises:
        TimeFFormatError: If a required member is absent or appears more than once.
    """
    required = set(_required_members())
    infos = [info for info in archive.infolist() if info.filename in required]
    by_name = {info.filename: info for info in infos}
    if len(by_name) != len(infos):
        raise TimeFFormatError("nested UCI-HAR archive declares a required member more than once")
    missing = required - by_name.keys()
    if missing:
        raise TimeFFormatError(f"nested UCI-HAR archive is missing required member {min(missing)!r}")
    return {name: (info.CRC, info.file_size) for name, info in by_name.items()}


def _member_destination(cache_dir: Path, member: str) -> Path:
    """Return one fixed member's destination, rejecting any symlink in its path.

    Returns:
        The member path below the cache root.

    Raises:
        TimeFFormatError: If the cache root or a member path component is a symlink.
    """
    destination = cache_dir / member
    current = cache_dir
    if current.is_symlink():
        raise TimeFFormatError(f"UCI-HAR cache root {cache_dir} must not be a symlink")
    for part in destination.relative_to(cache_dir).parts:
        current /= part
        if current.is_symlink():
            raise TimeFFormatError(f"UCI-HAR cache member path contains symlink {current}")
    return destination


def _file_crc32(path: Path) -> tuple[int, int]:
    """Return a file's CRC32 and byte length without loading it into memory.

    Returns:
        The unsigned CRC32 and total byte length.
    """
    checksum = 0
    length = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            checksum = zlib.crc32(chunk, checksum)
            length += len(chunk)
    return checksum & 0xFFFFFFFF, length


def _cached_members_match(cache_dir: Path, expected: dict[str, tuple[int, int]]) -> bool:
    """Return whether every extracted source member matches the pinned nested archive safely.

    Returns:
        ``True`` when every required cached member has its expected bytes.

    """
    for member, state in expected.items():
        destination = _member_destination(cache_dir, member)
        if not destination.is_file() or _file_crc32(destination) != state:
            return False
    return True


class UciHarConnector(BaseConnector[UciHarSource]):
    """Connector for UCI-HAR's 128-sample smartphone sensor windows."""

    async def download_async(self, cache_dir: Path) -> list[UciHarSource]:
        """Cache the official archive and selectively extract its required nested members.

        Returns:
            The one extracted UCI-HAR source root.

        """
        cache_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 (bounded local cache setup)
        outer = cache_dir / ARCHIVE_FILENAME
        if outer.exists():
            _check_archive(outer)
        if not outer.exists():
            await download_files([Artifact(UCI_HAR_URL, outer, sha256=ARCHIVE_SHA256)], skip_existing=False)
        _check_archive(outer)
        with _nested_archive(outer, cache_dir) as nested_archive:
            expected = _expected_members(nested_archive)
            if not _cached_members_match(cache_dir, expected):
                self._extract_required(nested_archive, cache_dir, expected)
        return [UciHarSource(dataset_root=cache_dir / _DATASET_DIR)]

    @staticmethod
    def _extract_required(
        nested_archive: zipfile.ZipFile,
        cache_dir: Path,
        expected: dict[str, tuple[int, int]],
    ) -> None:
        """Atomically replace only fixed safe members with bytes verified against the nested archive.

        Raises:
            TimeFFormatError: If a member path is unsafe or extracted bytes fail validation.
        """
        destinations = {member: _member_destination(cache_dir, member) for member in expected}
        for destination in destinations.values():
            destination.parent.mkdir(parents=True, exist_ok=True)
            _member_destination(cache_dir, str(destination.relative_to(cache_dir)))
        for member, destination in destinations.items():
            temporary: Path | None = None
            try:
                with (
                    nested_archive.open(member) as source,
                    tempfile.NamedTemporaryFile(
                        "wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
                    ) as target,
                ):
                    temporary = Path(target.name)
                    shutil.copyfileobj(source, target, length=1 << 20)
                if _file_crc32(temporary) != expected[member]:
                    raise TimeFFormatError(f"nested UCI-HAR member {member!r} changed while extracting")
                _member_destination(cache_dir, member)
                temporary.replace(destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def convert(self, raw_refs: list[UciHarSource]) -> TimeFDataset:
        """Validate source alignment and build lazy float64 series for every source window.

        Returns:
            One TimeF record and classification task for every UCI-HAR source row.

        Raises:
            TimeFFormatError: If source metadata, split membership, or signal rows are inconsistent.
        """
        if len(raw_refs) != 1:
            raise TimeFFormatError(f"UCI-HAR conversion needs one source root, got {len(raw_refs)}")
        root = raw_refs[0].dataset_root
        activities = _read_activities(root / "activity_labels.txt")
        dataset = TimeFDataset(metadata=self.metadata())
        dataset.register_annotations(
            [
                Annotation(
                    key="activity_vocabulary",
                    value=list(_ACTIVITIES.values()),
                    description="The exact activity labels written by UCI-HAR.",
                    id=ACTIVITY_VOCABULARY_ID,
                )
            ]
        )
        seen_subjects: set[int] = set()
        for split in _SPLITS:
            self._convert_split(dataset, root, split, activities, seen_subjects)
        return dataset

    @staticmethod
    def _convert_split(
        dataset: TimeFDataset,
        root: Path,
        split: str,
        activities: dict[int, str],
        seen_subjects: set[int],
    ) -> None:
        """Validate one frozen source split and attach its records and labels.

        Raises:
            TimeFFormatError: If source rows, labels, subjects, or split membership are invalid.
        """
        split_root = root / split
        subjects = _read_integer_rows(split_root / f"subject_{split}.txt", "subject")
        labels = _read_integer_rows(split_root / f"y_{split}.txt", "activity")
        if len(subjects) != len(labels):
            raise TimeFFormatError(f"{split_root}: subject and activity files have different row counts")
        if any(subject < 1 or subject > _MAX_SUBJECT_ID for subject in subjects):
            raise TimeFFormatError(f"{split_root}: subject ids must be in the inclusive range 1..30")
        overlap = seen_subjects.intersection(subjects)
        if overlap:
            raise TimeFFormatError(f"{split_root}: subject ids overlap another split: {min(overlap)}")
        seen_subjects.update(subjects)
        if any(label not in activities for label in labels):
            raise TimeFFormatError(f"{split_root}: unknown activity label in y_{split}.txt")

        signals = {
            signal: _scan_signal(split_root / "Inertial Signals" / f"{signal}_{split}.txt") for signal, _ in _SIGNALS
        }
        for signal, rows in signals.items():
            if len(rows.offsets) != len(subjects):
                raise TimeFFormatError(
                    f"{rows.path}: {len(rows.offsets)} rows disagree with {len(subjects)} metadata rows for {signal}"
                )

        axis = RegularAxis.from_rate_hz(50)
        for index, (subject, activity_id) in enumerate(zip(subjects, labels, strict=True), start=1):
            record_id = f"uci-har-{split}-{index}"
            series = tuple(
                TimeSeries(
                    spec=_SPECS[spec_name],
                    signal=signal,
                    time_axis=axis,
                    loader=signals[signal].loader_for(index - 1),
                    source_id=f"{split}/Inertial Signals/{signal}_{split}.txt",
                    time_series_id=f"{record_id}-{signal}",
                    n_values=_WINDOW_VALUES,
                )
                for signal, spec_name in _SIGNALS
            )
            record = dataset.add_record(time_series=series, record_id=record_id, subject_ids=(str(subject),))
            record.add_annotations(
                (
                    Annotation(key="activity_id", value=activity_id, id=f"{record_id}-activity-id"),
                    Annotation(key="split", value=split, id=f"{record_id}-split"),
                    Annotation(key="source_row", value=index, id=f"{record_id}-source-row"),
                    Annotation(key="subject_id", value=subject, id=f"{record_id}-subject-id"),
                )
            )
            dataset.add_task(
                record,
                ClassificationTask(
                    target=activities[activity_id],
                    target_schema=ACTIVITY_VOCABULARY_ID,
                    id=f"{record_id}-activity-task",
                ),
            )


CONNECTOR = UciHarConnector
