"""Convert native CAPTURE-24 participant CSV.GZ files into lazy TimeF records."""

from __future__ import annotations

from collections.abc import Callable
import csv
from dataclasses import dataclass
import gzip
import hashlib
from pathlib import Path
import shutil
import tempfile
import zipfile
import zlib

import numpy as np
import pyarrow as pa

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError
from timenet.types import Annotation, ClassificationTask, DataSource, TimeInterval, TimeSeriesSpec, ureg
from timenet_connectors.download import Artifact, download_files


CAPTURE24_URL = "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001/files/rpr76f381b"
ARCHIVE_FILENAME = "capture24.zip"
ARCHIVE_SHA256 = "69740c22d3e000367988373336dc6486c10fe3f3f929811fd66e4d84861e40e2"
_ROOT = "capture24"
_CSV_FIELDS = ("time", "x", "y", "z", "annotation")
_PARTICIPANT_FIELDS = ("pid", "age", "sex")
_METADATA_FILES = {"metadata.csv", "annotation-label-dictionary.csv"}
_SYMLINK_MODE = 0o120000
_DICTIONARY_MAP_COUNT = 6
_AXIS = RegularAxis.from_rate_hz(100)
_SOURCE = DataSource(data_source_type="wearable", name="Axivity AX3", provider="University of Oxford")
_SPECS = {
    "acceleration": TimeSeriesSpec("acceleration", "Acceleration", ureg.standard_gravity, _SOURCE, "float64"),
    "text": TimeSeriesSpec("source_text", "Source text", ureg.dimensionless, _SOURCE, "str"),
}


@dataclass(frozen=True)
class Capture24Participant:
    """One safely extracted participant recording and its source metadata."""

    path: Path
    participant_id: str
    age: str
    sex: str
    annotation_dictionary: dict[str, dict[str, str]]


@dataclass
class _ParticipantCache:
    """Materialize at most one participant's parsed native columns on first use."""

    path: Path | None = None
    values: dict[str, object] | None = None

    def load(self, path: Path) -> dict[str, object]:
        """Return validated source columns, parsing the compressed CSV once.

        Raises:
            TimeFFormatError: If acceleration values are invalid.
        """
        if self.path != path:
            self.path = path
            self.values = None
        if self.values is None:
            rows = _csv_rows(path)
            try:
                acceleration = {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in "xyz"}
            except (KeyError, ValueError) as exc:
                raise TimeFFormatError(f"{path}: invalid acceleration value") from exc
            if any(not np.isfinite(values).all() for values in acceleration.values()):
                raise TimeFFormatError(f"{path}: non-finite acceleration value")
            self.values = {
                **acceleration,
                "time": [row["time"] for row in rows],
                "annotation": [row["annotation"] for row in rows],
            }
        return self.values


_PARTICIPANT_CACHE = _ParticipantCache()


def _sha256(path: Path) -> str:
    """Return the streaming SHA-256 digest of a source artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_rows(path: Path) -> list[dict[str, str]]:
    """Read validated participant source rows.

    Returns:
        The native rows in source order.

    Raises:
        TimeFFormatError: If the native CSV header is malformed or empty.
    """
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != set(_CSV_FIELDS):
        raise TimeFFormatError(f"{path}: expected exactly {', '.join(_CSV_FIELDS)}")
    return rows


def _selected_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Return safe participant and metadata members from the official ZIP.

    Raises:
        TimeFFormatError: If the selected archive layout is unsafe or incomplete.
    """
    selected: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        name = Path(info.filename)
        allowed = name.name in _METADATA_FILES or (name.name.startswith("P") and name.suffixes == [".csv", ".gz"])
        if info.is_dir() or name.is_absolute() or ".." in name.parts or name.parts[:1] != (_ROOT,):
            continue
        if not allowed:
            continue
        if info.filename in selected or (info.external_attr >> 16) & 0o170000 == _SYMLINK_MODE:
            raise TimeFFormatError(f"unsafe CAPTURE-24 archive member: {info.filename}")
        selected[info.filename] = info
    required = {f"{_ROOT}/{name}" for name in _METADATA_FILES}
    if not required <= set(selected) or not any(Path(name).name.startswith("P") for name in selected):
        raise TimeFFormatError("CAPTURE-24 archive lacks participant or metadata files")
    return selected


def _member_crc(path: Path) -> tuple[int, int]:
    """Return an extracted member's CRC32 and byte count."""
    checksum = count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            checksum = zlib.crc32(block, checksum)
            count += len(block)
    return checksum & 0xFFFFFFFF, count


def _validate_cache(root: Path, members: dict[str, zipfile.ZipInfo]) -> None:
    """Reject a changed, extra, missing, or symlinked cached extraction.

    Raises:
        TimeFFormatError: If the cache differs from the pinned ZIP members.
    """
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise TimeFFormatError("CAPTURE-24 extraction contains a symlink")
    actual = {path.relative_to(root.parent).as_posix(): path for path in root.rglob("*") if path.is_file()}
    if set(actual) != set(members):
        raise TimeFFormatError("CAPTURE-24 extraction differs from pinned ZIP members")
    for name, info in members.items():
        if _member_crc(actual[name]) != (info.CRC, info.file_size):
            raise TimeFFormatError(f"CAPTURE-24 extraction changed: {name}")


def _safe_extract(archive_path: Path, cache_dir: Path) -> Path:
    """Extract the selected pinned ZIP members only once.

    Returns:
        The validated extraction root.

    Raises:
        TimeFFormatError: If the archive or existing cache is unsafe or changed.
    """
    with zipfile.ZipFile(archive_path) as archive:
        members = _selected_members(archive)
    root = cache_dir / _ROOT
    if root.exists():
        _validate_cache(root, members)
        return root
    staging = Path(tempfile.mkdtemp(prefix=".capture24-", dir=cache_dir))
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for name, info in members.items():
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1 << 20)
                if _member_crc(target) != (info.CRC, info.file_size):
                    raise TimeFFormatError(f"CAPTURE-24 member changed while extracting: {name}")
        (staging / _ROOT).replace(root)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return root


def _annotation_runs(rows: list[dict[str, str]]) -> list[tuple[str, int, int]]:
    """Return contiguous source annotation intervals as sample offsets."""
    runs: list[tuple[str, int, int]] = []
    start = 0
    label = rows[0]["annotation"]
    for index, row in enumerate(rows[1:], 1):
        if row["annotation"] != label:
            runs.append((label, start, index))
            start = index
            label = row["annotation"]
    runs.append((label, start, len(rows)))
    return runs


def _summary(path: Path) -> tuple[int, list[tuple[str, int, int]]]:
    """Stream a participant once to count samples and annotation boundaries.

    Returns:
        The source row count and its contiguous annotation intervals.

    Raises:
        TimeFFormatError: If the source CSV header is invalid or contains no rows.
    """
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(reader.fieldnames) != set(_CSV_FIELDS):
            raise TimeFFormatError(f"{path}: expected exactly {', '.join(_CSV_FIELDS)}")
        first = next(reader, None)
        if first is None:
            raise TimeFFormatError(f"{path}: expected at least one source row")
        label = first["annotation"]
        start = 0
        count = 1
        runs: list[tuple[str, int, int]] = []
        for row in reader:
            if row["annotation"] != label:
                runs.append((label, start, count))
                start = count
                label = row["annotation"]
            count += 1
    runs.append((label, start, count))
    return count, runs


def _participant_refs(root: Path) -> list[Capture24Participant]:
    """Build participant references from source metadata and all dictionary mappings.

    Returns:
        One source reference per participant CSV.GZ.

    Raises:
        TimeFFormatError: If metadata or all six source dictionary mappings are absent.
    """
    with (root / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        metadata = {row["pid"]: row for row in csv.DictReader(handle)}
    with (root / "annotation-label-dictionary.csv").open(newline="", encoding="utf-8") as handle:
        dictionaries = {
            row["annotation"]: {key: value for key, value in row.items() if key != "annotation"}
            for row in csv.DictReader(handle)
        }
    if not metadata or not dictionaries or any(len(value) != _DICTIONARY_MAP_COUNT for value in dictionaries.values()):
        raise TimeFFormatError("CAPTURE-24 metadata or six-column annotation dictionary is incomplete")
    refs = []
    for path in sorted(root.glob("P*.csv.gz")):
        participant_id = path.name.removesuffix(".csv.gz")
        row = metadata.get(participant_id)
        if row is None or set(row) != set(_PARTICIPANT_FIELDS):
            raise TimeFFormatError(f"CAPTURE-24 missing native metadata for {participant_id}")
        refs.append(Capture24Participant(path, participant_id, row["age"], row["sex"], dictionaries))
    if not refs:
        raise TimeFFormatError("CAPTURE-24 extraction contains no participant CSV files")
    return refs


class Capture24Connector(BaseConnector[Capture24Participant]):
    """Connector for CAPTURE-24's whole-participant native wrist recordings."""

    async def download_async(self, cache_dir: Path) -> list[Capture24Participant]:  # noqa: PLR6301
        """Fetch the pinned archive and return validated native participant references.

        Returns:
            One reference per participant CSV.GZ source file.

        Raises:
            TimeFFormatError: If archive or extraction integrity validation fails.
        """
        cache_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
        archive = cache_dir / ARCHIVE_FILENAME
        if archive.exists() and _sha256(archive) != ARCHIVE_SHA256:
            raise TimeFFormatError(f"cached CAPTURE-24 archive hash mismatch: {archive}")
        if not archive.exists():
            await download_files([Artifact(CAPTURE24_URL, archive, sha256=ARCHIVE_SHA256)], skip_existing=False)
        if _sha256(archive) != ARCHIVE_SHA256:
            raise TimeFFormatError(f"downloaded CAPTURE-24 archive hash mismatch: {archive}")
        return _participant_refs(_safe_extract(archive, cache_dir))

    def convert(self, raw_refs: list[Capture24Participant]) -> TimeFDataset:
        """Build one whole-participant record retaining native source columns and labels.

        Returns:
            A dataset with three acceleration and two text columns per participant.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        for ref in raw_refs:
            count, runs = _summary(ref.path)
            record_id = f"capture24-{ref.participant_id}"

            def loader(key: str, source_path: Path = ref.path) -> Callable[[], pa.Array]:
                return lambda: pa.array(
                    _PARTICIPANT_CACHE.load(source_path)[key],
                    type=pa.float64() if key in {"x", "y", "z"} else pa.string(),
                )

            series = tuple(
                TimeSeries(
                    spec=_SPECS["acceleration"],
                    signal=key,
                    time_axis=_AXIS,
                    loader=loader(key),
                    source_id=ref.path.name,
                    time_series_id=f"{record_id}-{key}",
                    n_values=count,
                )
                for key in ("x", "y", "z")
            ) + tuple(
                TimeSeries(
                    spec=_SPECS["text"],
                    signal=key,
                    time_axis=_AXIS,
                    loader=loader(key),
                    source_id=ref.path.name,
                    time_series_id=f"{record_id}-{key}",
                    n_values=count,
                )
                for key in ("time", "annotation")
            )
            record = dataset.add_record(time_series=series, subject_ids=(ref.participant_id,), record_id=record_id)
            record.add_annotations(
                (
                    Annotation(key="age", value=ref.age, id=f"{record_id}-age"),
                    Annotation(key="sex", value=ref.sex, id=f"{record_id}-sex"),
                    Annotation(
                        key="annotation_label_dictionary", value=ref.annotation_dictionary, id=f"{record_id}-dictionary"
                    ),
                    Annotation(
                        key="timestamp_timezone", value="naive_source_text", id=f"{record_id}-timestamp-timezone"
                    ),
                )
            )
            for index, (label, start, end) in enumerate(runs):
                dataset.add_task(
                    record,
                    ClassificationTask(
                        target=label,
                        scope=TimeInterval.seconds(start / 100, end / 100),
                        id=f"{record_id}-annotation-{index}",
                    ),
                )
        return dataset


CONNECTOR = Capture24Connector
