"""Convert the Shoaib native participant CSV files into lazy TimeF records."""

from __future__ import annotations

from collections.abc import Callable
import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pyarrow as pa

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError
from timenet.types import ClassificationTask, DataSource, TimeInterval, TimeSeriesSpec, ureg
from timenet_connectors.download import Artifact, download_files


URL = "https://www.utwente.nl/en/eemcs/ps/dataset-folder/sensors-activity-recognition-dataset-shoaib.rar"
ARCHIVE = "shoaib.rar"
SHA256 = "61d5dcc5ed9c78321a983667c608e4fcf594d3edab7d1cfc7fffae66218e05c2"
POSITIONS = ("left_pocket", "right_pocket", "wrist", "upper_arm", "belt")
FIELDS = ("Ax", "Ay", "Az", "Lx", "Ly", "Lz", "Gx", "Gy", "Gz", "Mx", "My", "Mz")
_ROW_FIELDS = 70
_MEMBERS = frozenset({f"Participant_{index}.csv" for index in range(1, 11)} | {"Readme.txt"})
SOURCE = DataSource(data_source_type="smartphone", name="Samsung Galaxy SII", provider="University of Twente")
AXIS = RegularAxis.from_rate_hz(50)
SPECS = {
    "A": TimeSeriesSpec(
        spec_type="acceleration",
        name="Acceleration",
        unit_value=ureg.meter / ureg.second**2,
        data_source=SOURCE,
        dtype="float64",
    ),
    "L": TimeSeriesSpec(
        spec_type="linear_acceleration",
        name="Linear acceleration",
        unit_value=ureg.meter / ureg.second**2,
        data_source=SOURCE,
        dtype="float64",
    ),
    "G": TimeSeriesSpec(
        spec_type="angular_velocity",
        name="Angular velocity",
        unit_value=ureg.radian / ureg.second,
        data_source=SOURCE,
        dtype="float64",
    ),
    "M": TimeSeriesSpec(
        spec_type="magnetic_field",
        name="Magnetic field",
        unit_value=ureg.microtesla,
        data_source=SOURCE,
        dtype="float64",
    ),
    "T": TimeSeriesSpec(
        spec_type="source_timestamp",
        name="Original timestamp token",
        unit_value=ureg.dimensionless,
        data_source=SOURCE,
        dtype="str",
    ),
    "Y": TimeSeriesSpec(
        spec_type="activity_label",
        name="Original activity label",
        unit_value=ureg.dimensionless,
        data_source=SOURCE,
        dtype="str",
    ),
}


@dataclass(frozen=True)
class ShoaibParticipant:
    """One safely extracted participant CSV."""

    path: Path
    participant: str


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        header = next(reader, None)
        if header is None or len(header) != _ROW_FIELDS:
            raise TimeFFormatError(f"{path}: expected two headers with 70 fields")
        for row_number, row in enumerate(reader, 3):
            if len(row) != _ROW_FIELDS:
                raise TimeFFormatError(f"{path}:{row_number}: expected 70 fields")
            yield row_number, row


def _summary(path: Path) -> tuple[int, list[tuple[str, int, int]]]:
    labels = [(row[69], index) for index, (_, row) in enumerate(_rows(path))]
    count = len(labels)
    if not count:
        raise TimeFFormatError(f"{path}: has no source rows")
    runs: list[tuple[str, int, int]] = []
    start, current = 0, labels[0][0]
    for label, index in labels[1:]:
        if label != current:
            runs.append((current, start, index))
            start, current = index, label
    runs.append((current, start, count))
    return count, runs


def _loader(path: Path, column: int, numeric: bool) -> Callable[[], pa.Array]:
    def load() -> pa.Array:
        values = [row[column] for _, row in _rows(path)]
        if numeric:
            try:
                array = np.asarray(values, dtype=np.float64)
            except ValueError as exc:
                raise TimeFFormatError(f"{path}: nonnumeric sensor value at column {column}") from exc
            if not np.isfinite(array).all():
                raise TimeFFormatError(f"{path}: nonfinite sensor value at column {column}")
            return pa.array(array, type=pa.float64())
        return pa.array(values, type=pa.string())

    return load


def _extract_verified(archive: Path, staging: Path) -> Path:
    """Extract only the complete official ordinary-file set into private staging.

    Returns:
        The private extracted source root.

    Raises:
        TimeFFormatError: If libarchive is unavailable or the archive layout is unsafe.
    """
    try:
        import libarchive  # noqa: PLC0415  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise TimeFFormatError("Shoaib requires libarchive-c; install this connector's requirements.txt") from exc
    root = staging / "DataSet"
    root.mkdir()
    seen: set[str] = set()
    with libarchive.file_reader(str(archive)) as entries:
        for entry in entries:
            name = entry.pathname
            if name in {"DataSet", "DataSet/"} and entry.isdir and not entry.islnk:
                continue
            allowed = {f"DataSet/{member}" for member in _MEMBERS}
            if name not in allowed or name in seen or not entry.isreg or entry.islnk or entry.issym:
                raise TimeFFormatError(f"unexpected Shoaib archive member: {name}")
            seen.add(name)
            with (root / Path(name).name).open("xb") as handle:
                for block in entry.get_blocks():
                    handle.write(block)
    if seen != {f"DataSet/{member}" for member in _MEMBERS}:
        raise TimeFFormatError("Shoaib archive does not contain all ten participants and README")
    return root


def _validate_cache(root: Path, original: Path) -> None:
    """Compare every cached file, including headers and README, to the pinned archive.

    Raises:
        TimeFFormatError: If cached bytes, member set, or file types differ.
    """
    if root.is_symlink() or not root.is_dir() or {path.name for path in root.iterdir()} != _MEMBERS:
        raise TimeFFormatError("Shoaib extracted cache has an invalid file set or directory type")
    for name in sorted(_MEMBERS):
        path = root / name
        if path.is_symlink() or not path.is_file() or _hash(path) != _hash(original / name):
            raise TimeFFormatError(f"Shoaib extracted cache differs from pinned archive: {name}")


class ShoaibConnector(BaseConnector[ShoaibParticipant]):
    """Connector for five-position native Shoaib participant recordings."""

    async def download_async(self, cache_dir: Path) -> list[ShoaibParticipant]:  # noqa: PLR6301
        """Download the pinned RAR and extract only ordinary official CSV files.

        Returns:
            One native source reference per participant file.

        Raises:
            TimeFFormatError: If the archive or extracted source is invalid.
        """
        if any(path.is_symlink() for path in (cache_dir, *cache_dir.parents)):
            raise TimeFFormatError("Shoaib cache path must not contain symlinks")
        cache_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
        archive = cache_dir / ARCHIVE
        if archive.is_symlink() or (archive.exists() and not archive.is_file()):
            raise TimeFFormatError("Shoaib archive cache must be an ordinary nonsymlink file")
        if archive.exists() and _hash(archive) != SHA256:
            raise TimeFFormatError(f"cached Shoaib archive hash mismatch: {archive}")
        if not archive.exists():
            await download_files([Artifact(URL, archive, sha256=SHA256)], skip_existing=False)
        if _hash(archive) != SHA256:
            raise TimeFFormatError(f"downloaded Shoaib archive hash mismatch: {archive}")
        root = cache_dir / "DataSet"
        staging = Path(tempfile.mkdtemp(prefix=".shoaib-", dir=cache_dir))
        try:
            original = _extract_verified(archive, staging)
            if root.exists() or root.is_symlink():
                _validate_cache(root, original)
            else:
                original.replace(root)
        finally:
            shutil.rmtree(staging)
        refs = [
            ShoaibParticipant(path=path, participant=path.stem.removeprefix("Participant_"))
            for path in sorted(root.glob("Participant_*.csv"))
        ]
        if not refs:
            raise TimeFFormatError("Shoaib archive contains no participant CSV files")
        return refs

    def convert(self, raw_refs: list[ShoaibParticipant]) -> TimeFDataset:
        """Build one relative-time record per participant without coercing timestamp tokens.

        Returns:
            A dataset retaining all five timestamp columns, sixty sensors, and label stream.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        for ref in raw_refs:
            count, runs = _summary(ref.path)
            record_id = f"shoaib-participant-{ref.participant}"
            series = []
            for position_index, position in enumerate(POSITIONS):
                base = position_index * 14
                series.append(
                    TimeSeries(
                        spec=SPECS["T"],
                        signal=f"{position}_timestamp",
                        time_axis=AXIS,
                        loader=_loader(ref.path, base, False),
                        source_id=ref.path.name,
                        time_series_id=f"{record_id}-{position}-timestamp",
                        n_values=count,
                    )
                )
                for offset, field in enumerate(FIELDS, 1):
                    series.append(
                        TimeSeries(
                            spec=SPECS[field[0]],
                            signal=f"{position}_{field.lower()}",
                            time_axis=AXIS,
                            loader=_loader(ref.path, base + offset, True),
                            source_id=ref.path.name,
                            time_series_id=f"{record_id}-{position}-{field.lower()}",
                            n_values=count,
                        )
                    )
            series.append(
                TimeSeries(
                    spec=SPECS["Y"],
                    signal="activity",
                    time_axis=AXIS,
                    loader=_loader(ref.path, 69, False),
                    source_id=ref.path.name,
                    time_series_id=f"{record_id}-activity",
                    n_values=count,
                )
            )
            record = dataset.add_record(time_series=tuple(series), subject_ids=(ref.participant,), record_id=record_id)
            for index, (label, start, end) in enumerate(runs):
                dataset.add_task(
                    record,
                    ClassificationTask(
                        target=label,
                        scope=TimeInterval.seconds(start / 50, end / 50),
                        id=f"{record_id}-activity-{index}",
                    ),
                )
        return dataset


CONNECTOR = ShoaibConnector
