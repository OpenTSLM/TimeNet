"""Connector for the Bosch Research CNC Machining dataset."""

from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pyarrow as pa

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFValidationError
from timenet.types import Annotation, ClassificationTask, DataSource, TimeSeriesSpec, ureg
from timenet_connectors.download import ensure_archive, find_dir_containing


_SOURCE_COMMIT = "d60581d6a3ab6015dcc5488c3d76112bb8e1bcb1"
_SOURCE_URL = f"https://github.com/boschresearch/CNC_Machining/archive/{_SOURCE_COMMIT}.zip"
_SOURCE_SHA256 = "a6b6f04c23850a977cd4d7359e960529fd9e23b1e09565f2a0fb76ab4336e2b5"
_LOCAL_SOURCE_ENV = "TIMENET_BOSCH_CNC_SOURCE_DIR"
_FILENAME = re.compile(
    r"^(?P<machine>M\d{2})_(?P<timeframe>[A-Za-z]{3}_\d{4})_"
    r"(?P<process>OP\d{2})_(?P<example>\d+)\.h5$"
)
_SAFE_FLOAT64_INTEGER = 2**53
_SIGNALS = ("x", "y", "z")
_SOURCE_PATH_PARTS = 4
_VIBRATION_NDIM = 2

_ACCELERATION_SPEC = TimeSeriesSpec(
    spec_type="acceleration",
    name="Acceleration",
    unit_value=ureg.Unit("milligravity"),
    data_source=DataSource(
        data_source_type="accelerometer",
        name="Bosch CISS Sensor",
        provider="Bosch",
    ),
    dtype="float64",
)


@dataclass(frozen=True)
class BoschCncSource:
    """A validated source HDF5 recording and its path-derived metadata."""

    path: Path
    source_file: str
    record_id: str
    machine_number: str
    process_number: str
    process_health: str
    timeframe: str
    example_number: str


def _h5py() -> Any:
    """Import h5py lazily so connector discovery works without its extra dependency.

    Returns:
        The imported :mod:`h5py` module.

    Raises:
        ImportError: If the connector dependency is not installed.
    """
    try:
        import h5py  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "The Bosch CNC connector requires h5py; install its requirements.txt before building."
        ) from exc
    return h5py


def _parse_source(data_root: Path, path: Path) -> BoschCncSource:
    """Parse and cross-check one source path.

    Args:
        data_root: Directory containing machine-number folders.
        path: HDF5 file below ``data_root``.

    Returns:
        The deterministic source reference.

    Raises:
        TimeFValidationError: If the hierarchy, label, or redundant filename metadata is invalid.
    """
    try:
        relative = path.relative_to(data_root)
    except ValueError as exc:
        raise TimeFValidationError(f"Bosch source file is outside the data root: {path}") from exc
    if len(relative.parts) != _SOURCE_PATH_PARTS:
        raise TimeFValidationError(
            f"Bosch source files must follow data/<machine>/<process>/<health>/<file>.h5, got {relative.as_posix()!r}"
        )
    machine, process, health, filename = relative.parts
    if health not in {"good", "bad"}:
        raise TimeFValidationError(
            f"Bosch process-health directory must be 'good' or 'bad', got {health!r} in {relative}"
        )
    match = _FILENAME.fullmatch(filename)
    if match is None:
        raise TimeFValidationError(f"malformed Bosch CNC filename: {relative.as_posix()!r}")
    if machine != match["machine"] or process != match["process"]:
        raise TimeFValidationError(
            "Bosch directory/filename metadata disagreement for "
            f"{relative.as_posix()!r}: directory=({machine}, {process}), "
            f"filename=({match['machine']}, {match['process']})"
        )
    if re.fullmatch(r"M\d{2}", machine) is None or re.fullmatch(r"OP\d{2}", process) is None:
        raise TimeFValidationError(f"malformed Bosch CNC hierarchy: {relative.as_posix()!r}")
    return BoschCncSource(
        path=path,
        source_file=relative.as_posix(),
        record_id=relative.with_suffix("").as_posix(),
        machine_number=machine,
        process_number=process,
        process_health=health,
        timeframe=match["timeframe"],
        example_number=match["example"],
    )


def _discover(data_root: Path) -> list[BoschCncSource]:
    """Discover all Bosch HDF5 records in deterministic order.

    Returns:
        One source reference per HDF5 file.

    Raises:
        TimeFValidationError: If the root has no source records or yields duplicate IDs.
    """
    if not data_root.is_dir():
        raise TimeFValidationError(f"Bosch CNC data directory does not exist: {data_root}")
    sources = [_parse_source(data_root, path) for path in sorted(data_root.rglob("*.h5"))]
    if not sources:
        raise TimeFValidationError(f"no Bosch CNC .h5 files found under {data_root}")
    record_ids = [source.record_id for source in sources]
    if len(record_ids) != len(set(record_ids)):
        raise TimeFValidationError("Bosch CNC source paths produced duplicate record IDs")
    return sources


def _inspect(path: Path) -> tuple[int, np.dtype[Any]]:
    """Validate one HDF5 container and return its signal length and dtype.

    Returns:
        The row count and source NumPy dtype.

    Raises:
        TimeFValidationError: If the file or ``vibration_data`` schema is unsupported.
    """
    h5py = _h5py()
    try:
        with h5py.File(path, "r") as handle:
            root_keys = set(handle.keys())
            if "vibration_data" not in handle:
                raise TimeFValidationError(f"missing 'vibration_data' dataset in {path}")
            if root_keys != {"vibration_data"}:
                raise TimeFValidationError(
                    f"unexpected HDF5 root schema in {path}: expected only 'vibration_data', got {sorted(root_keys)}"
                )
            values = handle["vibration_data"]
            if values.ndim != _VIBRATION_NDIM or values.shape[1] != len(_SIGNALS):
                raise TimeFValidationError(f"'vibration_data' in {path} must have shape (N, 3), got {values.shape}")
            if values.shape[0] <= 0:
                raise TimeFValidationError(f"'vibration_data' in {path} must not be empty")
            n_values = int(values.shape[0])
            dtype = np.dtype(values.dtype)
    except OSError as exc:
        raise TimeFValidationError(f"cannot read Bosch HDF5 file {path}: {exc}") from exc
    if dtype not in {np.dtype("float32"), np.dtype("float64"), np.dtype("int64")}:
        raise TimeFValidationError(
            f"unsupported 'vibration_data' dtype in {path}: {dtype}; expected float32, float64, or int64"
        )
    return n_values, dtype


def _load_channel(path: Path, channel: int) -> pa.Array:
    """Load one source column as checked float64 values.

    Returns:
        The source column represented as a float64 Arrow array.

    Raises:
        TimeFValidationError: If the column cannot be loaded or an integer would lose precision.
    """
    h5py = _h5py()
    try:
        with h5py.File(path, "r") as handle:
            source = np.asarray(handle["vibration_data"][:, channel])
    except (KeyError, OSError) as exc:
        raise TimeFValidationError(f"cannot load 'vibration_data' channel {channel} from {path}: {exc}") from exc
    if source.dtype == np.dtype("int64") and source.size:
        smallest = int(source.min())
        largest = int(source.max())
        if smallest < -_SAFE_FLOAT64_INTEGER or largest > _SAFE_FLOAT64_INTEGER:
            raise TimeFValidationError(
                f"unsafe int64 to float64 conversion in {path}: range [{smallest}, {largest}] "
                f"exceeds the exact-integer range [-{_SAFE_FLOAT64_INTEGER}, {_SAFE_FLOAT64_INTEGER}]"
            )
    return pa.array(source.astype(np.float64, copy=False), type=pa.float64())


def _channel_loader(path: Path, channel: int) -> Callable[[], pa.Array]:
    """Return a lazy loader for one HDF5 vibration column."""
    return lambda: _load_channel(path, channel)


def _local_data_root(local_source: str) -> Path:
    """Resolve a local source override to its data directory.

    Returns:
        The source's ``data`` directory, or the given path when it is already that directory.
    """
    root = Path(local_source).expanduser().resolve()
    return root / "data" if (root / "data").is_dir() else root


class BoschCncConnector(BaseConnector[BoschCncSource]):
    """Convert the Bosch CNC Machining corpus into TimeF."""

    async def download_async(  # noqa: PLR6301
        self, cache_dir: Path
    ) -> list[BoschCncSource]:
        """Locate an explicit local checkout or download the commit-pinned source archive.

        Returns:
            Deterministically ordered HDF5 source references.
        """
        local_source = os.environ.get(_LOCAL_SOURCE_ENV)
        if local_source:
            data_root = _local_data_root(local_source)
        else:
            extracted = await ensure_archive(
                _SOURCE_URL,
                cache_dir,
                filename=f"bosch-cnc-{_SOURCE_COMMIT}.zip",
                sha256=_SOURCE_SHA256,
            )
            data_root = find_dir_containing(extracted, "README.md") / "data"
        return _discover(data_root)

    def convert(self, raw_refs: list[BoschCncSource]) -> TimeFDataset:
        """Convert source references without fabricating timestamps, start times, or events.

        Returns:
            The in-memory TimeF dataset with lazy signal loaders.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        axis = RegularAxis.from_rate_hz(2000)
        for source in raw_refs:
            n_values, _ = _inspect(source.path)
            time_series = tuple(
                TimeSeries(
                    spec=_ACCELERATION_SPEC,
                    signal=signal,
                    time_axis=axis,
                    loader=_channel_loader(source.path, channel),
                    source_id=source.source_file,
                    time_series_id=f"{source.record_id}:{signal}",
                    n_values=n_values,
                )
                for channel, signal in enumerate(_SIGNALS)
            )
            record = dataset.add_record(time_series=time_series, record_id=source.record_id)
            metadata = {
                "source_file": source.source_file,
                "machine_number": source.machine_number,
                "process_number": source.process_number,
                "process_health": source.process_health,
                "timeframe": source.timeframe,
                "example_number": source.example_number,
            }
            record.add_annotations(
                Annotation(key=key, value=value, id=f"{source.record_id}:metadata:{key}")
                for key, value in metadata.items()
            )
            dataset.add_task(
                record,
                ClassificationTask(
                    target=source.process_health,
                    target_schema="process_health",
                    id=f"{source.record_id}:process-health",
                ),
            )
        return dataset


CONNECTOR = BoschCncConnector
