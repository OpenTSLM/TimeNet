"""Turning one HEARTS payload into time series: the specs, the axis rule, and the lazy loaders.

The axis is decided per series from that series' own time column, never per corpus. HARESPOD frames
step at a constant 10 ms or 1 s and get a :class:`~timenet.dataset.axis.RegularAxis`. CGMacros
windows are not uniform: a single reference window carries two-, three- and seven-minute steps
among its one-minute ones, so those get an :class:`~timenet.dataset.axis.IrregularAxis` and an
explicit stream of microsecond offsets. Audio buffers carry no time column and take their rate from
the payload, or from the reference implementation where the release ships none.
"""

from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from timenet.dataset import TimeSeries
from timenet.dataset.axis import IrregularAxis, RegularAxis, TimeAxis, to_time_offsets_us
from timenet.errors import TimeFFormatError
from timenet.types import DataSource, TimeSeriesSpec, ureg
from timenet_connectors.datasets.yang_ai_lab.hearts.pickles import dig, load_payload, require_pandas


_CGMACROS = DataSource(data_source_type="physionet", name="CGMacros", provider="PhysioNet")
_HARESPOD = DataSource(data_source_type="figshare", name="HARESPOD", provider="figshare")
_COSWARA = DataSource(data_source_type="github", name="Coswara-Data", provider="IISc LEAP Lab")
_COUGHVID = DataSource(data_source_type="zenodo", name="COUGHVID", provider="EPFL EMBED")
_VCTK = DataSource(data_source_type="datashare", name="VCTK Corpus", provider="University of Edinburgh")

_CGM = TimeSeriesSpec(
    spec_type="cgm",
    name="Interstitial glucose",
    unit_value=ureg.milligram / ureg.deciliter,
    dtype="float64",
    data_source=_CGMACROS,
)
_RESPIRATION_NORM = TimeSeriesSpec(
    spec_type="respiration_norm",
    name="Respiration (min-max scaled)",
    unit_value=ureg.dimensionless,
    dtype="float64",
    data_source=_HARESPOD,
)
_SPO2_NORM = TimeSeriesSpec(
    spec_type="spo2_norm",
    name="Oxygen saturation (min-max scaled)",
    unit_value=ureg.dimensionless,
    dtype="float64",
    data_source=_HARESPOD,
)
_HEART_RATE_NORM = TimeSeriesSpec(
    spec_type="heart_rate_norm",
    name="Heart rate (min-max scaled)",
    unit_value=ureg.dimensionless,
    dtype="float64",
    data_source=_HARESPOD,
)
_AUDIO_COSWARA = TimeSeriesSpec(
    spec_type="audio_coswara",
    name="Coswara respiratory audio",
    unit_value=ureg.dimensionless,
    dtype="float32",
    data_source=_COSWARA,
)
_AUDIO_COUGHVID = TimeSeriesSpec(
    spec_type="audio_coughvid",
    name="COUGHVID cough audio",
    unit_value=ureg.dimensionless,
    dtype="float32",
    data_source=_COUGHVID,
)
_AUDIO_VCTK = TimeSeriesSpec(
    spec_type="audio_vctk",
    name="VCTK speech waveform",
    unit_value=ureg.dimensionless,
    dtype="float32",
    data_source=_VCTK,
)

_COLUMN_SPECS: dict[tuple[str, str], TimeSeriesSpec] = {
    ("cgmacros", "Libre GL"): _CGM,
    ("cgmacros", "CGM (mg/dL)"): _CGM,
    ("harespod", "rsp"): _RESPIRATION_NORM,
    ("harespod", "spo"): _SPO2_NORM,
    ("harespod", "hr"): _HEART_RATE_NORM,
}
_AUDIO_SPECS: dict[str, TimeSeriesSpec] = {
    "coswara": _AUDIO_COSWARA,
    "coughvid": _AUDIO_COUGHVID,
    "vctk": _AUDIO_VCTK,
}

_TIME_COLUMNS = ("Timestamp", "timestamp", "Time (min)")
"""Columns that place a frame's rows in time. They become an axis, never a series."""
_INDEX_COLUMNS = ("timestamp_min",)
"""Columns that restate the time column as whole minutes. A build checks each against the axis."""
_ANSWER_KEY = "GT"
"""The payload key holding the answer. The series walk never descends into it."""

_VCTK_RATE_HZ = 16_000
"""VCTK carries no rate. ``exp/vctk/base.py`` defaults to this and its prompt states it."""
_US_PER_MINUTE = 60_000_000


def time_offsets_us(column: Any) -> np.ndarray:
    """Return one whole-microsecond offset per row, counted from the frame's first row.

    Args:
        column: The frame's time column.

    Returns:
        A non-decreasing int64 array of microseconds.

    Raises:
        TimeFFormatError: If the column's dtype is not a time column this connector reads.
    """
    pandas = require_pandas()
    kind = column.dtype.kind
    if kind == "O":
        # CGMacros writes wall-clock moments as strings.
        nanos = pandas.to_datetime(column).astype("int64").to_numpy()
    elif kind in "Mm":
        # HARESPOD writes datetime64 (altitude segments) or timedelta64 (pairing segments). Reading
        # either as int64 gives nanoseconds, so the division to microseconds has to be explicit.
        nanos = column.astype("int64").to_numpy()
    elif kind == "f":
        # The iAUC frame counts minutes from the meal.
        minutes = column.to_numpy()
        micros = np.rint(minutes * _US_PER_MINUTE).astype(np.int64)
        return to_time_offsets_us(micros - micros[0])
    else:
        raise TimeFFormatError(
            f"a HEARTS time column has dtype {column.dtype}, which this connector does not read as time"
        )
    micros = nanos // 1_000
    return to_time_offsets_us(micros - micros[0])


def axis_for(offsets: np.ndarray) -> TimeAxis:
    """Return the axis that describes a stream of offsets.

    A stream whose steps are all equal is a cadence and needs no stored offsets. Anything else keeps
    its offsets, because no formula places its values.

    Args:
        offsets: The per-row microsecond offsets.

    Returns:
        A regular axis when the step is constant, an irregular one otherwise.
    """
    steps = np.diff(offsets)
    if steps.size and int(steps.min()) == int(steps.max()) > 0:
        return RegularAxis(period_us=Fraction(int(steps[0])))
    return IrregularAxis.spanning(offsets)


def _column_loader(path: Path, keys: tuple[str, ...], column: str, dtype: str) -> Callable[[], pa.Array]:
    """Build a lazy loader for one column of one nested frame.

    Args:
        path: The test-case file.
        keys: The key path of the frame inside the payload.
        column: The column to read.
        dtype: The spec's dtype.

    Returns:
        A callable returning the column as an Arrow array.
    """
    location = str(path)

    def load() -> pa.Array:
        frame = dig(load_payload(location), keys)
        return pa.array(np.ascontiguousarray(frame[column].to_numpy(), dtype=np.dtype(dtype)))

    return load


def _array_loader(path: Path, keys: tuple[str, ...], dtype: str) -> Callable[[], pa.Array]:
    """Build a lazy loader for one nested 1-D array.

    Args:
        path: The test-case file.
        keys: The key path of the array inside the payload.
        dtype: The spec's dtype.

    Returns:
        A callable returning the array as an Arrow array.
    """
    location = str(path)

    def load() -> pa.Array:
        return pa.array(np.ascontiguousarray(dig(load_payload(location), keys), dtype=np.dtype(dtype)))

    return load


def _offsets_loader(path: Path, keys: tuple[str, ...], time_column: str) -> Callable[[], pa.Array]:
    """Build a lazy loader for an irregular frame's microsecond offsets.

    Args:
        path: The test-case file.
        keys: The key path of the frame inside the payload.
        time_column: The frame's time column.

    Returns:
        A callable returning one int64 microsecond offset per value.
    """
    location = str(path)

    def load() -> pa.Array:
        frame = dig(load_payload(location), keys)
        return pa.array(time_offsets_us(frame[time_column]))

    return load


def _walk(node: Any, path: Path, frame_type: type, keys: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Yield every frame and float array in a payload, depth first, skipping the answer.

    Args:
        node: The payload node to walk.
        path: The test-case file the node came from.
        frame_type: The pandas frame class, passed in so the walk imports nothing.
        keys: The key path that reached this node.

    Yields:
        The key path and the frame or array found there.

    Raises:
        TimeFFormatError: If an array is not the 1-D float buffer this connector reads.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key != _ANSWER_KEY:
                yield from _walk(value, path, frame_type, (*keys, str(key)))
    elif isinstance(node, np.ndarray):
        if node.ndim != 1 or node.dtype.kind != "f":
            raise TimeFFormatError(
                f"HEARTS array {'.'.join(keys)} in {path.name} has dtype {node.dtype} and ndim "
                f"{node.ndim}; this connector reads only 1-D float arrays. Every array the pinned "
                f"revision ships is one, so a later release can hold a shape that belongs here: an "
                f"integer PCM buffer is a real way to store audio. Decide whether this shape "
                f"converts or is dropped on purpose, and record which in this connector's README"
            )
        yield keys, node
    elif isinstance(node, frame_type):
        yield keys, node


def _audio_rate_hz(source: str, payload: dict[str, Any], keys: tuple[str, ...]) -> int:
    """Return the sampling rate of an audio buffer.

    Args:
        source: The corpus the payload came from.
        payload: The whole payload.
        keys: The key path of the buffer.

    Returns:
        The rate in whole hertz.

    Raises:
        TimeFFormatError: If a corpus that states its rate is missing it.
    """
    if source == "vctk":
        return _VCTK_RATE_HZ
    rate = dig(payload, [*keys[:-1], "sr"]) if len(keys) > 1 else payload.get("sr")
    if not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0:
        raise TimeFFormatError(f"HEARTS {source} audio has sampling rate {rate!r}, which is not a positive integer")
    return rate


def _check_restated_time(path: Path, keys: tuple[str, ...], column: str, values: Any, offsets: np.ndarray) -> None:
    """Check a column that restates the time column says what the axis already says.

    The column is read as part of the axis rather than stored, so it has to agree with the axis. A
    column that counts from somewhere else carries a fact the axis does not.

    Args:
        path: The test-case file.
        keys: The key path of the frame.
        column: The name of the restating column.
        values: That column's values.
        offsets: The frame's microsecond offsets.

    Raises:
        TimeFFormatError: If the column is not the offsets counted in whole minutes.
    """
    if not np.array_equal(np.asarray(values), offsets // _US_PER_MINUTE):
        raise TimeFFormatError(
            f"HEARTS column {column!r} of frame {'.'.join(keys)} in {path.name} is not that frame's "
            f"time column in whole minutes, so reading it as part of the axis would drop what it says"
        )


def _frame_series(
    source: str,
    path: Path,
    record_id: str,
    keys: tuple[str, ...],
    frame: Any,
) -> Iterator[TimeSeries]:
    """Yield one series per value column of one frame.

    Args:
        source: The corpus the payload came from.
        path: The test-case file.
        record_id: The owning record's id.
        keys: The key path of the frame.
        frame: The frame itself.

    Yields:
        One :class:`~timenet.dataset.TimeSeries` per value column.

    Raises:
        TimeFFormatError: If the frame holds no rows, has no time column, restates its time column
            as something else, or a value column has no spec.
    """
    time_column = next((name for name in _TIME_COLUMNS if name in frame.columns), None)
    if time_column is None:
        raise TimeFFormatError(
            f"HEARTS frame {'.'.join(keys)} in {path.name} has no time column; its columns are {list(frame.columns)}"
        )
    if len(frame) == 0:
        raise TimeFFormatError(
            f"HEARTS frame {'.'.join(keys)} in {path.name} holds no rows, so its {time_column!r} "
            f"column places nothing in time"
        )
    offsets = time_offsets_us(frame[time_column])
    axis = axis_for(offsets)
    irregular = isinstance(axis, IrregularAxis)
    for column in frame.columns:
        if column in _TIME_COLUMNS:
            continue
        if column in _INDEX_COLUMNS:
            _check_restated_time(path, keys, str(column), frame[column], offsets)
            continue
        spec = _COLUMN_SPECS.get((source, str(column)))
        if spec is None:
            raise TimeFFormatError(
                f"HEARTS column {column!r} of {source} has no TimeSeriesSpec in this connector. Add "
                f"the (source, column) pair to _COLUMN_SPECS"
            )
        signal = ".".join((*keys, str(column)))
        yield TimeSeries(
            spec=spec,
            signal=signal,
            time_axis=axis,
            loader=_column_loader(path, keys, str(column), spec.dtype),
            time_offsets_loader=_offsets_loader(path, keys, time_column) if irregular else None,
            source_id=record_id,
            time_series_id=f"{record_id}-{signal}",
            n_values=len(frame),
        )


def series_for(source: str, path: Path, payload: dict[str, Any], record_id: str) -> tuple[TimeSeries, ...]:
    """Build every series one test case carries, in signal order.

    Args:
        source: The corpus the payload came from.
        path: The test-case file.
        payload: The payload read from that file.
        record_id: The owning record's id.

    Returns:
        The record's series, sorted by signal so the order does not depend on dict order.

    Raises:
        TimeFFormatError: If an audio buffer has no spec, an array has a shape this connector does
            not read, or a frame cannot be read.
    """
    series: list[TimeSeries] = []
    for keys, node in _walk(payload, path, require_pandas().DataFrame):
        if isinstance(node, np.ndarray):
            spec = _AUDIO_SPECS.get(source)
            if spec is None:
                raise TimeFFormatError(f"HEARTS {source} has a bare array at {'.'.join(keys)} but no audio spec")
            signal = ".".join(keys)
            series.append(
                TimeSeries(
                    spec=spec,
                    signal=signal,
                    time_axis=RegularAxis.from_rate_hz(_audio_rate_hz(source, payload, keys)),
                    loader=_array_loader(path, keys, spec.dtype),
                    source_id=record_id,
                    time_series_id=f"{record_id}-{signal}",
                    n_values=int(node.size),
                )
            )
        else:
            series.extend(_frame_series(source, path, record_id, keys, node))
    return tuple(sorted(series, key=lambda item: item.signal))
