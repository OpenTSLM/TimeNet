"""Convert HEARTS frames, audio, and images to signals.

Time axes preserve gaps in CGMacros windows. Audio uses the payload sampling rate
or the rate from the reference harness. Each source identifies the upstream corpus.
"""

from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from timenet.dataset import Signal, Source
from timenet.dataset.axis import IrregularAxis, RegularAxis, TimeAxis, to_time_offsets_us
from timenet.errors import TimeFFormatError
from timenet.types import TimeSeriesSpec, ureg
from timenet_connectors.datasets.yang_ai_lab.hearts.pickles import dig, load_payload, require_pandas


_CORPORA: dict[str, tuple[str, str]] = {
    "cgmacros": ("CGMacros", "PhysioNet"),
    "harespod": ("HARESPOD", "figshare"),
    "coswara": ("Coswara-Data", "IISc LEAP Lab"),
    "coughvid": ("COUGHVID", "EPFL EMBED"),
    "vctk": ("VCTK Corpus", "University of Edinburgh"),
}
"""Corpus names and providers for source metadata."""

_CGM = TimeSeriesSpec(
    spec_type="cgm",
    name="Interstitial glucose",
    unit_value=ureg.milligram / ureg.deciliter,
    dtype="float64",
)
_RESPIRATION_NORM = TimeSeriesSpec(
    spec_type="respiration_norm",
    name="Respiration (min-max scaled)",
    unit_value=ureg.dimensionless,
    dtype="float64",
)
_SPO2_NORM = TimeSeriesSpec(
    spec_type="spo2_norm",
    name="Oxygen saturation (min-max scaled)",
    unit_value=ureg.dimensionless,
    dtype="float64",
)
_HEART_RATE_NORM = TimeSeriesSpec(
    spec_type="heart_rate_norm",
    name="Heart rate (min-max scaled)",
    unit_value=ureg.dimensionless,
    dtype="float64",
)
_AUDIO_COSWARA = TimeSeriesSpec(
    spec_type="audio_coswara",
    name="Coswara respiratory audio",
    unit_value=ureg.dimensionless,
    dtype="float32",
)
_AUDIO_COUGHVID = TimeSeriesSpec(
    spec_type="audio_coughvid",
    name="COUGHVID cough audio",
    unit_value=ureg.dimensionless,
    dtype="float32",
)
_AUDIO_VCTK = TimeSeriesSpec(
    spec_type="audio_vctk",
    name="VCTK speech waveform",
    unit_value=ureg.dimensionless,
    dtype="float32",
)
_ACTIVITY_CALORIES = TimeSeriesSpec(
    spec_type="activity_calories",
    name="Activity calories",
    unit_value=ureg.kilocalorie,
    dtype="float64",
)
_HEART_RATE = TimeSeriesSpec(
    spec_type="heart_rate",
    name="Heart rate",
    unit_value=ureg.bpm,
    dtype="float64",
)

_COLUMN_SPECS: dict[tuple[str, str], TimeSeriesSpec] = {
    ("cgmacros", "Libre GL"): _CGM,
    ("cgmacros", "CGM (mg/dL)"): _CGM,
    ("cgmacros", "Calories (Activity)"): _ACTIVITY_CALORIES,
    ("cgmacros", "HR"): _HEART_RATE,
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
"""Time columns used to build axes."""
_INDEX_COLUMNS = ("timestamp_min",)
"""Redundant minute columns to check against the axis."""
_ANSWER_KEY = "GT"
"""The payload key holding the answer. The series walk never descends into it."""

_VCTK_RATE_HZ = 16_000
"""VCTK carries no rate. ``exp/vctk/base.py`` defaults to this and its prompt states it."""
_US_PER_MINUTE = 60_000_000


def time_offsets_us(column: Any) -> np.ndarray:
    """Convert timestamps or minutes to microseconds relative to the first row.

    Returns:
        A non-decreasing int64 array of microseconds.

    Raises:
        TimeFFormatError: If the column's dtype is not a time column this connector reads.
    """
    pandas = require_pandas()
    kind = column.dtype.kind
    if kind == "O":
        # CGMacros writes wall-clock moments as strings.
        nanos = pandas.to_datetime(column).to_numpy(dtype="datetime64[ns]").astype("int64")
    elif kind in "Mm":
        # Explicit nanosecond units avoid differences between pandas versions.
        dtype = "datetime64[ns]" if kind == "M" else "timedelta64[ns]"
        nanos = column.to_numpy(dtype=dtype).astype("int64")
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
    """Use a regular axis for constant positive steps, or retain explicit offsets.

    Returns:
        A regular axis when the step is constant, an irregular one otherwise.
    """
    steps = np.diff(offsets)
    if steps.size and int(steps.min()) == int(steps.max()) > 0:
        return RegularAxis(period_us=Fraction(int(steps[0])))
    return IrregularAxis.spanning(offsets)


def _column_loader(path: Path, keys: tuple[str, ...], column: str, dtype: str) -> Callable[[], pa.Array]:
    """Create a lazy Arrow loader for a frame column.

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
                f"HEARTS array {'.'.join(keys)} in {path.name} has dtype {node.dtype} and "
                f"ndim {node.ndim}. Expected a 1-D float array."
            )
        yield keys, node
    elif isinstance(node, frame_type):
        yield keys, node


def _audio_rate_hz(source: str, payload: dict[str, Any], keys: tuple[str, ...]) -> int:
    """Read the audio rate, or use the reference harness rate for VCTK.

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
    """Check that a redundant minute column matches the time axis before omitting it.

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
) -> Iterator[Signal]:
    """Create one signal per value column, with a shared time axis.

    Yields:
        One :class:`~timenet.dataset.Signal` per value column.

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
        yield Signal.from_loader(
            spec=spec,
            name=signal,
            time_axis=axis,
            loader=_column_loader(path, keys, str(column), spec.dtype),
            time_offsets_loader=_offsets_loader(path, keys, time_column) if irregular else None,
            source_id=record_id,
            id=f"{record_id}-{signal}",
            n_values=len(frame),
            metadata={"source_time_origin": str(frame[time_column].iloc[0])},
        )


def series_for(
    source: str,
    path: Path,
    payload: dict[str, Any],
    record_id: str,
    series_keys: tuple[str, ...] | None = None,
) -> tuple[Signal, ...]:
    """Create signals from the selected payload fields, sorted by name.

    Returns:
        The record's signals, sorted by name so the order does not depend on dict order.

    Raises:
        TimeFFormatError: If an audio buffer has no spec, an array has a shape this connector does
            not read, or a frame cannot be read.
    """
    series: list[Signal] = []
    if series_keys is not None:
        missing = set(series_keys) - payload.keys()
        if missing:
            raise TimeFFormatError(f"HEARTS {path} lacks input series fields {sorted(missing)}")
        selected = {key: payload[key] for key in series_keys}
    else:
        selected = payload
    for keys, node in _walk(selected, path, require_pandas().DataFrame):
        if isinstance(node, np.ndarray):
            spec = _AUDIO_SPECS.get(source)
            if spec is None:
                raise TimeFFormatError(f"HEARTS {source} has a bare array at {'.'.join(keys)} but no audio spec")
            signal = ".".join(keys)
            series.append(
                Signal.from_loader(
                    spec=spec,
                    name=signal,
                    time_axis=RegularAxis.from_rate_hz(_audio_rate_hz(source, payload, keys)),
                    loader=_array_loader(path, keys, spec.dtype),
                    source_id=record_id,
                    id=f"{record_id}-{signal}",
                    n_values=int(node.size),
                )
            )
        else:
            series.extend(_frame_series(source, path, record_id, keys, node))
    return tuple(sorted(series, key=lambda item: item.name))


def source_for(  # noqa: PLR0913 - source identity and input selection are distinct data
    source: str,
    path: Path,
    payload: dict[str, Any],
    record_id: str,
    *,
    series_keys: tuple[str, ...] | None = None,
    images: bool = False,
) -> Source:
    """Create a source with the case's signals and corpus provenance.

    Returns:
        The source holding every signal of the case.

    Raises:
        TimeFFormatError: If the corpus directory is not one this connector converts, or a signal
            cannot be built from the payload.
    """
    corpus = _CORPORA.get(source)
    if corpus is None:
        raise TimeFFormatError(f"HEARTS directory {source!r} names no upstream corpus this connector converts")
    name, provider = corpus
    signals = list(series_for(source, path, payload, record_id, series_keys))
    if images:
        from timenet_connectors.sources.images import image_signal  # noqa: PLC0415 - optional Pillow dependency

        mapping = payload.get("image_mapping")
        if not isinstance(mapping, dict) or set(mapping) != {"a.jpg", "b.jpg", "c.jpg", "d.jpg"}:
            raise TimeFFormatError(f"HEARTS {path} does not contain the four meal photographs")
        for filename in sorted(mapping):
            content = mapping[filename]
            if not isinstance(content, bytes):
                raise TimeFFormatError(f"HEARTS {path} image {filename} is not JPEG bytes")
            signals.append(image_signal(content, signal_id=f"{record_id}-image-{filename}", name=filename))
    return Source(
        id=f"{record_id}-source",
        name=name,
        signals=tuple(signals),
        metadata={"provider": provider},
    )
