"""Convert HEARTS frames, audio, and images to signals.

Time axes preserve gaps in CGMacros windows. Audio uses the payload sampling rate
or the rate from the reference harness. Each source identifies the upstream corpus.
"""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import IrregularAxis, RegularAxis, to_time_offsets_us
from timenet.errors import TimeFFormatError
from timenet.types import TimeOrigin, TimeSeriesSpec, ureg
from timenet_connectors.datasets.yang_ai_lab.hearts.pickles import dig, load_payload, require_pandas
from timenet_connectors.time_axes import axis_for_offsets, offsets_from_origin


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
"""VCTK carries no rate. The HEARTS reference harness defaults to 16 kHz:
https://github.com/yang-ai-lab/HEARTS/blob/main/exp/vctk/base.py
"""
_US_PER_MINUTE = 60_000_000


def _raw_time_us(column: Any) -> np.ndarray:
    """Convert a timestamp or minute column to raw microseconds.

    Returns:
        One microsecond value per row, before subtraction of a source origin.

    Raises:
        TimeFFormatError: If the column has an unsupported time dtype.
    """
    pandas = require_pandas()
    if bool(pandas.isna(column).any()):
        raise TimeFFormatError("HEARTS time column contains missing timestamps")
    kind = column.dtype.kind
    if kind == "O":
        nanos = pandas.to_datetime(column).to_numpy(dtype="datetime64[ns]").astype("int64")
    elif kind in "Mm":
        dtype = "datetime64[ns]" if kind == "M" else "timedelta64[ns]"
        nanos = column.to_numpy(dtype=dtype).astype("int64")
    elif kind == "f":
        if not bool(np.isfinite(column.to_numpy()).all()):
            raise TimeFFormatError("HEARTS minute column contains non-finite timestamps")
        return np.rint(column.to_numpy() * _US_PER_MINUTE).astype(np.int64)
    else:
        raise TimeFFormatError(
            f"a HEARTS time column has dtype {column.dtype}, which this connector does not read as time"
        )
    return nanos // 1_000


def time_offsets_us(column: Any, origin_us: int | None = None) -> np.ndarray:
    """Convert timestamps or minutes to microseconds relative to a source origin.

    Returns:
        A non-decreasing int64 array of microseconds.

    """
    micros = _raw_time_us(column)
    return to_time_offsets_us(offsets_from_origin(micros, int(micros[0]) if origin_us is None else origin_us))


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


def _offsets_loader(
    path: Path, keys: tuple[str, ...], time_column: str, origin_us: int | None
) -> Callable[[], pa.Array]:
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
        return pa.array(time_offsets_us(frame[time_column], origin_us))

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


def _frame_series(  # noqa: PLR0913 - source identity and frame timing are separate inputs
    source: str,
    path: Path,
    record_id: str,
    keys: tuple[str, ...],
    frame: Any,
    *,
    origin_us: int | None = None,
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
    offsets = time_offsets_us(frame[time_column], origin_us)
    axis = axis_for_offsets(offsets)
    irregular = isinstance(axis, IrregularAxis)
    for column in frame.columns:
        if column in _TIME_COLUMNS:
            continue
        if column in _INDEX_COLUMNS:
            _check_restated_time(path, keys, str(column), frame[column], time_offsets_us(frame[time_column]))
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
            time_offsets_loader=_offsets_loader(path, keys, time_column, origin_us) if irregular else None,
            source_id=record_id,
            id=f"{record_id}-{signal}",
            n_values=len(frame),
        )


def _selected_nodes(
    path: Path, payload: dict[str, Any], series_keys: tuple[str, ...] | None
) -> list[tuple[tuple[str, ...], Any]]:
    """Select input nodes without walking into an answer.

    Returns:
        The selected frame and audio nodes.

    Raises:
        TimeFFormatError: If a required input field is absent.
    """
    if series_keys is not None:
        missing = set(series_keys) - payload.keys()
        if missing:
            raise TimeFFormatError(f"HEARTS {path} lacks input series fields {sorted(missing)}")
        selected = {key: payload[key] for key in series_keys}
    else:
        selected = payload
    return list(_walk(selected, path, require_pandas().DataFrame))


def _datetime_group(column: Any) -> bool | None:
    """Return timezone awareness for a datetime column, or None for relative time."""
    if column.dtype.kind not in "MO":
        return None
    parsed = require_pandas().to_datetime(column)
    return parsed.dt.tz is not None


def _images_source(path: Path, payload: dict[str, Any], record_id: str) -> Source:
    """Read the four meal photographs without assigning them capture timestamps.

    Returns:
        A Source of ordinal image Signals.

    Raises:
        TimeFFormatError: If the payload does not contain the four JPEG buffers.
    """
    from timenet_connectors.sources.images import image_signal  # noqa: PLC0415 - optional Pillow

    mapping = payload.get("image_mapping")
    if not isinstance(mapping, dict) or set(mapping) != {"a.jpg", "b.jpg", "c.jpg", "d.jpg"}:
        raise TimeFFormatError(f"HEARTS {path} does not contain the four meal photographs")
    pictures = []
    for filename in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
        content = mapping[filename]
        if not isinstance(content, bytes):
            raise TimeFFormatError(f"HEARTS {path} image {filename} is not JPEG bytes")
        pictures.append(image_signal(content, signal_id=f"{record_id}-image-{filename}", name=filename))
    return Source(id=f"{record_id}-source-images", name="meal photographs", signals=tuple(pictures))


def _alignment_group(source: str, keys: tuple[str, ...], aware: bool | None) -> tuple[str, ...]:
    """Group only frames whose alignment follows from source timestamps or case structure.

    CGMacros reference_cgm_df and window_df retain the subject's calendar timestamps:
    https://huggingface.co/datasets/yang-ai-lab/HEARTS/blob/7c18df521ae36cbc6b61e17782f1ac08dc378ea1/cgmacros/meal_forecasting/0.pkl
    Naive timestamps retain relative differences, without an invented timezone.
    Other relative frames and audio clips have no established cross-record alignment.

    Returns:
        A shared group for aligned frames, otherwise a key unique to the payload node.
    """
    if aware is True:
        return ("absolute",)
    if source == "cgmacros" and aware is False and keys in {("reference_cgm_df",), ("window_df",)}:
        return ("cgmacros-session",)
    return ("independent", *keys)


def records_for(  # noqa: PLR0913 - payload groups and lazy loaders
    source: str,
    path: Path,
    payload: dict[str, Any],
    case_id: str,
    *,
    series_keys: tuple[str, ...] | None = None,
    images: bool = False,
) -> tuple[Record, ...]:
    """Create one Record per established input timeline, with corpus provenance.

    Returns:
        Aligned frames share a Record. Independent recordings remain separate task inputs.

    Raises:
        TimeFFormatError: If a corpus, frame, or audio field cannot be converted.
    """
    corpus = _CORPORA.get(source)
    if corpus is None:
        raise TimeFFormatError(f"HEARTS directory {source!r} names no supported upstream corpus")
    groups: dict[tuple[str, ...], list[tuple[tuple[str, ...], Any]]] = {}
    columns: dict[tuple[str, ...], str] = {}
    starts: dict[tuple[str, ...], int] = {}
    for keys, node in _selected_nodes(path, payload, series_keys):
        aware = None
        if not isinstance(node, np.ndarray):
            column = next((name for name in _TIME_COLUMNS if name in node.columns), None)
            if column is None or len(node) == 0:
                raise TimeFFormatError(f"HEARTS frame {'.'.join(keys)} in {path.name} has no usable time column")
            columns[keys] = column
            starts[keys] = int(_raw_time_us(node[column])[0])
            aware = _datetime_group(node[column])
        group = _alignment_group(source, keys, aware)
        groups.setdefault(group, []).append((keys, node))
    if not groups:
        if not images:
            raise TimeFFormatError(f"HEARTS {path} has no selected input signals")
        groups["images",] = []

    records = []
    for index, (group, nodes) in enumerate(groups.items()):
        record_id = case_id if len(groups) == 1 else f"{case_id}-{index}"
        origin_us = min((starts[keys] for keys, _ in nodes if keys in starts), default=0)
        origin = TimeOrigin(origin_us if group == ("absolute",) else None)
        children = []
        for keys, node in nodes:
            signal_name = ".".join(keys)
            metadata: dict[str, object] = {}
            if isinstance(node, np.ndarray):
                spec = _AUDIO_SPECS.get(source)
                if spec is None:
                    raise TimeFFormatError(f"HEARTS {source} has an array at {signal_name} but no audio spec")
                signals = (
                    Signal.from_loader(
                        spec=spec,
                        name=signal_name,
                        time_axis=RegularAxis.from_rate_hz(_audio_rate_hz(source, payload, keys)),
                        loader=_array_loader(path, keys, spec.dtype),
                        source_id=record_id,
                        id=f"{record_id}-{signal_name}",
                        n_values=int(node.size),
                    ),
                )
            else:
                signals = tuple(_frame_series(source, path, record_id, keys, node, origin_us=origin_us))
                metadata["source_time_origin"] = str(node[columns[keys]].iloc[0])
            children.append(
                Source(
                    id=f"{record_id}-source-{signal_name}",
                    name=signal_name,
                    signals=signals,
                    metadata=metadata,
                )
            )
        if images and index == 0:
            children.append(_images_source(path, payload, record_id))
        records.append(
            Record(
                record_id=record_id,
                start_time=origin,
                sources=(
                    Source(
                        id=f"{record_id}-source",
                        name=corpus[0],
                        sources=tuple(children),
                        metadata={"provider": corpus[1]},
                    ),
                ),
            )
        )
    return tuple(records)
