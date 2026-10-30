"""Build one six-channel IMU Record per source row, with lazy values and reconstructed timing."""

from functools import partial

import pyarrow as pa

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import IrregularAxis
from timenet.types import Annotation, TimeSeriesSpec
from timenet_connectors.datasets.bash_lab.sensorcaps.parse import (
    EXPORTED_SAMPLES,
    FEATURE_HEADINGS,
    ParsedRow,
    RowRef,
    parse_row,
)
from timenet_connectors.datasets.bash_lab.sensorcaps.release import source_file


DECLARED_SAMPLING_RATE_HZ = 10
SOURCE_SAMPLING_RATE_HZ = 20
SOURCE_WINDOW_SAMPLES = 120
TIME_AXIS_RECONSTRUCTION = "round(linspace(0, 119, 60)) / 20 seconds relative to window start"
# Apply the shared SensorCaps timing interpretation to all five partitions: upstream selects
# round(linspace(0, 119, 60)) from a 20 Hz window. These are
# reconstructed relative offsets, not acquisition timestamps or a regular 10 Hz cadence.
# https://github.com/BASHLab/LLaSA/blob/main/llasa_v2_data_generation.py
_TIME_OFFSETS_US = pa.array(
    [
        round(index * (SOURCE_WINDOW_SAMPLES - 1) / (EXPORTED_SAMPLES - 1)) * (1_000_000 // SOURCE_SAMPLING_RATE_HZ)
        for index in range(EXPORTED_SAMPLES)
    ],
    type=pa.int64(),
)
_AXIS = IrregularAxis(first_us=0, last_us=(SOURCE_WINDOW_SAMPLES - 1) * (1_000_000 // SOURCE_SAMPLING_RATE_HZ))
_SPECS = (
    # Upstream rounds IMU values to six decimal places. Use compact float32 storage, accepting
    # normal binary floating-point rounding rather than implying exact decimal preservation.
    TimeSeriesSpec(spec_type="accelerometer", name="Accelerometer", unit_value=None, dtype="float32"),
    TimeSeriesSpec(spec_type="gyroscope", name="Gyroscope", unit_value=None, dtype="float32"),
)
_AXES = ("x", "y", "z")
_CHANNELS = tuple(
    (spec, f"{spec.spec_type}_{axis}", sensor, dimension)
    for sensor, spec in enumerate(_SPECS)
    for dimension, axis in enumerate(_AXES)
)
# The published PAMAP2 arrays are shifted by one raw column relative to XYZ groups.
# Verified against all 9,672 examples and UCI's per-IMU column layout:
# https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
# Keep the published values/order, but identify their actual physical channels.
_PAMAP2_CHANNELS = (
    (_SPECS[0], "accelerometer_16g_y", 0, 0),
    (_SPECS[0], "accelerometer_16g_z", 0, 1),
    (_SPECS[0], "accelerometer_6g_x", 0, 2),
    (_SPECS[1], "gyroscope_y", 1, 0),
    (_SPECS[1], "gyroscope_z", 1, 1),
    (
        TimeSeriesSpec(spec_type="magnetometer", name="Magnetometer", unit_value=None, dtype="float32"),
        "magnetometer_x",
        1,
        2,
    ),
)


class ValueLoader:
    """Share one bounded sensor-row cache across a dataset's lazy channel loaders."""

    def __init__(self) -> None:
        self.cached: tuple[RowRef, tuple[list[list[float]], list[list[float]]]] | None = None

    def __call__(self, ref: RowRef, sensor: int, axis: int) -> pa.Array:
        """Return one channel of a row, decoding the row only if it isn't the cached one.

        Returns:
            The channel's float32 values in source order.
        """
        cached = self.cached
        if cached is None or cached[0] != ref:
            with ref.file.path.open("rb") as handle:
                handle.seek(ref.offset)
                values = parse_row(handle.readline(), ref.context).values
            self.cached = ref, values
        else:
            values = cached[1]
        # Preserve the HF benchmark values. Do not infer nulls from zeros or substitute
        # original PAMAP2 readings: resampling has already mixed zero-filled gaps into
        # the published arrays. See README.md for the missing-data limitation.
        # Use the local values even if another consumer replaces the shared cache during this read.
        return pa.array((sample[axis] for sample in values[sensor]), type=pa.float32())


def _load_time_offsets() -> pa.Array:
    """Return the shared reconstructed offsets in integer microseconds.

    Returns:
        One relative offset per exported sensor reading.
    """
    return _TIME_OFFSETS_US


def feature_key(heading: str) -> str:
    """Return the annotation key of a derived feature, such as ``gyroscope_moving_average``.

    Returns:
        The heading in lowercase, with underscores for spaces.
    """
    return heading.lower().replace(" ", "_")


def make_record(ref: RowRef, row: ParsedRow, loader: ValueLoader) -> Record:
    """Build one six-channel IMU Record with its source provenance.

    Returns:
        A Record with lazy sensor values and annotations from the original row.
    """
    channels = _PAMAP2_CHANNELS if ref.file.part == "pamap2_v2" else _CHANNELS
    signals = tuple(
        Signal.from_loader(
            name=name,
            spec=spec,
            time_axis=_AXIS,
            time_offsets_loader=_load_time_offsets,
            n_values=len(row.values[sensor]),
            loader=partial(loader, ref, sensor, dimension),
        )
        for spec, name, sensor, dimension in channels
    )
    record = Record(sources=(Source(name="IMU", signals=signals),))
    source = ref.file.part.rsplit("_", 1)[0]
    metadata: dict[str, str | int] = {
        "source_dataset": source,
        "source_partition": ref.file.part,
        "source_file": source_file(ref.file.part),
        "source_row": ref.index,
        "activity_summary": row.summary,
    }
    if row.location is not None:
        metadata["sensor_location"] = row.location
    for key, value in metadata.items():
        record.annotate(Annotation(key=key, value=value))
    # The derived features are Record annotations only, not task inputs: they are long NumPy text that
    # the generator computed from the 120-reading source window, not from the 60 stored readings.
    for heading in FEATURE_HEADINGS:
        record.annotate(Annotation(key=feature_key(heading), value=row.features[heading]))
    return record
