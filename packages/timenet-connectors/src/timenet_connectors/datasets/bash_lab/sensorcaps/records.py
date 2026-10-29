"""Build one six-channel IMU Record per source row, with lazy values and the row's generator inputs.

Sensor values stay in the source files and are read back by the row's byte offset. Annotations,
including the derived-feature text, are held in memory: about 1.5 GB of feature text across the
release.
"""

from functools import partial

import pyarrow as pa

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import RegularAxis
from timenet.types import Annotation, TimeSeriesSpec
from timenet_connectors.datasets.bash_lab.sensorcaps.parse import FEATURE_HEADINGS, ParsedRow, RowRef, parse_row
from timenet_connectors.metadata import RecordMetadata


SAMPLING_RATE_HZ = 10
"""The rate the LLaSA paper and the dataset card state: values are "downsampled to 10 Hz"."""
_AXIS = RegularAxis.from_rate_hz(SAMPLING_RATE_HZ)
_SPECS = (
    # The source states no units. Its values have six decimal places, as the caption generator read
    # them and as captions quote them. float32 changes some at the sixth decimal (PAMAP2's -61.793769),
    # so store float64, which keeps every one.
    TimeSeriesSpec(spec_type="accelerometer", name="Accelerometer", unit_value=None, dtype="float64"),
    TimeSeriesSpec(spec_type="gyroscope", name="Gyroscope", unit_value=None, dtype="float64"),
)
_AXES = ("x", "y", "z")
# Every partition, PAMAP2 included, as the source labels its arrays. TimeF orders a source's signals
# by name, so accelerometer comes first although the user message gives the gyroscope first.
# README.md describes how PAMAP2's published arrays differ from these labels.
_CHANNELS = tuple(
    (spec, f"{spec.spec_type}_{axis}", sensor, dimension)
    for sensor, spec in enumerate(_SPECS)
    for dimension, axis in enumerate(_AXES)
)


class SensorCapsMetadata(RecordMetadata):
    """A SensorCaps Record's metadata: where its row sits in the pinned release."""

    partition: str
    """The source file's partition, such as ``"hhar_v2"``."""
    row: int
    """The row's zero-based index within its file."""


def feature_key(heading: str) -> str:
    """Return the annotation key of a derived feature, such as ``gyroscope_moving_average``.

    Returns:
        The heading in lowercase, with underscores for spaces.
    """
    return heading.lower().replace(" ", "_")


class RowCache:
    """Keep the one row decoded last, shared by a dataset's lazy channel values."""

    def __init__(self) -> None:
        self.cached: tuple[RowRef, ParsedRow] | None = None

    def row(self, ref: RowRef) -> ParsedRow:
        """Return a row, decoding it by its byte offset only if it isn't the cached one.

        Returns:
            The decoded row.
        """
        cached = self.cached
        if cached is not None and cached[0] == ref:
            return cached[1]
        with ref.file.path.open("rb") as handle:
            handle.seek(ref.offset)
            row = parse_row(handle.readline(), ref.context)
        self.cached = ref, row
        return row


def _channel_values(cache: RowCache, ref: RowRef, sensor: int, axis: int) -> pa.Array:
    """Return one channel of a row as float64, in source order.

    Returns:
        The channel's values.
    """
    # Use the local row even if another consumer replaces the shared cache during this read.
    values = cache.row(ref).values
    # Preserve the HF benchmark values. Do not infer nulls from zeros or substitute
    # original PAMAP2 readings: resampling has already mixed zero-filled gaps into
    # the published arrays. See README.md for the missing-data limitation.
    return pa.array((sample[axis] for sample in values[sensor]), type=pa.float64())


def make_record(ref: RowRef, row: ParsedRow, cache: RowCache) -> Record:
    """Build one six-channel IMU Record, annotated with what the caption generator received.

    Returns:
        A Record with lazy sensor values, its row's provenance as metadata, and annotations in the
        user message's order: the summary, each derived feature, and the location when given. Keys
        follow the source's labels, which the generation instruction names.
    """
    signals = tuple(
        Signal.from_loader(
            name=name,
            spec=spec,
            time_axis=_AXIS,
            n_values=len(row.values[sensor]),
            loader=partial(_channel_values, cache, ref, sensor, dimension),
        )
        for spec, name, sensor, dimension in _CHANNELS
    )
    metadata = SensorCapsMetadata(partition=ref.file.part, row=ref.index)
    record = Record(sources=(Source(name="IMU", signals=signals),), metadata=metadata.to_json())
    record.annotate(Annotation(key="summary", value=row.summary))
    for heading in FEATURE_HEADINGS:
        record.annotate(Annotation(key=feature_key(heading), value=row.features[heading]))
    if row.location is not None:
        record.annotate(Annotation(key="location", value=row.location))
    return record
