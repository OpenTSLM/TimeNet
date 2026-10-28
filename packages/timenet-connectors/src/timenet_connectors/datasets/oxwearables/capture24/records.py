"""One Record per participant: a Source of lazy x, y and z signals per segment, its labels, and its facts."""

from functools import lru_cache, partial
from pathlib import Path
from typing import ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError
from timenet.types import Annotation, TimeInterval, TimeSeriesSpec, ureg
from timenet_connectors.datasets.oxwearables.capture24.keys import AnnotationKey
from timenet_connectors.datasets.oxwearables.capture24.participants import Participant, Segment, Timeline
from timenet_connectors.datasets.oxwearables.capture24.release import SAMPLE_RATE_HZ
from timenet_connectors.datasets.oxwearables.capture24.vocabulary import LabelDictionary
from timenet_connectors.metadata import MetadataExtension, RecordMetadata


AXES = ("x", "y", "z")
DEVICE = "Axivity AX3"

# The source writes eight significant digits, more than float32 keeps.
_SPEC = TimeSeriesSpec(
    spec_type="acceleration", name="Wrist acceleration", unit_value=ureg.standard_gravity, dtype="float64"
)


class Capture24RecordMetadata(MetadataExtension):
    """CAPTURE-24's extension of a Record's metadata: the recording file the Record holds.

    The participant id is the core ``subject_id``.
    """

    NAME: ClassVar[str] = "capture24"

    file: str


def record_metadata(record: Record) -> tuple[RecordMetadata, Capture24RecordMetadata]:
    """Read a CAPTURE-24 Record's typed metadata back.

    Returns:
        The shared core, whose ``subject_id`` is the participant, and the CAPTURE-24 extension.

    Raises:
        TimeFValidationError: If the Record has no CAPTURE-24 metadata.
    """  # noqa: DOC502 - raised by from_json and extension
    metadata = RecordMetadata.from_json(record.metadata, [Capture24RecordMetadata])
    return metadata, metadata.extension(Capture24RecordMetadata)


def read_acceleration(path: Path, n_rows: int) -> dict[str, np.ndarray]:
    """Read a recording's x, y and z columns as float64.

    Args:
        path: The recording.
        n_rows: The rows its timeline has, which the columns must match.

    Returns:
        One array per axis.

    Raises:
        TimeFFormatError: If a value is missing, unreadable or not finite, or the row count changed.
    """
    try:
        table = pa_csv.read_csv(
            path,
            # Only an empty field is missing, so a written nan reaches the finite check below.
            convert_options=pa_csv.ConvertOptions(
                include_columns=list(AXES), column_types=dict.fromkeys(AXES, pa.float64()), null_values=[""]
            ),
        )
    except pa.ArrowInvalid as exc:
        raise TimeFFormatError(f"{path}: {exc}") from exc
    if table.num_rows != n_rows:
        raise TimeFFormatError(f"{path}: holds {table.num_rows} rows, {n_rows} when it was parsed")
    values = {}
    for axis in AXES:
        column = table[axis].combine_chunks()
        if column.null_count:
            raise TimeFFormatError(f"{path}: {column.null_count} samples have no {axis} value")
        values[axis] = column.to_numpy()
        if not np.isfinite(values[axis]).all():
            raise TimeFFormatError(f"{path}: a {axis} value is not finite")
    return values


class AccelerationLoader:
    """Load one axis of one segment lazily, reading each recording once for all its signals.

    A loader serves the Records of one load. A Record's signals load one after another, so one cached
    recording suffices.
    """

    def __init__(self) -> None:
        self._read = lru_cache(maxsize=1)(read_acceleration)

    def __call__(self, path: Path, n_rows: int, axis: str, segment: Segment) -> pa.Array:
        """Return one segment's values on one axis.

        Returns:
            The segment's float64 values.
        """
        values = self._read(path, n_rows)[axis]
        return pa.array(values[segment.first_index : segment.first_index + segment.n_values], type=pa.float64())


def build_record(
    participant: Participant, timeline: Timeline, dictionary: LabelDictionary, loader: AccelerationLoader
) -> Record:
    """Build a participant's Record: one Source per segment, and its raw labels as span annotations.

    A label run is one annotation shared by the x, y and z signals of its segment. Unlabelled samples
    carry none. The Record keeps nothing of the timeline's per-sample arrays.

    Args:
        participant: The participant and their ``metadata.csv`` row.
        timeline: Their recording's timeline.
        dictionary: The label dictionary the runs' codes index.
        loader: The loader the signals read their values through.

    Returns:
        The Record, whose values stay in the recording until read.
    """
    sources = []
    for number, segment in enumerate(timeline.segments):
        axis = RegularAxis.from_rate_hz(SAMPLE_RATE_HZ, offset_us=segment.offset_us)
        signals = tuple(
            Signal.from_loader(
                name=name,
                spec=_SPEC,
                time_axis=axis,
                n_values=segment.n_values,
                loader=partial(loader, participant.path, timeline.n_rows, name, segment),
            )
            for name in AXES
        )
        sources.append(Source(name=f"{DEVICE} segment {number}", signals=signals))
    metadata = RecordMetadata(subject_id=participant.pid).extend(Capture24RecordMetadata(file=participant.path.name))
    record = Record(sources=tuple(sources), metadata=metadata.to_json())
    record.add_annotations(
        (
            Annotation(key=AnnotationKey.AGE, value=participant.age),
            Annotation(key=AnnotationKey.SEX, value=participant.sex),
            Annotation(key=AnnotationKey.RECORDING_START_LOCAL, value=timeline.start_local.isoformat()),
        )
    )
    for run in timeline.label_runs:
        label = Annotation(
            key=AnnotationKey.ACTIVITY_LABEL,
            value=dictionary.raw_labels[run.raw_label],
            span=TimeInterval.micros(run.start_us, run.end_us),
        )
        for signal in sources[run.segment].signals:
            signal.annotate(label)
    return record
