"""One Record per participant: a Source of lazy x, y and z signals per segment, its labels, and its facts."""

from functools import lru_cache, partial

import pyarrow as pa

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import RegularAxis
from timenet.types import Annotation, TimeSeriesSpec, ureg
from timenet_connectors.datasets.oxwearables.capture24.keys import AnnotationKey
from timenet_connectors.datasets.oxwearables.capture24.participants import Participant
from timenet_connectors.datasets.oxwearables.capture24.recordings import AXES, Recording, Segment, read_acceleration
from timenet_connectors.datasets.oxwearables.capture24.release import SAMPLE_RATE_HZ
from timenet_connectors.metadata import RecordMetadata


DEVICE = "Axivity AX3"

# The source writes eight significant digits, more than float32 keeps.
_SPEC = TimeSeriesSpec(
    spec_type="acceleration", name="Wrist acceleration", unit_value=ureg.standard_gravity, dtype="float64"
)


class Capture24RecordMetadata(RecordMetadata):
    """A CAPTURE-24 Record's metadata: the recording file the Record holds.

    The participant id is the shared ``subject_id``.
    """

    file: str


def record_metadata(record: Record) -> Capture24RecordMetadata:
    """Read a CAPTURE-24 Record's typed metadata back.

    Returns:
        The metadata: the participant as ``subject_id``, and the recording file.

    Raises:
        TimeFValidationError: If the Record's metadata is not CAPTURE-24 metadata.
    """  # noqa: DOC502 - raised by from_json
    return Capture24RecordMetadata.from_json(record.metadata)


class AccelerationLoader:
    """Load one acceleration column of one segment lazily, reading each recording once for all its signals.

    A loader serves the Records of one load. A Record's signals load one after another, so one cached
    recording suffices.
    """

    def __init__(self) -> None:
        self._read = lru_cache(maxsize=1)(read_acceleration)

    def __call__(self, recording: Recording, column: str, segment: Segment) -> pa.Array:
        """Return one segment's values of one acceleration column.

        Returns:
            The segment's float64 values.
        """
        values = self._read(recording)[column]
        return pa.array(values[segment.first_row : segment.first_row + segment.n_values], type=pa.float64())


def build_record(
    participant: Participant,
    recording: Recording,
    acceleration_loader: AccelerationLoader,
) -> Record:
    """Build a participant's Record: one Source of lazy x, y and z signals per segment, and its facts.

    The raw labels stay out of the dataset: the windows' classes are the only labels the tasks need.

    Args:
        participant: The participant and their ``metadata.csv`` row.
        recording: Their recording's segments.
        acceleration_loader: The loader the signals read their values through.

    Returns:
        The Record, whose values stay in the recording until read.
    """
    sources = []
    for segment_index, segment in enumerate(recording.segments):
        time_axis = RegularAxis.from_rate_hz(SAMPLE_RATE_HZ, offset_us=segment.offset_us)
        signals = tuple(
            Signal.from_loader(
                name=name,
                spec=_SPEC,
                time_axis=time_axis,
                n_values=segment.n_values,
                loader=partial(acceleration_loader, recording, name, segment),
            )
            for name in AXES
        )
        sources.append(Source(name=f"{DEVICE} segment {segment_index}", signals=signals))
    metadata = Capture24RecordMetadata(subject_id=participant.pid, file=participant.recording_path.name)
    record = Record(sources=tuple(sources), metadata=metadata.to_json())
    record.add_annotations(
        (
            Annotation(key=AnnotationKey.AGE, value=participant.age.value),
            Annotation(key=AnnotationKey.SEX, value=participant.sex.value),
            Annotation(key=AnnotationKey.RECORDING_START_LOCAL, value=recording.start_local.isoformat()),
        )
    )
    return record
