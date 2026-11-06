"""Build one CAPTURE-24 Record per participant from the checked first-pass summary.

Each segment between gaps becomes one Source with ``x``, ``y``, and ``z`` signals that load when needed.
Each run of consecutive samples with the same raw label becomes one ``activity`` annotation on its Source.
"""

from collections.abc import Callable, Iterator
from enum import StrEnum, unique
from functools import partial

import pyarrow as pa

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import RegularAxis
from timenet.types import Annotation, TimeInterval, TimeSeriesSpec, ureg
from timenet_connectors.datasets.oxwearables.capture24.source import (
    AXES,
    AXIS_TYPE,
    Acceleration,
    LabelDictionary,
    RecordingSummary,
)
from timenet_connectors.datasets.oxwearables.capture24.windowing import SAMPLE_RATE_HZ, LabelRuns, Segment


ReadAcceleration = Callable[[RecordingSummary], Acceleration]
"""A function that returns a participant's x, y and z values."""

# The paper's Methods: an Axivity AX3 on the dominant wrist, with a range of ±8 g.
_DEVICE = "Axivity AX3"
_SPEC = TimeSeriesSpec(
    spec_type="acceleration", name="Wrist acceleration", unit_value=ureg.standard_gravity, dtype="float32"
)
_RECORD_FILE_KEY = "file"
"""The Record metadata key that holds the recording's file name."""


@unique
class _Key(StrEnum):
    """The key of an annotation that this connector writes."""

    SUBJECT_IDS = "subject_ids"
    AGE_BAND = "age_band"
    """The participant's age band. Other connectors use ``age`` for an exact age in years."""
    SEX = "sex"
    RECORDING_START_LOCAL = "recording_start_local"
    """The first sample's published timestamp. The authors randomized its date and shifted its time."""
    ACTIVITY = "activity"
    """A Source annotation for consecutive samples with the same raw label."""


_AGE_BAND_DESCRIPTION = "The participant's age band in years, as metadata.csv writes it. The authors give no exact age."
_RECORDING_START_DESCRIPTION = (
    "The first sample's timestamp as the authors published it, with no zone. For anonymity they randomized "
    "the date and shifted the time by a small random amount. The time of day is as published, not an "
    "exact local time."
)


def _load_segment(acceleration: ReadAcceleration, summary: RecordingSummary, segment: Segment, axis: str) -> pa.Array:
    """Return a named axis sliced to the segment's sample range."""
    values = acceleration(summary)[axis]
    return pa.array(values[segment.first_row : segment.first_row + segment.n_values], type=AXIS_TYPE)


def _activity_annotations(runs: LabelRuns, labels: LabelDictionary) -> Iterator[Annotation]:
    """Turn a segment's raw label runs into one annotation per run.

    Yields:
        An annotation with the raw label as its value and the run as its span.
        Its metadata maps each of the six scheme names to the raw label's class in that scheme.
    """
    for start_us, end_us, code in zip(runs.starts_us.tolist(), runs.ends_us.tolist(), runs.codes.tolist(), strict=True):
        yield Annotation(
            key=_Key.ACTIVITY,
            value=labels.raw_labels[code],
            span=TimeInterval.micros(start_us, end_us),
            metadata={scheme.value: name for scheme, name in labels.scheme_classes_of_raw[code].items()},
        )


def _source(
    index: int, segment: Segment, runs: LabelRuns, labels: LabelDictionary, load: Callable[[Segment, str], pa.Array]
) -> Source:
    """Build one segment's Source, with its signals and its raw label runs.

    Returns:
        The Source, whose values stay in the recording until read.
    """
    time_axis = RegularAxis.from_rate_hz(SAMPLE_RATE_HZ, offset_us=segment.offset_us)
    signals = tuple(
        Signal.from_loader(
            name=column,
            spec=_SPEC,
            time_axis=time_axis,
            n_values=segment.n_values,
            loader=partial(load, segment, column),
        )
        for column in AXES
    )
    # Pass all runs to the Source constructor at once. Adding one run at a time copies the annotation
    # tuple for every run.
    return Source(
        name=f"{_DEVICE} segment {index}",
        signals=signals,
        annotations=tuple(_activity_annotations(runs, labels)),
    )


def build_record(summary: RecordingSummary, labels: LabelDictionary, acceleration: ReadAcceleration) -> Record:
    """Build a participant's Record with one Source per segment.

    Each Source has x, y and z signals that load when needed, and carries the segment's raw label runs
    as ``activity`` annotations.

    Args:
        summary: The recording summary from the first pass.
        labels: The label dictionary, which names each run's raw label and its classes.
        acceleration: The function that the signal loaders call to read the recording's values.

    Returns:
        The Record, whose values stay in the recording until read.
    """
    load = partial(_load_segment, acceleration, summary)
    sources = tuple(
        _source(index, segment, runs, labels, load)
        for index, (segment, runs) in enumerate(zip(summary.segments, summary.label_runs, strict=True))
    )
    participant = summary.participant
    # TODO(#677): build subject_ids and sex with the shared demographics builders after #677 lands.
    return Record(
        sources=sources,
        metadata={_RECORD_FILE_KEY: participant.path.name},
        annotations=(
            Annotation(key=_Key.SUBJECT_IDS, value=[participant.pid]),
            Annotation(key=_Key.AGE_BAND, value=participant.age_band.value, description=_AGE_BAND_DESCRIPTION),
            Annotation(key=_Key.SEX, value=participant.sex.value),
            Annotation(
                key=_Key.RECORDING_START_LOCAL,
                value=summary.start_local,
                description=_RECORDING_START_DESCRIPTION,
            ),
        ),
    )
