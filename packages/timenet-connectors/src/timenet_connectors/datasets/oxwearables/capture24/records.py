"""Build one CAPTURE-24 Record per participant from the checked first-pass summary.

Each segment between gaps becomes one Source with ``x``, ``y``, and ``z`` signals that load when needed.
Each run of consecutive samples with the same raw label becomes one ``activity`` annotation on its Source.
"""

from collections.abc import Callable, Iterator, Mapping
from enum import StrEnum, unique
from functools import partial
from pathlib import Path

import pyarrow as pa

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import RegularAxis
from timenet.types import Annotation, TimeInterval, TimeSeriesSpec, ureg
from timenet_connectors.datasets.oxwearables.capture24.source import (
    AXES,
    LabelDictionary,
    RecordingSummary,
)
from timenet_connectors.datasets.oxwearables.capture24.windowing import SAMPLE_RATE_HZ, LabelRuns, Segment
from timenet_connectors.demographics import sex_annotation, subject_ids_annotation
from timenet_connectors.sources.channels import SharedChannelLoader


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

    AGE_BAND = "age_band"
    """The participant's age band. Other connectors use ``age`` for an exact age in years."""
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


def _load_segment(channel_loader: Callable[[], pa.Array], segment: Segment) -> pa.Array:
    """Return a channel sliced to the segment's sample range."""
    return channel_loader().slice(segment.first_row, segment.n_values)


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
    index: int,
    segment: Segment,
    runs: LabelRuns,
    labels: LabelDictionary,
    channel_loaders: Mapping[str, Callable[[], pa.Array]],
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
            loader=partial(_load_segment, channel_loaders[column], segment),
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


def build_record(
    summary: RecordingSummary, labels: LabelDictionary, shared_channels: SharedChannelLoader[Path]
) -> Record:
    """Build a participant's Record with one Source per segment.

    Each Source has x, y and z signals that load when needed, and carries the segment's raw label runs
    as ``activity`` annotations.

    Args:
        summary: The recording summary from the first pass.
        labels: The label dictionary, which names each run's raw label and its classes.
        shared_channels: The conversion's shared recording reader and channel cache.

    Returns:
        The Record, whose values stay in the recording until read.
    """
    channel_loaders = {
        axis: shared_channels.channel_loader(summary.participant.path, index) for index, axis in enumerate(AXES)
    }
    sources = tuple(
        _source(index, segment, runs, labels, channel_loaders)
        for index, (segment, runs) in enumerate(zip(summary.segments, summary.label_runs, strict=True))
    )
    participant = summary.participant
    return Record(
        sources=sources,
        metadata={_RECORD_FILE_KEY: participant.path.name},
        annotations=(
            subject_ids_annotation([participant.pid]),
            Annotation(key=_Key.AGE_BAND, value=participant.age_band.value, description=_AGE_BAND_DESCRIPTION),
            sex_annotation(participant.sex),
            Annotation(
                key=_Key.RECORDING_START_LOCAL,
                value=summary.start_local,
                description=_RECORDING_START_DESCRIPTION,
            ),
        ),
    )
