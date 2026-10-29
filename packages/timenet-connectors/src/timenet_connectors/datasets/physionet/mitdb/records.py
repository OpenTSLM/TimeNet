"""One Record per MIT-BIH record: its two leads as lazy signals, the subject's facts, and every reference label."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import RegularAxis
from timenet.json import JsonMapping
from timenet.types import Annotation, TimeInterval, TimePoint, TimeSeriesSpec, ureg
from timenet_connectors.bases.physionet import WfdbHeader
from timenet_connectors.datasets.physionet.mitdb.annotations import ReferenceAnnotations
from timenet_connectors.datasets.physionet.mitdb.header import RecordHeader
from timenet_connectors.datasets.physionet.mitdb.keys import SUBTYPE, AnnotationKey
from timenet_connectors.datasets.physionet.mitdb.release import SAMPLE_RATE_HZ
from timenet_connectors.metadata import RecordMetadata


LeadLoader = Callable[[Path, WfdbHeader, int], Callable[[], pa.Array]]
"""Builds the lazy loader of one lead: ``BasePhysioNetConnector._lead_loader``."""

DEVICE = "Del Mar Avionics 445"
"""The two-channel Holter recorder model every record was made on."""

SUBJECT_OF = {"202": "201"}
"""The records whose subject is another record's: records 201 and 202 came from the same subject, as the
authors' introduction states. Every other record is its own subject."""

AXIS = RegularAxis.from_rate_hz(SAMPLE_RATE_HZ)
"""Every lead's axis. A sample's time is ``AXIS.time_offset_us(sample)``: 360 Hz is 25000/9 us a sample, so
the time is floored to whole microseconds, and ``AXIS.index_at_or_after`` maps it back to the sample."""

# The ADC samples 11 bits over 10 mV, 200 units per mV, so float32 holds every value WFDB reads.
_SPEC = TimeSeriesSpec(spec_type="ecg", name="ECG lead", unit_value=ureg.millivolt)


class MitdbRecordMetadata(RecordMetadata):
    """MIT-BIH's Record metadata: the record it holds, and its header's comment lines.

    The subject is the shared ``subject_id``.
    """

    record: str
    """The record name, such as ``100``."""
    leads: tuple[str, ...]
    """The leads in the header's order, upper signal first. A Record lists its signals by name, so this keeps
    which signal the authors' notes and the signal quality subtype call upper."""
    comments: tuple[str, ...]
    """The header's comment lines, as written: age, sex and undocumented fields, the medications, then the
    authors' notes on the record, which name its arrhythmias."""


def record_metadata(record: Record) -> MitdbRecordMetadata:
    """Read an MIT-BIH Record's typed metadata back.

    Returns:
        The metadata, whose ``subject_id`` is the subject.

    Raises:
        TimeFValidationError: If the Record has no MIT-BIH metadata.
    """  # noqa: DOC502 - raised by from_json
    return MitdbRecordMetadata.from_json(record.metadata)


@dataclass(frozen=True)
class ParsedRecord:
    """One record, built: its Record, and the beat and rhythm annotations its tasks point at."""

    header: RecordHeader
    reference: ReferenceAnnotations
    record: Record
    beats: tuple[Annotation, ...]
    """The Record's beat annotations, in the order of ``reference.beats``."""
    rhythms: tuple[Annotation, ...]
    """The Record's rhythm annotations, in the order of ``reference.rhythm_episodes``."""


def _subtype(subtype: int) -> JsonMapping:
    """Return an annotation's metadata: its subtype where the source writes one."""
    return {SUBTYPE: subtype} if subtype else {}


def build_record(header: RecordHeader, reference: ReferenceAnnotations, lead_loader: LeadLoader) -> ParsedRecord:
    """Build a record's Record: one Source of its two leads, its subject's facts, and its reference labels.

    Every label's time is its sample's time on the leads' 360 Hz axis. A signal quality episode becomes one
    annotation on each signal, valued by that signal's quality.

    Args:
        header: The record's header.
        reference: Its reference annotations.
        lead_loader: Builds the lazy loader of one lead.

    Returns:
        The Record, whose values stay in the ``.dat`` until read, and the annotations its tasks point at.

    Raises:
        TimeFValidationError: If a label falls outside the record.
    """  # noqa: DOC502 - raised by add_annotations
    at = AXIS.time_offset_us
    signals = tuple(
        Signal.from_loader(
            name=lead,
            spec=_SPEC,
            time_axis=AXIS,
            n_values=header.wfdb.sig_len,
            loader=lead_loader(header.base, header.wfdb, index),
        )
        for index, lead in enumerate(header.wfdb.sig_name)
    )
    metadata = MitdbRecordMetadata(
        subject_id=SUBJECT_OF.get(header.name, header.name),
        record=header.name,
        leads=tuple(header.wfdb.sig_name),
        comments=header.comments,
    )
    record = Record(sources=(Source(name=DEVICE, signals=signals),), metadata=metadata.to_json())
    facts = [
        Annotation(key=AnnotationKey.SEX, value=header.sex.value),
        Annotation(key=AnnotationKey.MEDICATIONS, value=header.medications),
    ]
    if header.age is not None:
        facts.insert(0, Annotation(key=AnnotationKey.AGE, value=header.age, unit="year"))
    beats = [
        Annotation(
            key=AnnotationKey.BEAT,
            value=beat.code.value,
            span=TimePoint.micros(at(beat.sample)),
            metadata=_subtype(beat.subtype),
        )
        for beat in reference.beats
    ]
    rhythms = [
        Annotation(
            key=AnnotationKey.RHYTHM,
            value=episode.label.value,
            span=TimeInterval.micros(at(episode.start), at(episode.end)),
        )
        for episode in reference.rhythm_episodes
    ]
    # A signal's own annotation is scoped to it. `read_reference` keeps every episode within the record.
    for episode in reference.quality_episodes:
        for signal, quality in zip(signals, episode.qualities, strict=True):
            signal.annotate(
                Annotation(
                    key=AnnotationKey.SIGNAL_QUALITY,
                    value=quality.value,
                    span=TimeInterval.micros(at(episode.start), at(episode.end)),
                    metadata=_subtype(episode.subtype),
                )
            )
    events = [
        Annotation(
            key=AnnotationKey.EVENT,
            value=event.code.value,
            span=TimePoint.micros(at(event.sample)),
            metadata=_subtype(event.subtype),
        )
        for event in reference.events
    ]
    comments = [
        Annotation(key=AnnotationKey.COMMENT, value=comment.text, span=TimePoint.micros(at(comment.sample)))
        for comment in reference.comments
    ]
    attached = record.add_annotations([*facts, *beats, *rhythms, *events, *comments], warn_when_outside=False)
    first_beat = len(facts)
    first_rhythm = first_beat + len(beats)
    return ParsedRecord(
        header=header,
        reference=reference,
        record=record,
        beats=attached[first_beat:first_rhythm],
        rhythms=attached[first_rhythm : first_rhythm + len(rhythms)],
    )
