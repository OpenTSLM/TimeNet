"""Build the annotations that state where a recording sits and who was in the bed.

Every function here takes values and gives annotations. Nothing here opens a file or takes a
path. ``connector.py`` reads the release and the subject tables, then passes the values it read.
A test can call these functions with values alone.

Some facts repeat across the release. The study, the night number, the sex, the drug condition
and the prefiltering of the signals each come from a set that the release fixes. One annotation
covers every record that states that value. :class:`MetadataAnnotation` holds one instance for
each value. The writer keys its annotation table by id, and lists the records that carry each
one. Many records that share one annotation write one row.

The age, the note and the lights-off time measure one recording. Each record gets its own.

An annotation with a span is always per-record. One shared annotation carries one span, and
these spans differ for each recording.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, time
import re
from typing import Protocol

from timenet.errors import TimeFValidationError
from timenet.types import US_PER_S, Annotation, TimePoint
from timenet_connectors.bases.edf import reader
from timenet_connectors.datasets.physionet.sleep_edfx import tables
from timenet_connectors.datasets.physionet.sleep_edfx.keys import AnnotationKey


# The patient field of an EDF header is anonymous in this release, and it keeps a sex and an age
# in its name field, as ``Female_33yr``. A header that holds neither states no second reading.
_HEADER_DEMOGRAPHICS = re.compile(r"(male|female)_(\d+)yr", re.IGNORECASE)

# What each key of a closed set states. Every value of one key states the same thing, so the
# description belongs to the key and not to the annotation built for a value.
_DESCRIPTIONS: dict[AnnotationKey, str] = {
    AnnotationKey.STUDY: "The study of the release that recorded this, as its directory names it.",
    AnnotationKey.NIGHT: "Which of the subject's nights this recording is.",
    AnnotationKey.SEX: "The sex of the subject, as the subject table of the study states it.",
    AnnotationKey.CONDITION: "The drug the subject took, from the column pair of the subject table.",
    AnnotationKey.PREFILTERING: (
        "What each signal writes in its EDF prefiltering field, from a signal name to that "
        "string. Most of these strings state an analog filter. A marker signal states how to "
        "use its button."
    ),
}


class MetadataAnnotation:
    """One annotation for each distinct value of a key whose values are a closed set.

    Five keys take their value from a small fixed set: ``study``, ``night``, ``sex``,
    ``condition`` and ``prefiltering``. Every recording states one value of each. Without this
    class, each recording builds its own copy of the same few annotations.

    This class builds the annotation for a value one time. It gives the same instance back for
    every later recording that states that value. Two records with the same sex then carry one
    annotation between them, and the writer stores it one time.

    Each instance carries one id. Every record that states the value points at that id, so the
    writer stores the annotation one time and no caller must keep two copies in step.

    Nothing reads the values up front. The build walks the release one time.
    """

    def __init__(self) -> None:
        self._built: dict[tuple[str, str], Annotation] = {}

    def get_annotation(self, key: AnnotationKey, value: object) -> Annotation:
        """Give the one annotation of a key and a value, and build it on first sight.

        Args:
            key: The annotation key. Its description is the same for every value of it.
            value: The value, as it is stored. A sex must be the decoded ``F`` or ``M``. A raw
                sheet code must never reach this, because one code means the opposite thing in
                the other study.

        Returns:
            The annotation for that value.
        """
        built = self._built.get((key, str(value)))
        if built is None:
            built = Annotation(key=key, value=value, description=_DESCRIPTIONS[key])
            self._built[key, str(value)] = built

        return built


def build_age(years: int) -> Annotation:
    """Give the age annotation of one recording.

    The age comes from the subject table and never from the EDF header. The table is the
    registry of the study, and published work joins against it.

    Args:
        years: The age in whole years that the subject table states.

    Returns:
        The age annotation, with no span.
    """
    return Annotation(
        key=AnnotationKey.AGE,
        value=years,
        unit="years",
        description="The age of the subject in whole years, as the subject table of the study states it.",
    )


def build_recording_start_local(record_id: str, start_time: datetime) -> Annotation:
    """Give the annotation that carries the start moment of an EDF header.

    A record of this connector sets no ``start_time``. Every header of this release states a
    date and a time and no zone, and TimeF refuses a naive datetime for that field. This
    annotation keeps the fact the header does state. Its text carries no offset and names no
    zone, so nothing is lost and nothing is invented. A consumer who knows the zone can anchor
    the text themselves.

    Args:
        record_id: The id of the record, which the error message names.
        start_time: The start moment the header states, as a local wall clock.

    Returns:
        The start annotation, with the date and the time as ISO 8601 text.

    Raises:
        TimeFValidationError: If the caller passes a moment that names a zone. The value then
            carries an offset that the release never states.
    """
    if start_time.tzinfo is not None:
        raise TimeFValidationError(
            f"{record_id}: the header start moment names the zone {start_time.tzinfo}, and this release states none"
        )

    return Annotation(
        key=AnnotationKey.RECORDING_START_LOCAL,
        value=start_time.isoformat(timespec="seconds"),
        description="The local date and time the EDF header states for the first sample. The release names no zone.",
    )


def build_prefiltering(metadata_annotation: MetadataAnnotation, header: reader.EdfHeader) -> tuple[Annotation, ...]:
    """Give the annotation that keeps the EDF prefiltering field of each signal.

    EDF states one prefiltering string for each signal, and this release uses that field to say
    what no other field says: which study rectified its submental EMG, and what analog filter a
    signal passed through. A marker signal states an instruction for its button there, so not
    every string of the map is a filter.

    Every recording of a study states the same strings, so the annotation is shared. The map is
    keyed in signal-name order, so the sharing does not depend on the order a header lists its
    signals.

    Args:
        metadata_annotation: The holder of the annotations whose values are a closed set.
        header: The header of the signal file.

    Returns:
        One annotation whose value maps a signal name to its prefiltering string, or nothing
        when no signal of the file states one.
    """
    paired = zip(header.signals, header.prefiltering, strict=True)
    stated = dict(sorted((signal, text) for signal, text in paired if text))
    if not stated:
        return ()

    return (metadata_annotation.get_annotation(AnnotationKey.PREFILTERING, stated),)


def build_demographics_note(patient_id: str, years: int, sex: str) -> Annotation | None:
    """Compare the two readings of a subject's demographics, and note where they differ.

    The EDF header keeps an age and a sex, and so does the subject table. The two sometimes
    differ. The table wins, and this note keeps both readings. The connector README states why.
    A header with no second reading gives no note.

    The note is per-record. It states what these two sources say about this recording, and a
    shared note claims that two recordings disagree for one reason.

    Args:
        patient_id: The patient field of the EDF header.
        years: The age the subject table states.
        sex: The sex the subject table states.

    Returns:
        The note that records both readings, or ``None`` where the two agree.
    """
    match = _HEADER_DEMOGRAPHICS.search(patient_id)
    if match is None:
        return None

    header_sex = "F" if match[1].lower() == "female" else "M"
    header_years = int(match[2])
    kinds = tuple(kind for kind, differs in (("age", header_years != years), ("sex", header_sex != sex)) if differs)
    if not kinds:
        return None

    return Annotation(
        key=AnnotationKey.DEMOGRAPHICS_NOTE,
        value=(
            f"the EDF header states sex {header_sex} and age {header_years}. The subject table states "
            f"sex {sex} and age {years}. This record carries what the table states."
        ),
        description="The EDF header and the subject table disagree here on age or sex. See the connector README.",
    )


class RecordingIdentity(Protocol):
    """What the filename of a recording states, as this module needs it.

    ``SleepEdfxRecording`` in ``connector.py`` matches this. The protocol keeps the import one
    way: the connector knows this module, and this module knows no connector.
    """

    study: str
    subject_number: int
    night: int
    recording_id: str


class SubjectTables:
    """The table of each study, read one time, and the join from a recording to its row.

    ``convert`` builds one of these before its loop and asks it for the annotations of each
    recording. The description of a sheet states the key, so the caller passes the numbers the
    filename states and never learns how a sheet keys itself.
    """

    def __init__(
        self, rows_by_study: Mapping[str, Sequence[Sequence[object]]], shapes: Mapping[str, tables.SheetShape]
    ) -> None:
        self._shapes = dict(shapes)
        self._tables = {study: tables.parse_subject_table(shapes[study], rows) for study, rows in rows_by_study.items()}

    def build_annotations(
        self,
        record_id: str,
        metadata_annotation: MetadataAnnotation,
        recording: RecordingIdentity,
        header: reader.EdfHeader,
    ) -> list[Annotation]:
        """Give every metadata annotation one recording carries.

        A value from a closed set comes from ``metadata_annotation``, so every record that
        states it carries one instance. A value measured for this recording is built here.

        Args:
            record_id: The record these annotations belong to.
            metadata_annotation: The holder of the annotations whose values are a closed set.
            recording: What the filename of the recording states.
            header: The header of its signal file, which keeps a clock and a second reading.

        Returns:
            The annotations, in a stable order.

        Raises:
            TimeFFormatError: If the table of that study holds no row for the recording.
        """  # noqa: DOC502 (raised by find_subject_row and find_night, not directly here)
        shape = self._shapes[recording.study]
        row = tables.find_subject_row(
            self._tables[recording.study], shape, recording.subject_number, recording.night, recording.recording_id
        )
        night = tables.find_night(row, recording.night, recording.recording_id)

        built = [
            metadata_annotation.get_annotation(AnnotationKey.STUDY, recording.study),
            metadata_annotation.get_annotation(AnnotationKey.NIGHT, recording.night),
            metadata_annotation.get_annotation(AnnotationKey.SEX, row.sex),
            build_age(row.age),
            build_recording_start_local(record_id, header.start_time),
        ]
        built.extend(build_prefiltering(metadata_annotation, header))

        if night.condition is not None:
            built.append(metadata_annotation.get_annotation(AnnotationKey.CONDITION, night.condition))

        note = build_demographics_note(header.patient_id, row.age, row.sex)
        if note is not None:
            built.append(note)

        built.append(build_lights_off(night.lights_off, header.start_time))

        return built


def build_lights_off(at: time, start_time: datetime) -> Annotation:
    """Give the lights-off annotation of one recording.

    The annotation carries both forms. The offset places the night on the timeline, and the
    value keeps the clock time that the sheet states. The derivation stays checkable, because
    the stated fact is still there.

    The sheets state a clock time and no date. A recording starts before the lights went off,
    and the lights-off time falls in the night that follows. The offset wraps forward across
    midnight, so it is the first occurrence of that clock time at or after the start.

    The span names no time series. Lights off is a fact about the room and not a reading taken
    from a signal, so TimeF checks it against the span the record declares.

    Args:
        at: The clock time the subject table states.
        start_time: The start moment the EDF header states, which the offset is measured from.

    Returns:
        The lights-off annotation, as a point on the recording timeline.
    """
    start_of_day = start_time.hour * 60 * 60 + start_time.minute * 60 + start_time.second
    lights_off_of_day = at.hour * 60 * 60 + at.minute * 60 + at.second
    offset_microseconds = ((lights_off_of_day - start_of_day) % tables.SECONDS_PER_DAY) * US_PER_S

    return Annotation(
        key=AnnotationKey.LIGHTS_OFF,
        value=at.isoformat(timespec="seconds"),
        span=TimePoint.micros(offset_microseconds),
        description="The clock time the subject turned the light out, as the subject table of the study states it.",
    )
