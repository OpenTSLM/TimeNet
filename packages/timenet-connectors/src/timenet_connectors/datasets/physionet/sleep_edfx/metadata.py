"""Build the annotations that state where a recording sits and who was in the bed.

Every function here takes values and gives annotations. Nothing here opens a file or takes a
path. ``connector.py`` reads the release and the subject tables, then passes the values it read.
A test can call these functions with values alone.

Some facts repeat across the release. The study, the night number, the sex and the drug
condition each come from a set that the release fixes. One annotation covers every sample that
states that value, and its id names the value. :class:`MetadataAnnotation` holds one instance
for each value. The writer keys its annotation table by id, and lists the samples that carry
each one. Many samples that share one annotation write one row.

The age, the note and the lights-off time measure one recording. Each sample
gets its own, and its id names the recording. An id that names a value says the annotation *is*
that value. That is right for a term of a vocabulary. It is wrong for a measurement, because a
correction must point the sample at a new id and leave the old one behind.

An annotation with a span is always per-sample. One shared annotation carries one span, and
these spans differ for each recording.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, time
import re
from typing import Protocol

from timenet.errors import TimeFValidationError
from timenet.types import US_PER_S, Annotation, TimePoint
from timenet_connectors.datasets.physionet.sleep_edfx import reader, tables


_SECONDS_PER_DAY = 24 * 60 * 60

# The patient field of an EDF header is anonymous in this release, and it keeps a sex and an age
# in its name field, as ``Female_33yr``. A header that holds neither states no second reading.
_HEADER_DEMOGRAPHICS = re.compile(r"(male|female)_(\d+)yr", re.IGNORECASE)


class MetadataAnnotation:
    """One annotation for each distinct value of a key whose values are a closed set.

    Four keys take their value from a small fixed set: ``study``, ``night``, ``sex`` and
    ``condition``. Every recording states one value of each. Without this class, each recording
    builds its own copy of the same few annotations.

    This class builds the annotation for a value one time. It gives the same instance back for
    every later recording that states that value. Two samples with the same sex then carry one
    annotation between them, and the writer stores it one time.

    The writer rejects two annotations that share an id and differ in any field. One instance
    for each value makes them equal. No caller must then keep two copies in step.

    Nothing reads the values up front. The build walks the release one time.
    """

    def __init__(self, id_prefix: str) -> None:
        self._id_prefix = id_prefix
        self._built: dict[tuple[str, str], Annotation] = {}

    def get_study(self, study: str) -> Annotation:
        """Give the annotation for one study of the release.

        Args:
            study: The name of the directory the recording sits in.

        Returns:
            The annotation for that study.
        """
        return self._get_annotation(
            "study", study, "The study of the release that recorded this, as its directory names it."
        )

    def get_night(self, night: int) -> Annotation:
        """Give the annotation for one night number.

        Args:
            night: Which of a subject's recordings this is, as the filename states it.

        Returns:
            The annotation for that night number.
        """
        return self._get_annotation("night", night, "Which of the subject's nights this recording is.")

    def get_sex(self, sex: str) -> Annotation:
        """Give the annotation for one sex.

        Args:
            sex: The decoded sex, ``F`` or ``M``. A raw sheet code must never reach this,
                because one code means the opposite thing in the other study.

        Returns:
            The annotation for that sex.
        """
        return self._get_annotation("sex", sex, "The sex of the subject, as the subject table of the study states it.")

    def get_condition(self, condition: str) -> Annotation:
        """Give the annotation for one arm of the telemetry experiment.

        Args:
            condition: The drug the subject took that night.

        Returns:
            The annotation for that condition.
        """
        return self._get_annotation(
            "condition", condition, "The drug the subject took, from the column pair of the subject table."
        )

    def count_built(self) -> dict[str, int]:
        """Give the number of distinct annotations this holder built for each key.

        Each key has a fixed set of values. A count above the size of that set means a value
        reached an id that the release does not define.

        Returns:
            The count of distinct annotations, keyed by annotation key.
        """
        return dict(Counter(key for key, _ in self._built))

    def _get_annotation(self, key: str, value: object, description: str) -> Annotation:
        """Give the one annotation of a key and a value, and build it on first sight.

        Args:
            key: The annotation key.
            value: The value, written into the id as it is stored.
            description: The description of the key, equal for every value of it.

        Returns:
            The annotation for that value.
        """
        annotation_id = f"{self._id_prefix}-{key}-{value}"
        built = self._built.get((key, annotation_id))
        if built is None:
            built = Annotation(key=key, value=value, description=description, id=annotation_id)
            self._built[key, annotation_id] = built

        return built


def build_age(sample_id: str, years: int) -> Annotation:
    """Give the age annotation of one recording.

    The age comes from the subject table and never from the EDF header. The table is the
    registry of the study, and published work joins against it.

    Args:
        sample_id: The id of the sample, which the annotation id is built on.
        years: The age in whole years that the subject table states.

    Returns:
        The age annotation, with no span.
    """
    return Annotation(
        key="age",
        value=years,
        unit="years",
        description="The age of the subject in whole years, as the subject table of the study states it.",
        id=f"{sample_id}-age",
    )


def build_recording_start_local(sample_id: str, start_time: datetime) -> Annotation:
    """Give the annotation that carries the start moment of an EDF header.

    A sample of this connector sets no ``start_time``. Every header of this release states a
    date and a time and no zone, and TimeF refuses a naive datetime for that field. This
    annotation keeps the fact the header does state. Its text carries no offset and names no
    zone, so nothing is lost and nothing is invented. A consumer who knows the zone can anchor
    the text themselves.

    Args:
        sample_id: The id of the sample, which the annotation id is built on.
        start_time: The start moment the header states, as a local wall clock.

    Returns:
        The start annotation, with the date and the time as ISO 8601 text.

    Raises:
        TimeFValidationError: If the caller passes a moment that names a zone. The value then
            carries an offset that the release never states.
    """
    if start_time.tzinfo is not None:
        raise TimeFValidationError(
            f"{sample_id}: the header start moment names the zone {start_time.tzinfo}, and this release states none"
        )

    return Annotation(
        key="recording_start_local",
        value=start_time.isoformat(timespec="seconds"),
        description="The local date and time the EDF header states for the first sample. The release names no zone.",
        id=f"{sample_id}-start-local",
    )


def build_demographics_note(sample_id: str, patient_id: str, years: int, sex: str) -> Annotation | None:
    """Compare the two readings of a subject's demographics, and note where they differ.

    The EDF header keeps an age and a sex, and so does the subject table. The two sometimes
    differ. The table wins, and this note keeps both readings. The connector README states why.
    A header with no second reading gives no note.

    The note is per-sample. It states what these two sources say about this recording, and a
    shared note claims that two recordings disagree for one reason.

    Args:
        sample_id: The id of the sample, which the annotation id is built on.
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
        key="demographics_note",
        value=(
            f"the EDF header states sex {header_sex} and age {header_years}. The subject table states "
            f"sex {sex} and age {years}. This sample carries what the table states."
        ),
        description="The EDF header and the subject table disagree here on age or sex. See the connector README.",
        id=f"{sample_id}-demographics_note",
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
        sample_id: str,
        metadata_annotation: MetadataAnnotation,
        recording: RecordingIdentity,
        header: reader.EdfHeader,
    ) -> list[Annotation]:
        """Give every metadata annotation one recording carries.

        Args:
            sample_id: The sample these annotations belong to.
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
        return _build_annotations(sample_id, metadata_annotation, recording, header, row, night)


def _build_annotations(  # noqa: PLR0913, PLR0917
    sample_id: str,
    metadata_annotation: MetadataAnnotation,
    recording: RecordingIdentity,
    header: reader.EdfHeader,
    row: tables.SubjectRow,
    night: tables.SubjectNight,
) -> list[Annotation]:
    """Give every metadata annotation one recording carries.

    A value from a closed set comes from ``metadata_annotation``, so every sample that states
    it carries one instance. A value measured for this recording is built here.

    PLR0913, PLR0917: each argument is one source a sample draws a fact from, and the caller has
    already resolved every one of them. To group them would name the group after this call.

    Args:
        sample_id: The sample these annotations belong to.
        metadata_annotation: The holder of the annotations whose values are a closed set.
        recording: What the filename of the recording states.
        header: The header of its signal file.
        row: The table row of the person it belongs to.
        night: The night of that row.

    Returns:
        The annotations, in a stable order.
    """
    built = [
        metadata_annotation.get_study(recording.study),
        metadata_annotation.get_night(recording.night),
        metadata_annotation.get_sex(row.sex),
        build_age(sample_id, row.age),
        build_recording_start_local(sample_id, header.start_time),
    ]

    if night.condition is not None:
        built.append(metadata_annotation.get_condition(night.condition))

    note = build_demographics_note(sample_id, header.patient_id, row.age, row.sex)
    if note is not None:
        built.append(note)

    built.append(
        build_lights_off(sample_id, night.lights_off, find_lights_off_offset(header.start_time, night.lights_off))
    )

    return built


def find_lights_off_offset(start_time: datetime, lights_off: time) -> int:
    """Place a stated clock time on the timeline of a recording.

    The sheets state a clock time and no date. A recording starts before the lights went off,
    and the lights-off time falls in the night that follows. The offset wraps forward across
    midnight. The result is the first occurrence of that clock time at or after the start.

    Args:
        start_time: The start moment the EDF header states.
        lights_off: The clock time the subject table states.

    Returns:
        The offset from the first sample, in microseconds.
    """
    start_of_day = start_time.hour * 60 * 60 + start_time.minute * 60 + start_time.second
    lights_off_of_day = lights_off.hour * 60 * 60 + lights_off.minute * 60 + lights_off.second
    return ((lights_off_of_day - start_of_day) % _SECONDS_PER_DAY) * US_PER_S


def build_lights_off(sample_id: str, at: time, offset_microseconds: int) -> Annotation:
    """Give the lights-off annotation of one recording.

    The annotation carries both forms. The offset places the night on the timeline, and the
    value keeps the clock time that the sheet states. The derivation stays checkable, because
    the stated fact is still there.

    The span names no time series. Lights off is a fact about the room and not a reading taken
    from a channel, so TimeF checks it against the span the sample declares.

    Args:
        sample_id: The id of the sample, which the annotation id is built on.
        at: The clock time the subject table states.
        offset_microseconds: Where that clock time falls, from :func:`find_lights_off_offset`.

    Returns:
        The lights-off annotation, as a point on the recording timeline.
    """
    return Annotation(
        key="lights_off",
        value=at.isoformat(timespec="seconds"),
        span=TimePoint.micros(offset_microseconds),
        description="The clock time the subject turned the light out, as the subject table of the study states it.",
        id=f"{sample_id}-lights_off",
    )
