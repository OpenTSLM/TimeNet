"""Build one record for each Sleep-EDFx recording, with its sleep stages and its metadata.

The study, the night, the sex, and the drug condition each take a value from a small fixed set.
:func:`build_closed_set` builds the annotation of such a value for each record, and the writer
stores equal content once. The age, the start clock, the note, and the lights-off time belong to one
recording.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, time
import re

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.errors import TimeFValidationError, TimeNetConnectorError
from timenet.types import US_PER_S, Annotation, TimeInterval, TimePoint
from timenet_connectors.bases.edf import reader
from timenet_connectors.datasets.physionet.sleep_edfx import source
from timenet_connectors.datasets.physionet.sleep_edfx.labels import AnnotationKey
from timenet_connectors.demographics import Sex, age_annotation, subject_ids_annotation


# The scoring followed the 1968 Rechtschaffen and Kales manual, which scores from EEG, EOG and EMG
# together. These are the four signals of the release that the technician read.
_signal_names_for_annotations = ("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental")

# The patient field of an EDF header keeps a second reading of the sex and the age, as ``Female_33yr``.
_HEADER_DEMOGRAPHICS = re.compile(r"(male|female)_(\d+)yr", re.IGNORECASE)

_DESCRIPTIONS: dict[AnnotationKey, str] = {
    AnnotationKey.STUDY: "The study of the release that recorded this, as its directory names it.",
    AnnotationKey.NIGHT: "Which of the subject's nights this recording is.",
    AnnotationKey.SEX: "The sex of the subject, as the subject table of the study states it.",
    AnnotationKey.CONDITION: "The drug the subject took, from the column pair of the subject table.",
}


def add_record(
    dataset: TimeFDataset,
    recording: source.SleepEdfxRecording,
    files: source.RecordingFiles,
    subject_tables: "SubjectTables",
) -> Record:
    """Add the record of one recording, with its sleep stages and its metadata annotations.

    Returns:
        The added record.
    """
    # The scoring is padded to a full day, so the session may end after the signals stop.
    session_end = files.signal_end + measure_overrun_microseconds(files.entries, files.signal_end)
    sleep_stages = build_sleep_stages(files.entries)
    recording_metadata = subject_tables.build_annotations(recording, files.header)

    record = Record(
        sources=(Source(name="Sleep-EDF polysomnograph", signals=files.series),),
        time_span=TimeInterval.micros(0, session_end),
        metadata={"recording_id": recording.recording_id},
    )
    dataset.add_record(record=record)
    record.annotate(subject_ids_annotation([recording.subject_id]))
    attach_sleep_stages(recording.recording_id, files.series, sleep_stages)
    record.add_annotations(recording_metadata)
    return record


def _signals_of_annotation(recording_id: str, series: tuple[Signal, ...]) -> tuple[Signal, ...]:
    by_signal = {one.name: one for one in series}
    missing = [signal for signal in _signal_names_for_annotations if signal not in by_signal]
    if missing:
        raise TimeNetConnectorError(f"{recording_id}: does not hold the signals {missing}")

    return tuple(by_signal[signal] for signal in _signal_names_for_annotations)


def measure_overrun_microseconds(entries: tuple[reader.EdfAnnotation, ...], end_microseconds: int) -> int:
    """Give how far a scoring reaches past the last recorded sample.

    Returns:
        The overrun in microseconds, or 0 when the scoring fits.
    """
    if not entries:
        return 0

    last = max(entry.onset_microseconds + entry.duration_microseconds for entry in entries)
    return max(0, last - end_microseconds)


def build_sleep_stages(entries: tuple[reader.EdfAnnotation, ...]) -> list[Annotation]:
    """Give one sleep-stage annotation for each scoring entry, in file order.

    Returns:
        One annotation for each entry, with the onset and the duration that the file states.
    """
    return [
        Annotation(
            key=AnnotationKey.SLEEP_STAGE.value,
            value=entry.label,
            span=TimeInterval.micros(
                entry.onset_microseconds,
                entry.onset_microseconds + entry.duration_microseconds,
            ),
        )
        for entry in entries
    ]


def attach_sleep_stages(recording_id: str, series: tuple[Signal, ...], stages: list[Annotation]) -> None:
    """Attach each sleep stage to the four signals that the technician scored from.

    Raises:
        TimeNetConnectorError: If the recording does not hold one of those signals.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by _signals_of_annotation
    for signal in _signals_of_annotation(recording_id, series):
        for stage in stages:
            signal.annotate(stage)


def build_closed_set(key: AnnotationKey, value: object) -> Annotation:
    """Give the annotation of a value of the study, the night, the sex, or the condition.

    A sex must be a decoded :class:`~timenet_connectors.demographics.Sex` value, never a raw sheet
    code, because one code means the opposite sex in the other study.

    Returns:
        The annotation for that value.
    """
    return Annotation(key=key.value, value=value, description=_DESCRIPTIONS[key])


def build_age(years: int) -> Annotation:
    """Give the age annotation of one recording, from the subject table.

    Returns:
        The age annotation.
    """
    return age_annotation(
        years, description="The age of the subject in whole years, as the subject table of the study states it."
    )


def build_recording_start_local(recording_id: str, start_time: datetime) -> Annotation:
    """Give the start clock that an EDF header states, as local ISO 8601 text with no zone.

    The record sets no ``start_time``, because the release names no zone and TimeF refuses a naive
    datetime there.

    Returns:
        The start annotation.

    Raises:
        TimeFValidationError: If the caller passes a moment with a zone.
    """
    if start_time.tzinfo is not None:
        raise TimeFValidationError(
            f"{recording_id}: the header start moment names the zone {start_time.tzinfo}, and this release states none"
        )

    return Annotation(
        key=AnnotationKey.RECORDING_START_LOCAL.value,
        value=start_time.isoformat(timespec="seconds"),
        description="The local date and time the EDF header states for the first sample. The release names no zone.",
    )


def build_demographics_note(patient_id: str, years: int, sex: Sex) -> Annotation | None:
    """Note where the EDF header and the subject table disagree on the age or the sex.

    The table wins. The connector README states why.

    Returns:
        The note with both readings, or ``None`` where they agree or the header states none.
    """
    match = _HEADER_DEMOGRAPHICS.search(patient_id)
    if match is None:
        return None

    header_sex = Sex.FEMALE if match[1].lower() == "female" else Sex.MALE
    header_years = int(match[2])
    kinds = tuple(kind for kind, differs in (("age", header_years != years), ("sex", header_sex != sex)) if differs)
    if not kinds:
        return None

    return Annotation(
        key=AnnotationKey.DEMOGRAPHICS_NOTE.value,
        value=(
            f"the EDF header states sex {header_sex.value} and age {header_years}. The subject table states "
            f"sex {sex.value} and age {years}. This record carries what the table states."
        ),
        description="The EDF header and the subject table disagree here on age or sex. See the connector README.",
    )


class SubjectTables:
    """The subject table of each study, parsed once, and the join from a recording to its row."""

    def __init__(
        self, rows_by_study: Mapping[str, Sequence[Sequence[object]]], shapes: Mapping[str, source.SheetShape]
    ) -> None:
        self._shapes = dict(shapes)
        self._tables = {study: source.parse_subject_table(shapes[study], rows) for study, rows in rows_by_study.items()}

    def build_annotations(self, recording: source.SleepEdfxRecording, header: reader.EdfHeader) -> list[Annotation]:
        """Give every metadata annotation of one recording, in a stable order.

        Returns:
            The annotations.

        Raises:
            TimeNetConnectorError: If the table of that study holds no row for the recording.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by find_subject_row and find_night
        shape = self._shapes[recording.study]
        row = source.find_subject_row(
            self._tables[recording.study], shape, recording.subject_number, recording.night, recording.recording_id
        )
        night = source.find_night(row, recording.night, recording.recording_id)

        built = [
            build_closed_set(AnnotationKey.STUDY, recording.study),
            build_closed_set(AnnotationKey.NIGHT, recording.night),
            build_closed_set(AnnotationKey.SEX, row.sex.value),
            build_age(row.age),
            build_recording_start_local(recording.recording_id, header.start_time),
        ]

        if night.condition is not None:
            built.append(build_closed_set(AnnotationKey.CONDITION, night.condition))

        note = build_demographics_note(header.patient_id, row.age, row.sex)
        if note is not None:
            built.append(note)

        built.append(build_lights_off(night.lights_off, header.start_time))

        return built


def build_lights_off(at: time, start_time: datetime) -> Annotation:
    """Give the lights-off moment of one recording, as a point on its timeline.

    The sheet states a clock time and no date. The lights go out after the recording starts, so
    the offset is the first time the clock shows that value at or after the start, across midnight.

    Returns:
        The lights-off annotation, with the clock time as its value.
    """
    start_of_day = start_time.hour * 60 * 60 + start_time.minute * 60 + start_time.second
    lights_off_of_day = at.hour * 60 * 60 + at.minute * 60 + at.second
    offset_microseconds = ((lights_off_of_day - start_of_day) % source.SECONDS_PER_DAY) * US_PER_S

    return Annotation(
        key=AnnotationKey.LIGHTS_OFF.value,
        value=at.isoformat(timespec="seconds"),
        span=TimePoint.micros(offset_microseconds),
        description="The clock time the subject turned the light out, as the subject table of the study states it.",
    )
