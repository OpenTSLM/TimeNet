"""Read the Sleep-EDFx release: its recordings, their scorings, and the two subject tables.

The two subject sheets have different shapes. A :class:`SheetShape` states where each sheet keeps
its columns, so one parser reads both. The sheets also code sex with opposite meanings, so each
shape carries its own code map, and a raw code never leaves this module.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import time
from pathlib import Path
import re

from timenet.dataset import Signal
from timenet.errors import TimeNetConnectorError, TimeNetDownloadError
from timenet_connectors.bases import excel
from timenet_connectors.bases.edf import reader, timeseries
from timenet_connectors.bases.excel import decode_day_fraction_as_time, decode_whole_number
from timenet_connectors.datasets.physionet.sleep_edfx.labels import SPECS, Condition
from timenet_connectors.demographics import Sex


# The full release, as one zip on the open S3 bucket of PhysioNet. boto3 reads it anonymously.
SLEEP_EDFX_ZIP_URL = "s3://physionet-open/sleep-edfx/sleep-edfx-1.0.0.zip"
SLEEP_EDFX_ZIP_SHA256 = "a6d978fd0f9fd4693c05222f667f4d532169f8dac65ea201168b6e11ae3a80a0"

_CASSETTE_STUDY = "sleep-cassette"
_TELEMETRY_STUDY = "sleep-telemetry"

# The three characters that each study writes at the front of its recording ids.
_STUDY_CODES = {_CASSETTE_STUDY: "SC4", _TELEMETRY_STUDY: "ST7"}

# A recording id: the study code, two subject digits, the night, the recorder, and a zero.
_RECORDING_ID_SHAPE = re.compile(r"(?P<code>.{3})(?P<subject>[0-9]{2})(?P<night>[0-9]).{2}")

_NIGHTS = (1, 2)

CASSETTE_TABLE_NAME = "SC-subjects.xls"
TELEMETRY_TABLE_NAME = "ST-subjects.xls"

SECONDS_PER_DAY = 24 * 60 * 60


@dataclass(frozen=True)
class NightColumns:
    """The columns of one night in a sheet, and the drug condition that their position states."""

    night: int
    lights_off: int
    condition: str | None


@dataclass(frozen=True)
class SheetShape:
    """Where one subject sheet keeps its columns, and how it keys its rows."""

    table_name: str
    header_rows: int
    subject: int
    age: int
    sex: int
    sex_codes: Mapping[int, Sex]
    keyed_by_night: bool
    nights: tuple[NightColumns, ...]


@dataclass(frozen=True)
class SubjectNight:
    """One night of a subject."""

    night: int
    condition: str | None
    lights_off: time


@dataclass(frozen=True)
class SubjectRow:
    """What one row of a subject sheet states about a subject and their nights."""

    subject: int
    age: int
    sex: Sex
    nights: tuple[SubjectNight, ...]


# The cassette sheet has one row for each recording, keyed by subject and night.
CASSETTE_SHEET = SheetShape(
    table_name=CASSETTE_TABLE_NAME,
    header_rows=1,
    subject=0,
    age=2,
    sex=3,
    sex_codes={1: Sex.FEMALE, 2: Sex.MALE},
    keyed_by_night=True,
    nights=(NightColumns(night=1, lights_off=4, condition=None),),
)

# The telemetry sheet has one row for each subject, with both nights side by side. Its header
# cells are merged, so the parser takes fixed column positions.
TELEMETRY_SHEET = SheetShape(
    table_name=TELEMETRY_TABLE_NAME,
    header_rows=2,
    subject=0,
    age=1,
    sex=2,
    sex_codes={1: Sex.MALE, 2: Sex.FEMALE},
    keyed_by_night=False,
    nights=(
        NightColumns(night=3, lights_off=4, condition=Condition.PLACEBO),
        NightColumns(night=5, lights_off=6, condition=Condition.TEMAZEPAM),
    ),
)

# Each study with the subject sheet that describes it, in record order.
STUDIES = ((_CASSETTE_STUDY, CASSETTE_SHEET), (_TELEMETRY_STUDY, TELEMETRY_SHEET))


def parse_subject_table(shape: SheetShape, rows: Sequence[Sequence[object]]) -> dict[tuple[int, ...], SubjectRow]:
    """Parse one subject sheet and key each row as its shape states.

    Returns:
        The facts of each row, by key.

    Raises:
        TimeNetConnectorError: If a row is short, a cell does not decode, or a key repeats.
    """
    named = [shape.subject, shape.age, shape.sex]
    for columns_of_night in shape.nights:
        named += [columns_of_night.night, columns_of_night.lights_off]

    columns = max(named) + 1
    table: dict[tuple[int, ...], SubjectRow] = {}
    for row_number, row in enumerate(rows[shape.header_rows :], start=shape.header_rows + 1):
        if len(row) < columns:
            raise TimeNetConnectorError(
                f"{shape.table_name} row {row_number}: holds {len(row)} columns and needs {columns}"
            )

        subject = decode_whole_number(row[shape.subject], "subject", shape.table_name, row_number)
        nights = tuple(
            _read_night(columns_of_night, row, shape.table_name, row_number) for columns_of_night in shape.nights
        )
        key = build_key(shape, subject, nights[0].night)
        if key in table:
            raise TimeNetConnectorError(
                f"{shape.table_name} row {row_number}: {_name_key(shape, key)} appears twice, and one key names one row"
            )

        table[key] = SubjectRow(
            subject=subject,
            age=decode_whole_number(row[shape.age], "age", shape.table_name, row_number),
            sex=_decode_sex(row[shape.sex], shape.sex_codes, shape.table_name, row_number),
            nights=nights,
        )

    return table


def _read_night(columns: NightColumns, row: Sequence[object], workbook: str, row_number: int) -> SubjectNight:
    field = "night" if columns.condition is None else f"{columns.condition} night"
    return SubjectNight(
        night=decode_whole_number(row[columns.night], field, workbook, row_number),
        condition=columns.condition,
        lights_off=decode_day_fraction_as_time(row[columns.lights_off], workbook, row_number),
    )


def build_key(shape: SheetShape, subject: int, night: int) -> tuple[int, ...]:
    """Build the key of one row or one recording.

    Returns:
        ``(subject, night)`` for a sheet keyed by night, else ``(subject,)``.
    """
    return (subject, night) if shape.keyed_by_night else (subject,)


def _name_key(shape: SheetShape, key: tuple[int, ...]) -> str:
    return f"subject {key[0]} night {key[1]}" if shape.keyed_by_night else f"subject {key[0]}"


def find_subject_row(
    table: Mapping[tuple[int, ...], SubjectRow], shape: SheetShape, subject: int, night: int, recording_id: str
) -> SubjectRow:
    """Find the row that describes one recording.

    Returns:
        The row of that recording.

    Raises:
        TimeNetConnectorError: If the table holds no row for that recording.
    """
    key = build_key(shape, subject, night)
    row = table.get(key)
    if row is None:
        raise TimeNetConnectorError(f"{recording_id}: {shape.table_name} holds no row for {_name_key(shape, key)}")

    return row


def find_night(row: SubjectRow, night: int, recording_id: str) -> SubjectNight:
    """Find the night of a row that a recording's night number names.

    Returns:
        That night, with its condition and lights-off time.

    Raises:
        TimeNetConnectorError: If the night number matches no night of the row, or more than one.
    """
    matches = [one for one in row.nights if one.night == night]
    if len(matches) != 1:
        raise TimeNetConnectorError(
            f"{recording_id}: night {night} matches {len(matches)} of the {len(row.nights)} night columns "
            f"that the subject table gives for subject {row.subject}, and it must match one"
        )

    return matches[0]


def _decode_sex(cell: object, codes: Mapping[int, Sex], workbook: str, row_number: int) -> Sex:
    code = decode_whole_number(cell, "sex", workbook, row_number)
    sex = codes.get(code)
    if sex is None:
        raise TimeNetConnectorError(
            f"{workbook} row {row_number}: the sex column holds the code {code}, which this sheet does not define"
        )

    return sex


@dataclass(frozen=True)
class SleepEdfxSource:
    """The fetched release: the directory and the subject table of each study."""

    studies: tuple[tuple[str, Path], ...]  # (study, study_dir) per study
    subject_tables: tuple[tuple[str, Path], ...]  # (study, table path) per study


@dataclass(frozen=True)
class SleepEdfxRecording:
    """One recording: its signal file, its scoring, and what its name states."""

    recording_id: str
    study: str
    subject_id: str
    subject_number: int  # the number that identifies this person inside its own study
    night: int  # 1 or 2, which of the subject's recordings this is
    psg_path: Path
    hypnogram_path: Path


@dataclass(frozen=True)
class RecordingFiles:
    """What the two files of one recording state."""

    header: reader.EdfHeader
    series: tuple[Signal, ...]
    entries: tuple[reader.EdfAnnotation, ...]
    signal_end: int


def read_subject_tables(source: SleepEdfxSource) -> dict[str, Sequence[Sequence[object]]]:
    """Read the rows of each study's subject table.

    Returns:
        The rows of each table, by study.
    """
    return {study: excel.read_table_rows(path) for study, path in source.subject_tables}


def read_recording(recording: SleepEdfxRecording) -> RecordingFiles:
    """Open the signal file of a recording and read its scoring.

    Returns:
        The header, the lazy signals, the scoring entries, and where the signals stop.
    """
    file = reader.open_edf(recording.psg_path)
    series = timeseries.build(file, SPECS, loader=reader.build_signal_loader)
    entries = reader.read_annotations(reader.open_edf(recording.hypnogram_path))
    return RecordingFiles(
        header=file.header,
        series=series,
        entries=entries,
        signal_end=reader.compute_signal_end_microseconds(file.header),
    )


def find_hypnogram(psg_path: Path) -> Path:
    """Find the one scoring of a PSG recording.

    Returns:
        The ``*-Hypnogram.edf`` file of the same night.

    Raises:
        TimeNetDownloadError: If the directory holds no scoring for this recording, or more than one.
    """
    # A hypnogram name ends with the initial of the technician who scored it, so match the prefix.
    prefix = psg_path.name[:7]
    matches = sorted(psg_path.parent.glob(f"{prefix}?-Hypnogram.edf"))
    if len(matches) != 1:
        found = ", ".join(match.name for match in matches) or "none"
        raise TimeNetDownloadError(f"expected one scoring for {psg_path.name}, found {found}")

    return matches[0]


def parse_subject_id(study: str, recording_id: str, psg_path: Path) -> str:
    """Give the id of the person a recording belongs to, such as ``sleep-cassette-93``.

    Returns:
        The study and the two subject digits, joined by a hyphen.
    """
    subject, _ = parse_recording_number(study, recording_id, psg_path)
    return f"{study}-{subject:02d}"


def parse_recording_number(study: str, recording_id: str, psg_path: Path) -> tuple[int, int]:
    """Give the subject number and the night number that a recording name states.

    Returns:
        The subject number and the night number.

    Raises:
        TimeNetConnectorError: If the study is unknown, the name does not fit the release's shape, or
            the night is not 1 or 2.
    """
    code = _STUDY_CODES.get(study)
    if code is None:
        raise TimeNetConnectorError(f"{psg_path}: lies in {study!r}, which is not a study of this release")

    match = _RECORDING_ID_SHAPE.fullmatch(recording_id)
    if match is None or match["code"] != code:
        raise TimeNetConnectorError(
            f"{psg_path.name}: a {study} recording is named {code}<2 subject digits><night><recorder>0, "
            f"as in {code}001E0, thus {recording_id!r} names no subject"
        )

    night = int(match["night"])
    if night not in _NIGHTS:
        raise TimeNetConnectorError(f"{psg_path.name}: states night {night}, and a subject slept night 1 or night 2")

    return int(match["subject"]), night


def iter_recordings(source: SleepEdfxSource) -> Iterator[SleepEdfxRecording]:
    """Walk the release and give one recording for each PSG file.

    Yields:
        Each recording, by study and then by recording id.
    """
    for study, study_dir in source.studies:
        for psg_path in sorted(study_dir.glob("*-PSG.edf")):
            # The filename is the only identifier. The EDF header's patient field is anonymous.
            recording_id = psg_path.stem.removesuffix("-PSG")
            subject_number, night = parse_recording_number(study, recording_id, psg_path)
            yield SleepEdfxRecording(
                recording_id=recording_id,
                study=study,
                subject_id=parse_subject_id(study, recording_id, psg_path),
                subject_number=subject_number,
                night=night,
                psg_path=psg_path,
                hypnogram_path=find_hypnogram(psg_path),
            )
