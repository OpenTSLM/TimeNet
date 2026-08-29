"""Turn the rows of a subject table into the facts it states.

:func:`~timenet_connectors.datasets.physionet.sleep_edfx.reader.read_table_rows` opens the
workbook and gives the rows. Nothing here takes a path, so a test checks the decoding, the
reshape and the join with rows it wrote itself, and no ``.xls`` binary.

The release ships one workbook for each study, and the two sheets are two shapes. One parser
takes both. A per-sheet description states the difference: the header rows to skip, the columns
to take by position, the key of a row, and the map of its sex column. A new sheet is a new
description and not a new parser, and no function here is written for one study.

The cassette sheet holds a row for each recording, keyed by the subject number and the night.
The telemetry sheet holds a row for each subject, with both of that subject's nights side by
side, keyed by the subject number alone. Both give the same row, which carries the nights its
own sheet states.

The two sheets code sex with opposite meanings. The cassette sheet heads its column
``sex (F=1)``. The telemetry sheet heads its own ``M1/F2``. Each description carries the map of
its own sheet, and both decode into the two letters ``F`` and ``M``. A raw code never leaves
this module, because one code means the opposite thing in the other study.

Both sheets state a lights-off time as a number. Excel stores a bare time as the fraction of a
day past midnight, so a value below 0.5 is a time after midnight and not an error.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import time

from timenet.errors import TimeFFormatError


# The two workbooks of the release, named here and by the description of each sheet.
CASSETTE_TABLE_NAME = "SC-subjects.xls"
TELEMETRY_TABLE_NAME = "ST-subjects.xls"

# The two arms of the telemetry experiment. The sheet names them in a merged header cell over
# the pair of columns that holds each one.
PLACEBO = "placebo"
TEMAZEPAM = "temazepam"

# The parts a key holds. A description names the parts of its own key, and one function builds
# that key from a row and from the numbers a recording name states.
SUBJECT = "subject"
NIGHT = "night"

_SECONDS_PER_DAY = 24 * 60 * 60


@dataclass(frozen=True)
class NightColumns:
    """Where one night sits in a row, and the condition that pair of columns carries.

    A sheet states its drug condition in the position of these columns and nowhere else. No
    filename encodes it, no header field names it, and no cell of the sheet spells it out. A
    sheet that states no condition gives ``None``.
    """

    night: int
    lights_off: int
    condition: str | None


@dataclass(frozen=True)
class SheetShape:
    """How to parse one subject sheet.

    The parser takes this and the rows of a sheet. The difference between the two sheets is data
    here, so it never becomes a branch on the study.
    """

    table_name: str
    header_rows: int
    subject: int
    age: int
    sex: int
    sex_codes: Mapping[int, str]
    key: tuple[str, ...]
    nights: tuple[NightColumns, ...]


@dataclass(frozen=True)
class SubjectNight:
    """One night of a subject, with the condition of the columns it was read from."""

    night: int
    condition: str | None
    lights_off: time


@dataclass(frozen=True)
class SubjectRow:
    """What one row of a subject table states about a subject and their nights.

    A sheet whose row is one recording gives one night here. A sheet whose row is one subject
    gives both nights of that subject.
    """

    subject: int
    age: int
    sex: str
    nights: tuple[SubjectNight, ...]


# The cassette sheet: one header row, then one row for each recording. The pair its recording id
# encodes is the key, because a row is one recording.
CASSETTE_SHEET = SheetShape(
    table_name=CASSETTE_TABLE_NAME,
    header_rows=1,
    subject=0,
    age=2,
    sex=3,
    sex_codes={1: "F", 2: "M"},
    key=(SUBJECT, NIGHT),
    nights=(NightColumns(night=1, lights_off=4, condition=None),),
)

# The telemetry sheet: two header rows, then one row for each subject. The group labels of the
# first header row are merged cells, and every column but the first of each pair reads back
# empty. A parser cannot build its column names from the header, so it takes fixed positions.
TELEMETRY_SHEET = SheetShape(
    table_name=TELEMETRY_TABLE_NAME,
    header_rows=2,
    subject=0,
    age=1,
    sex=2,
    sex_codes={1: "M", 2: "F"},
    key=(SUBJECT,),
    nights=(
        NightColumns(night=3, lights_off=4, condition=PLACEBO),
        NightColumns(night=5, lights_off=6, condition=TEMAZEPAM),
    ),
)


def parse_subject_table(shape: SheetShape, rows: Sequence[Sequence[object]]) -> dict[tuple[int, ...], SubjectRow]:
    """Parse one subject sheet, and key each row as its description states.

    The subject number comes from its own column and never from the position of a row. The
    numbers of a sheet can skip values, so the column is an id and the position is not.

    Args:
        shape: The description of that sheet.
        rows: The rows of the sheet, from :func:`read_table_rows`, with its header rows first.

    Returns:
        The facts of each row, keyed as the description states.

    Raises:
        TimeFFormatError: If a row is short, if a cell does not decode, or if one key appears
            twice.
    """
    columns = _count_columns(shape)
    table: dict[tuple[int, ...], SubjectRow] = {}
    for row_number, row in enumerate(rows[shape.header_rows :], start=shape.header_rows + 1):
        if len(row) < columns:
            raise TimeFFormatError(f"{shape.table_name} row {row_number}: holds {len(row)} columns and needs {columns}")

        subject = _decode_whole_number(row[shape.subject], SUBJECT, shape.table_name, row_number)
        nights = tuple(
            _parse_night(columns_of_night, row, shape.table_name, row_number) for columns_of_night in shape.nights
        )
        # A sheet whose row is one recording holds that recording's night, and its key takes
        # it. A sheet whose key is the subject alone drops it.
        key = build_key(shape, subject, nights[0].night)
        if key in table:
            raise TimeFFormatError(
                f"{shape.table_name} row {row_number}: {_name_key(shape, key)} appears twice, and one key names one row"
            )

        table[key] = SubjectRow(
            subject=subject,
            age=_decode_whole_number(row[shape.age], "age", shape.table_name, row_number),
            sex=_decode_sex(row[shape.sex], shape.sex_codes, shape.table_name, row_number),
            nights=nights,
        )

    return table


def build_key(shape: SheetShape, subject: int, night: int) -> tuple[int, ...]:
    """Build the key of one row or one recording, from the parts the description names.

    Args:
        shape: The description of the sheet.
        subject: The subject number.
        night: The night number. A sheet whose key names no night drops it.

    Returns:
        The key, with one number for each part the description names.
    """
    parts = {SUBJECT: subject, NIGHT: night}
    return tuple(parts[part] for part in shape.key)


def _name_key(shape: SheetShape, key: tuple[int, ...]) -> str:
    """Write a key out for an error message, with each number under the part it belongs to.

    Args:
        shape: The description of the sheet.
        key: The key, from :func:`build_key`.

    Returns:
        The key as text, as in ``subject 0 night 1``.
    """
    return " ".join(f"{part} {value}" for part, value in zip(shape.key, key, strict=True))


def find_subject_row(
    table: Mapping[tuple[int, ...], SubjectRow], shape: SheetShape, subject: int, night: int, recording_id: str
) -> SubjectRow:
    """Find the row that describes one recording.

    Args:
        table: The table of that sheet, from :func:`parse_subject_table`.
        shape: The description of that sheet, which states the key.
        subject: The subject number the recording name states.
        night: The night number the recording name states.
        recording_id: The recording, for the error message.

    Returns:
        The row of that recording.

    Raises:
        TimeFFormatError: If the table holds no row for that key. The join ties a recording
            to a person, and a recording with no row states no age, sex or lights-off time.
    """
    key = build_key(shape, subject, night)
    row = table.get(key)
    if row is None:
        raise TimeFFormatError(f"{recording_id}: {shape.table_name} holds no row for {_name_key(shape, key)}")

    return row


def find_night(row: SubjectRow, night: int, recording_id: str) -> SubjectNight:
    """Find the night of a row that the night number of a recording names.

    The night that matches decides the drug condition, because the column it was read from is
    the only place the release states it.

    Args:
        row: The row of that subject, from :func:`find_subject_row`.
        night: The night number the recording name states.
        recording_id: The recording, for the error message.

    Returns:
        The night of that recording, with its condition and its lights-off time.

    Raises:
        TimeFFormatError: If the night number matches no night of the row, or more than one.
    """
    matches = [one for one in row.nights if one.night == night]
    if len(matches) != 1:
        raise TimeFFormatError(
            f"{recording_id}: night {night} matches {len(matches)} of the {len(row.nights)} night columns "
            f"that the subject table gives for subject {row.subject}, and it must match one"
        )

    return matches[0]


def _parse_night(columns: NightColumns, row: Sequence[object], workbook: str, row_number: int) -> SubjectNight:
    """Parse one night of a row, from the columns the description names.

    Args:
        columns: Where that night sits, and the condition its columns carry.
        row: The row of the sheet.
        workbook: The name of the workbook, for the error message.
        row_number: The row of the sheet, counted from 1, for the error message.

    Returns:
        That night, with its condition and its lights-off time.

    Raises:
        TimeFFormatError: If the night number or the lights-off time does not decode.
    """  # noqa: DOC502 (raised by the decoders, not directly here)
    field = NIGHT if columns.condition is None else f"{columns.condition} night"
    return SubjectNight(
        night=_decode_whole_number(row[columns.night], field, workbook, row_number),
        condition=columns.condition,
        lights_off=_decode_lights_off(row[columns.lights_off], workbook, row_number),
    )


def _count_columns(shape: SheetShape) -> int:
    """Give how many columns a row of one sheet must hold.

    Args:
        shape: The description of the sheet.

    Returns:
        One more than the highest column the description names.
    """
    named = [shape.subject, shape.age, shape.sex]
    for columns in shape.nights:
        named += [columns.night, columns.lights_off]

    return max(named) + 1


def _decode_whole_number(cell: object, field: str, workbook: str, row_number: int) -> int:
    """Decode the whole number that a cell holds.

    Args:
        cell: The value of the cell, as the workbook states it.
        field: The name of the column, for the error message.
        workbook: The name of the workbook, for the error message.
        row_number: The row of the sheet, counted from 1, for the error message.

    Returns:
        The value as a whole number.

    Raises:
        TimeFFormatError: If the cell holds no whole number.
    """
    if isinstance(cell, bool) or not isinstance(cell, int | float) or not float(cell).is_integer():
        raise TimeFFormatError(f"{workbook} row {row_number}: {field} holds {cell!r}, which is not a whole number")

    return int(cell)


def _decode_sex(cell: object, codes: Mapping[int, str], workbook: str, row_number: int) -> str:
    """Decode a sex code with the map of the sheet it was read from.

    Args:
        cell: The value of the cell, as the workbook states it.
        codes: The map of that sheet, from a code to a letter.
        workbook: The name of the workbook, for the error message.
        row_number: The row of the sheet, counted from 1, for the error message.

    Returns:
        ``F`` or ``M``.

    Raises:
        TimeFFormatError: If the sheet defines no such code.
    """
    code = _decode_whole_number(cell, "sex", workbook, row_number)
    sex = codes.get(code)
    if sex is None:
        raise TimeFFormatError(
            f"{workbook} row {row_number}: the sex column holds the code {code}, which this sheet does not define"
        )

    return sex


def _decode_lights_off(cell: object, workbook: str, row_number: int) -> time:
    """Convert an Excel day fraction into a time of day.

    Excel stores a bare time as the fraction of a day past midnight. A fraction below 0.5 is a
    time after midnight, which most of the lights-off times of this release are.

    A string reaches this function only by accident. A sheet whose column has become text is
    a release this connector does not know.

    Args:
        cell: The value of the cell, as the workbook states it.
        workbook: The name of the workbook, for the error message.
        row_number: The row of the sheet, counted from 1, for the error message.

    Returns:
        The clock time that the fraction states, to the second.

    Raises:
        TimeFFormatError: If the cell holds no number, or a number outside 0 up to 1.
    """
    if isinstance(cell, bool) or not isinstance(cell, int | float):
        raise TimeFFormatError(
            f"{workbook} row {row_number}: the lights-off column holds {cell!r}, which is not a number. "
            "Excel states a bare time as the fraction of a day past midnight"
        )

    seconds = round(cell * _SECONDS_PER_DAY)
    if not 0 <= seconds < _SECONDS_PER_DAY:
        raise TimeFFormatError(
            f"{workbook} row {row_number}: the lights-off column holds {cell!r}, which is not a day "
            "fraction from 0 up to 1"
        )

    hours, rest = divmod(seconds, 60 * 60)
    minutes, seconds_of_minute = divmod(rest, 60)
    return time(hours, minutes, seconds_of_minute)
