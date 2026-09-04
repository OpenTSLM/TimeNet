"""Read a legacy ``.xls`` workbook, and decode the cell values it gives back.

A connector whose source ships a spreadsheet opens it with :func:`read_table_rows` and gets the
rows of the first sheet. The cells come back as the workbook states them, so nothing here knows
what a column means. The connector states that.

The decoders cover what Excel itself states about a cell and nothing more. A whole number is a
whole number in every workbook, and Excel writes a bare time as the fraction of a day past
midnight in every workbook. A code whose meaning changes from one sheet to the next belongs to
the connector that reads that sheet.

Each decoder takes the workbook name and the row number so that its error names the cell a
reader has to go and look at.
"""

from datetime import time
from pathlib import Path

import xlrd

from timenet.errors import TimeFFormatError


SECONDS_PER_DAY = 24 * 60 * 60


def read_table_rows(path: Path) -> list[tuple[object, ...]]:
    """Open a workbook and give every row of its first sheet, header rows included.

    This gives the cells as the workbook states them and decodes nothing.

    A workbook can carry more than one sheet. This takes the first and searches for no sheet by
    name, because the first sheet is where a source that ships one table puts it.

    Args:
        path: The ``.xls`` workbook.

    Returns:
        One tuple of cell values for each row of the first sheet, in sheet order.

    Raises:
        TimeFFormatError: If the file does not open as a workbook, or holds no sheet.
    """
    try:
        # These are legacy BIFF8 workbooks that Excel wrote. openpyxl reads only the ZIP-based
        # .xlsx format and cannot open them at all.
        book = xlrd.open_workbook(path)
    # xlrd has no single exception type for a bad file, so catch broadly and re-raise.
    except Exception as exc:
        raise TimeFFormatError(f"{path}: cannot read this file as a workbook") from exc

    if book.nsheets == 0:
        raise TimeFFormatError(f"{path}: holds no data sheet")

    sheet = book.sheet_by_index(0)
    return [tuple(sheet.row_values(index)) for index in range(sheet.nrows)]


def decode_whole_number(cell: object, field: str, workbook: str, row_number: int) -> int:
    """Decode the whole number that a cell holds.

    Excel writes every number as a float, so a whole number reaches this as ``44.0``. A cell that
    holds a fraction, a bool or text states no whole number and raises.

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


def decode_day_fraction_as_time(cell: object, workbook: str, row_number: int) -> time:
    """Convert an Excel day fraction into a time of day.

    Excel stores a bare time as the fraction of a day past midnight, so 0.5 is 12:00:00 and a
    fraction below 0.5 is a time after midnight and not an error.

    A string reaches this function only by accident. A column that has become text is a workbook
    the caller does not know.

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
            f"{workbook} row {row_number}: the cell holds {cell!r}, which is not a number. "
            "Excel states a bare time as the fraction of a day past midnight"
        )

    seconds = round(cell * SECONDS_PER_DAY)
    if not 0 <= seconds < SECONDS_PER_DAY:
        raise TimeFFormatError(
            f"{workbook} row {row_number}: the cell holds {cell!r}, which is not a day fraction from 0 up to 1"
        )

    hours, rest = divmod(seconds, 60 * 60)
    minutes, seconds_of_minute = divmod(rest, 60)
    return time(hours, minutes, seconds_of_minute)
