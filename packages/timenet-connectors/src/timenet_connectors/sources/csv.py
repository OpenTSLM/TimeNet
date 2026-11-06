"""Stream typed CSV metadata rows with physical source locations."""

from collections.abc import Generator
import csv
from pathlib import Path

from pydantic import ValidationError

from timenet.errors import TimeFValidationError, TimeNetConnectorError
from timenet_connectors.sources.rows import RowT, row_error_message


def _validate_header(path: Path, header: list[str], model: type[RowT]) -> None:
    """Check required, extra, and duplicate columns before reading any data.

    Raises:
        TimeNetConnectorError: If columns are missing, unexpected, or repeated.
    """
    columns = {field.alias or name: field for name, field in model.model_fields.items()}
    missing = {name for name, field in columns.items() if field.is_required()} - set(header)
    extra = set(header) - columns.keys()
    if not header or len(set(header)) != len(header) or missing or extra:
        raise TimeNetConnectorError(f"{path}: expected the columns {list(columns)}, got {header}")


def _parse_row(header: list[str], cells: list[str], model: type[RowT], location: str) -> RowT:
    """Validate row width and typed cell values with their physical source location.

    Returns:
        The typed row.

    Raises:
        TimeNetConnectorError: If the row is blank, ragged, or fails model validation.
    """
    if not cells:
        raise TimeNetConnectorError(f"{location}: blank row")
    if len(cells) != len(header):
        raise TimeNetConnectorError(f"{location}: expected {len(header)} fields, got {len(cells)}")
    try:
        return model.model_validate_strings(dict(zip(header, cells, strict=True)), strict=True)
    except ValidationError as exc:
        raise TimeNetConnectorError(row_error_message(exc, location)) from exc


def _check_unique(row: RowT, field: str, first_lines: dict[object, int], path: Path, line: int) -> None:
    """Remember a validated value's first line, or report both occurrences.

    Raises:
        TimeFValidationError: If the selected field's value is unhashable.
        TimeNetConnectorError: If the value has already appeared.
    """
    value = getattr(row, field)
    try:
        previous = first_lines.setdefault(value, line)
    except TypeError as exc:
        raise TimeFValidationError(f"{field}: uniqueness requires hashable field values") from exc
    if previous != line:
        raise TimeNetConnectorError(f"{path}:{line}: {field}: duplicate {value!r}; first listed on line {previous}")


def iter_csv_rows(
    path: Path, model: type[RowT], *, unique: str | None = None, skip_repeated_header: bool = False
) -> Generator[tuple[int, RowT]]:
    """Yield physical starting lines and typed rows, optionally enforcing a unique field.

    Models declare columns, aliases, constraints, and defaults. CSV cells use Pydantic's explicit
    string-input validation; text is not stripped and blanks are not implicitly converted to null.
    Missing optional columns use model defaults. Ragged rows and duplicate headers are rejected.
    The reader retains only the uniqueness index, not the rows, and closes its file when exhausted
    or closed. Uniqueness is checked on validated values and names a model field, not its CSV alias.

    Args:
        path: The CSV file.
        model: The connector's row model.
        unique: A scalar model field whose values must not repeat.
        skip_repeated_header: Skip exact repeats of the header in releases that embed them.

    Yields:
        Each physical starting line and its validated row.

    Raises:
        TimeFValidationError: If ``unique`` is not a model field or has unhashable values.
        TimeNetConnectorError: If the file, its structure, or a row is invalid.
    """
    if unique is not None and unique not in model.model_fields:
        raise TimeFValidationError(f"{unique!r} is not a field of {model.__name__}")
    first_lines: dict[object, int] = {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader, [])
            _validate_header(path, header, model)
            while True:
                line = reader.line_num + 1
                cells = next(reader, None)
                if cells is None:
                    break
                if skip_repeated_header and cells == header:
                    continue
                row = _parse_row(header, cells, model, f"{path}:{line}")
                if unique is not None:
                    _check_unique(row, unique, first_lines, path, line)
                yield line, row
    except csv.Error as exc:
        raise TimeNetConnectorError(f"{path}:{reader.line_num}: {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise TimeNetConnectorError(f"{path}: {exc}") from exc
