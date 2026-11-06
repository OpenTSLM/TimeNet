"""Stream typed CSV metadata rows with physical source locations."""

from collections.abc import Generator
import csv
from pathlib import Path

from pydantic import ValidationError

from timenet.errors import TimeFValidationError, TimeNetConnectorError
from timenet_connectors.sources.rows import RowT, row_error_message


def iter_csv_rows(path: Path, model: type[RowT], *, unique: str | None = None) -> Generator[tuple[int, RowT]]:
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

    Yields:
        Each physical starting line and its validated row.

    Raises:
        TimeFValidationError: If ``unique`` is not a model field or has unhashable values.
        TimeNetConnectorError: If the file, its structure, or a row is invalid.
    """
    if unique is not None and unique not in model.model_fields:
        raise TimeFValidationError(f"{unique!r} is not a field of {model.__name__}")
    columns = {field.alias or name: field for name, field in model.model_fields.items()}
    first_lines: dict[object, int] = {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader, [])
            missing = {name for name, field in columns.items() if field.is_required()} - set(header)
            extra = set(header) - columns.keys()
            if not header or len(set(header)) != len(header) or missing or extra:
                raise TimeNetConnectorError(f"{path}: expected the columns {list(columns)}, got {header}")
            while True:
                line = reader.line_num + 1
                cells = next(reader, None)
                if cells is None:
                    break
                if not cells:
                    raise TimeNetConnectorError(f"{path}:{line}: blank row")
                if len(cells) != len(header):
                    raise TimeNetConnectorError(f"{path}:{line}: expected {len(header)} fields, got {len(cells)}")
                try:
                    row = model.model_validate_strings(dict(zip(header, cells, strict=True)), strict=True)
                except ValidationError as exc:
                    raise TimeNetConnectorError(row_error_message(exc, f"{path}:{line}")) from exc
                if unique is not None:
                    value = getattr(row, unique)
                    try:
                        previous = first_lines.setdefault(value, line)
                    except TypeError as exc:
                        raise TimeFValidationError(f"{unique}: uniqueness requires hashable field values") from exc
                    if previous != line:
                        raise TimeNetConnectorError(
                            f"{path}:{line}: {unique}: duplicate {value!r}; first listed on line {previous}"
                        )
                yield line, row
    except csv.Error as exc:
        raise TimeNetConnectorError(f"{path}:{reader.line_num}: {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise TimeNetConnectorError(f"{path}: {exc}") from exc
