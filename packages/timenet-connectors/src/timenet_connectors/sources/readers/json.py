"""Read JSON and JSON Lines source rows."""

from collections.abc import Generator
import json
from typing import Any

from timenet.errors import TimeNetConnectorError
from timenet_connectors.sources.readers.base import DataReader, RowT


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


class JsonArrayDataReader(DataReader[RowT]):
    """Read rows from a UTF-8 JSON array, loaded into memory."""

    def iter_rows(self) -> Generator[RowT, None, None]:
        """Yield validated elements of the array.

        Yields:
            Validated source rows.

        Raises:
            TimeNetConnectorError: If the file or an element is invalid.
        """
        try:
            with self.path.open(encoding="utf-8") as handle:
                values = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
        except (OSError, UnicodeError, ValueError, RecursionError) as exc:
            raise TimeNetConnectorError(f"{self.path}: {exc}") from exc
        if not isinstance(values, list):
            raise TimeNetConnectorError(f"{self.path}: expected a JSON array")
        for index, value in enumerate(values, start=1):
            yield self._validate(value, f"entry {index}")


class JsonlDataReader(DataReader[RowT]):
    """Read one JSON value per line of a UTF-8 file."""

    def iter_rows(self) -> Generator[RowT, None, None]:
        """Yield validated lines.

        Yields:
            Validated source rows.

        Raises:
            TimeNetConnectorError: If the file or a line is invalid.
        """
        try:
            with self.path.open(encoding="utf-8") as handle:
                for number, line in enumerate(handle, start=1):
                    try:
                        value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
                    except (ValueError, RecursionError) as exc:
                        raise TimeNetConnectorError(f"{self.path}: line {number}: {exc}") from exc
                    yield self._validate(value, f"line {number}")
        except (OSError, UnicodeError) as exc:
            raise TimeNetConnectorError(f"{self.path}: {exc}") from exc
