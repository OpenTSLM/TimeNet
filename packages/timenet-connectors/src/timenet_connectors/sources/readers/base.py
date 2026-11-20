"""Common contract for validated source rows."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from timenet.errors import TimeNetConnectorError


class DataRow(BaseModel):
    """A connector-owned schema for a source row."""

    model_config = ConfigDict(extra="forbid")


RowT = TypeVar("RowT", bound=DataRow)


class DataReader(ABC, Generic[RowT]):
    """Read and validate rows from one local source file.

    The row schema determines validation strictness. CSV parses its string cells.
    """

    def __init__(self, path: Path, row_type: type[RowT]) -> None:
        self.path = path
        self.row_type = row_type

    def _validate(self, value: object, context: str, *, strings: bool = False) -> RowT:
        try:
            if strings:
                return self.row_type.model_validate_strings(value)
            return self.row_type.model_validate(value)
        except ValidationError as exc:
            raise TimeNetConnectorError(f"{self.path}: {context}: {exc}") from exc

    @abstractmethod
    def iter_rows(self) -> Generator[RowT, None, None]:
        """Yield validated rows in source order.

        Close the iterator explicitly if stopping early to release its file.

        Yields:
            Validated source rows.
        """
        ...
