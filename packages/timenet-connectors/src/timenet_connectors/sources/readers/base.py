"""Validated source rows and the common contract for reading them with provenance."""

from abc import ABC, abstractmethod
from collections.abc import Generator, Iterator, Mapping, Sequence
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from pathlib import Path
from typing import Annotated, Generic, Protocol, Self, TypeVar

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictStr,
    StringConstraints,
    TypeAdapter,
    ValidationError,
)

from timenet.errors import TimeNetConnectorError


NonBlankText = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
"""Nonblank source text, preserved without stripping whitespace."""

_EnumT = TypeVar("_EnumT", bound=Enum)
PublishedEnum = Annotated[_EnumT, Field(strict=False), BeforeValidator(TypeAdapter(StrictStr).validate_python)]
"""Parse a published string into an ordinary enum, rejecting other decoded input types."""


class DataRow(BaseModel):
    """A connector-owned source schema with strict decoded values and immutable fields.

    Subclasses own aliases, defaults, field constraints, and explicit missing-value rules.
    CSV strings are parsed separately from already-decoded JSON or Parquet values.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @classmethod
    def from_decoded(cls, values: object, *, context: str) -> Self:
        """Validate decoded values without coercion, reporting their source context.

        Returns:
            The validated row.

        Raises:
            TimeNetConnectorError: If the values violate the declared schema.
        """
        try:
            return cls.model_validate(values)
        except ValidationError as exc:
            raise TimeNetConnectorError(cls._error_message(exc, context)) from exc

    @classmethod
    def from_strings(cls, cells: Mapping[str, str], *, context: str) -> Self:
        """Parse string cells without stripping text or replacing blanks with null.

        Returns:
            The validated row.

        Raises:
            TimeNetConnectorError: If the cells violate the declared schema.
        """
        try:
            return cls.model_validate_strings(dict(cells), strict=True)
        except ValidationError as exc:
            raise TimeNetConnectorError(cls._error_message(exc, context)) from exc

    @staticmethod
    def _error_message(error: ValidationError, context: str) -> str:
        first = error.errors(include_url=False)[0]
        field = ".".join(str(part) for part in first["loc"]) or "row"
        return f"{context}: {field}: {first['msg']}"

    @classmethod
    def column_names(cls) -> tuple[str, ...]:
        """Return the declared source column names in field order.

        Returns:
            Field names or their single-string validation aliases.

        Raises:
            TimeNetConnectorError: If aliases are nested, alternative, or shared by multiple fields.
        """
        columns: list[str] = []
        for name, field in cls.model_fields.items():
            alias = field.validation_alias if field.validation_alias is not None else name
            if not isinstance(alias, str):
                raise TimeNetConnectorError(f"{cls.__name__}.{name}: table columns require a single string alias")
            if alias in columns:
                raise TimeNetConnectorError(f"{cls.__name__}: multiple fields use column {alias!r}")
            columns.append(alias)
        return tuple(columns)

    @classmethod
    def check_columns(cls, columns: Sequence[str], *, context: str) -> None:
        """Check table columns against field names or simple validation aliases.

        Raises:
            TimeNetConnectorError: If required columns are missing, unknown columns appear, or
                the model has unsupported or ambiguous aliases.
        """
        known = dict(zip(cls.column_names(), (field.is_required() for field in cls.model_fields.values()), strict=True))
        missing = [name for name, required in known.items() if required and name not in columns]
        unexpected = [name for name in columns if name not in known]
        if missing or unexpected:
            raise TimeNetConnectorError(f"{context}: missing columns {missing}; unexpected columns {unexpected}")


class SourceLocation(Protocol):
    """Format-specific provenance with a human-readable error context."""

    @property
    def context(self) -> str:
        """Describe the source position without losing its format-specific meaning."""
        ...


RowT = TypeVar("RowT", bound=DataRow)
LocationT = TypeVar("LocationT", bound=SourceLocation)


@dataclass(frozen=True)
class LocatedRow(Generic[RowT, LocationT]):
    """A validated row paired with its concrete source location."""

    row: RowT
    location: LocationT


@dataclass(frozen=True)
class RowSource(ABC, Generic[RowT, LocationT]):
    """A repeatable typed source owning configuration, resource lifetime and uniqueness.

    Format implementations open their resources through the supplied ExitStack and return a
    generator of located rows. The common read context closes both generators and resources
    on exhaustion, early exit, or any failure. A fresh uniqueness index belongs to each read.
    """

    path: Path
    model: type[RowT]
    unique: str | None = dataclass_field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.unique is not None:
            UniqueValues.check_field(self.model, self.unique)

    @contextmanager
    def open_rows(self) -> Generator[Iterator[LocatedRow[RowT, LocationT]], None, None]:
        """Open an independent streaming read with deterministic cleanup.

        Consume rows inside the context. Caller exceptions propagate unchanged; format readers
        translate only their source errors into TimeNetConnectorError with provenance.

        Yields:
            Validated rows with concrete source locations.
        """
        with ExitStack() as resources:
            rows = resources.enter_context(closing(self._open_rows(resources)))
            yield resources.enter_context(closing(self._check_uniqueness(rows)))

    def _check_uniqueness(
        self, rows: Iterator[LocatedRow[RowT, LocationT]]
    ) -> Generator[LocatedRow[RowT, LocationT], None, None]:
        unique = None if self.unique is None else UniqueValues[RowT, LocationT](self.model, self.unique)
        for entry in rows:
            if unique is not None:
                unique.add(entry)
            yield entry

    @abstractmethod
    def _open_rows(self, resources: ExitStack) -> Generator[LocatedRow[RowT, LocationT], None, None]:
        """Open format resources and return its row generator.

        Returns:
            Located rows whose resources are owned by the supplied stack.
        """
        ...


class UniqueValues(Generic[RowT, LocationT]):
    """Enforce one field's uniqueness within a read, retaining its first source location.

    Memory grows with the number of distinct values. This is a collection constraint, not a
    row invariant; create a fresh instance for each read.
    """

    def __init__(self, model: type[RowT], field: str) -> None:
        self.check_field(model, field)
        self.field = field
        self._first: dict[object, LocationT] = {}

    @staticmethod
    def check_field(model: type[RowT], field: str) -> None:
        """Check that uniqueness names a model field rather than a source alias.

        Raises:
            TimeNetConnectorError: If the selected field is not declared by the model.
        """
        if field not in model.model_fields:
            raise TimeNetConnectorError(f"{field!r} is not a field of {model.__name__}")

    def add(self, entry: LocatedRow[RowT, LocationT]) -> None:
        """Record a value or reject its repetition, naming both source locations.

        Raises:
            TimeNetConnectorError: If the field value is unhashable or has already appeared.
        """
        value = getattr(entry.row, self.field)
        try:
            if value not in self._first:
                self._first[value] = entry.location
                return
            first = self._first[value]
        except TypeError as exc:
            raise TimeNetConnectorError(
                f"{entry.location.context}: {self.field}: uniqueness requires hashable values"
            ) from exc
        raise TimeNetConnectorError(
            f"{entry.location.context}: {self.field}: duplicate {value!r}; first at {first.context}"
        )
