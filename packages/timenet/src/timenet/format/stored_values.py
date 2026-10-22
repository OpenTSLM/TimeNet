"""Read and write text, string lists, and timestamps stored in Parquet payloads."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum, unique
from functools import cached_property
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Literal, Self, TypeVar

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import Field, ValidationError, field_validator, model_validator

from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.checksums import CHECKSUM_PREFIX
from timenet.format.constants import DEFAULT_COMPRESSION, PAYLOAD_TEMPLATE, check_relative_path, part_path
from timenet.format.control_schema import EXTERNAL_VALUES
from timenet.format.table_writer import ModelTable
from timenet.storage.parquet import PayloadTableCache
from timenet.types._model import TimeFModel
from timenet.types.object_kind import ObjectKind
from timenet.types.text import TextValue


if TYPE_CHECKING:
    from timenet.registry.version import DatasetVersion


INLINE_LIMIT = 4096
"""Largest JSON-encoded size, in bytes, of text kept inline in the catalog.

Labels and short prompts stay queryable in SQL, while long text no longer inflates the catalog that
every read loads.
"""

_PACK_BYTES = 4 * 1024 * 1024
"""Encoded bytes buffered before the writer starts a new payload file.

This bounds writer memory and keeps payload files few enough to list in the manifest.
"""

_CACHE_FILES = 8
"""Payload files a reader keeps decoded, about 32 MiB of text at the packing target."""

_SCHEMA = pa.schema([("text", pa.string()), ("texts", pa.list_(pa.string())), ("offsets", pa.list_(pa.int64()))])


@unique
class PackedValueKind(StrEnum):
    """Represent the value shape stored at a packed payload location."""

    TEXT = "text"
    TEXT_LIST = "texts"
    OFFSETS = "offsets"


@unique
class PackedField(StrEnum):
    """Name the object fields that can store external values."""

    PROMPT = "prompt"
    RATIONALE = "rationale"
    TARGETS = "targets"
    VALUE = "value"
    OFFSETS = "offsets"


_FIELD_OWNERS = {
    PackedField.PROMPT: ObjectKind.TASK,
    PackedField.RATIONALE: ObjectKind.TASK,
    PackedField.TARGETS: ObjectKind.TASK,
    PackedField.VALUE: ObjectKind.ANNOTATION,
    PackedField.OFFSETS: ObjectKind.AXIS,
}


_FIELD_VALUE_KINDS = {
    PackedField.PROMPT: (PackedValueKind.TEXT,),
    PackedField.RATIONALE: (PackedValueKind.TEXT,),
    PackedField.TARGETS: (PackedValueKind.TEXT,),
    PackedField.VALUE: (PackedValueKind.TEXT, PackedValueKind.TEXT_LIST),
    PackedField.OFFSETS: (PackedValueKind.OFFSETS,),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class ExternalObjectFieldKey:
    """Identify an external field on an object, including its position in an ordered field."""

    object_kind: ObjectKind
    object_id: str
    field: PackedField
    position: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.field, PackedField):
            raise TimeFValidationError(f"external field must be a PackedField, got {self.field!r}")
        if _FIELD_OWNERS[self.field] != self.object_kind:
            raise TimeFValidationError(f"unsupported external field {self.object_kind.value}.{self.field.value}")
        if self.position < 0 or (self.field != PackedField.TARGETS and self.position != 0):
            raise TimeFValidationError(f"invalid position {self.position} for external field {self.field.value}")


class PackedValueLocator(TimeFModel):
    """Validate the catalog location and value kind of one external field."""

    object_kind: ObjectKind
    object_id: str = Field(strict=True, min_length=1)
    field: PackedField
    position: int = Field(strict=True, ge=0)
    value_path: str = Field(strict=True, min_length=1)
    row_index: int = Field(strict=True, ge=0)
    value_kind: PackedValueKind
    n_values: int = Field(strict=True, ge=0)
    value_digest: str | None = Field(default=None, pattern=rf"^{CHECKSUM_PREFIX}[0-9a-f]{{64}}$", strict=True)

    @property
    def key(self) -> ExternalObjectFieldKey:
        """Return the validated object field addressed by this locator."""
        return ExternalObjectFieldKey(
            object_kind=self.object_kind, object_id=self.object_id, field=self.field, position=self.position
        )

    @field_validator("value_path")
    @classmethod
    def _check_path(cls, value: str) -> str:
        return check_relative_path("external value_path", value)

    @model_validator(mode="after")
    def _check_field(self) -> Self:
        key = self.key
        if self.value_kind not in _FIELD_VALUE_KINDS[key.field]:
            raise TimeFValidationError(f"external field {key.field.value} cannot store {self.value_kind.value}")
        if (self.value_kind == PackedValueKind.OFFSETS) != (self.value_digest is not None):
            raise TimeFValidationError("external timestamp arrays require a digest; text values must not store one")
        return self


LOCATOR_TABLE = ModelTable[PackedValueLocator](EXTERNAL_VALUES)


@dataclass(frozen=True, slots=True)
class InlineValue:
    """Keep an original value in the catalog, including a legitimate absent value."""

    value: object


@dataclass(frozen=True, slots=True)
class _PackableValue:
    """Pair a supported payload value with its storage kind before deciding where it fits."""

    value: str | list[str] | pa.Array
    kind: PackedValueKind


def _classify_value(value: object) -> InlineValue | _PackableValue:
    """Resolve loaded handles and distinguish payload candidates from ordinary inline values.

    Returns:
        A typed payload candidate or an inline value, including an absent value.

    Raises:
        TimeFValidationError: If a string list or timestamp array has an unsupported element type.
    """
    if isinstance(value, TextValue):
        value = value.read()
    if isinstance(value, str):
        return _PackableValue(value, PackedValueKind.TEXT)
    if isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise TimeFValidationError("packed string lists must contain only strings")
        return _PackableValue(value, PackedValueKind.TEXT_LIST)
    if isinstance(value, pa.Array):
        if value.type != pa.int64() or value.null_count:
            raise TimeFValidationError("packed offsets must be non-null int64 timestamps")
        return _PackableValue(value, PackedValueKind.OFFSETS)
    return InlineValue(value)


@dataclass(frozen=True, slots=True)
class ExternalValue:
    """Identify a value written to a payload instead of the catalog."""

    locator: PackedValueLocator


def offsets_digest(offsets: np.ndarray | pa.Array) -> str:
    """Return the content ID of an irregular axis's timestamps.

    Writers compare a stored axis with in-memory timestamps by this digest instead of reading the
    stored array, so every caller hashes the same bytes: contiguous little-endian int64.

    Returns:
        The algorithm-qualified SHA-256 digest.
    """
    values = offsets.to_numpy(zero_copy_only=False) if isinstance(offsets, pa.Array) else offsets
    canonical = np.ascontiguousarray(values, dtype="<i8")
    return CHECKSUM_PREFIX + hashlib.sha256(memoryview(canonical)).hexdigest()


_Native_co = TypeVar("_Native_co", covariant=True)


@dataclass(frozen=True, slots=True, eq=False)
class ValueHandle(ABC, Generic[_Native_co]):
    """Locate one value in a payload file; constructing a handle performs no payload I/O."""

    store: "PackedValueReader"
    path: str
    row: int
    kind: PackedValueKind
    length: int

    @abstractmethod
    def read(self) -> _Native_co:
        """Return the stored value."""

    def __deepcopy__(self, memo: dict) -> Self:
        return self


@dataclass(frozen=True, slots=True, eq=False)
class TextHandle(ValueHandle[str | list[str]], TextValue[str | list[str]]):
    """Locate packed text or a string list, which compares by content like any text value."""

    @property
    def value_type(self) -> Literal["str", "list"]:
        """Return the native text type without opening a payload file."""
        return "str" if self.kind == PackedValueKind.TEXT else "list"

    def read(self) -> str | list[str]:
        """Materialize this external value.

        Returns:
            Native text or a string list.
        """
        return self.store.read(self)


@dataclass(frozen=True, eq=False)
class OffsetHandle(ValueHandle[pa.Array]):
    """Locate one irregular axis's timestamps and serve them as its Signals' offsets loader.

    One handle exists per axis, shared by the Signals on it. The first read validates the
    timestamps against the axis and keeps them, so later reads do no I/O. The digest identifies the
    stored array, so writers can compare and reuse it without reading it.
    """

    axis_id: str
    first: int
    last: int
    digest: str

    def read(self) -> pa.Array:
        """Return the axis timestamps.

        Returns:
            Validated integer timestamps.
        """
        return self._validated

    def __call__(self) -> pa.Array:
        """Return the validated timestamps, so the handle serves as a Signal's offsets loader."""
        return self.read()

    @cached_property
    def _validated(self) -> pa.Array:
        """Read the timestamps once and check them against the axis.

        Returns:
            Validated integer timestamps.

        Raises:
            TimeFFormatError: If the timestamps disagree with the axis.
        """
        offsets = self.store.read_offsets(self)
        if len(offsets) != self.length:
            raise TimeFFormatError(
                f"axis {self.axis_id!r} stores {len(offsets)} offsets but its Signals declare length {self.length}"
            )
        if offsets.null_count:
            raise TimeFFormatError(f"axis {self.axis_id!r} stores null offsets")
        endpoints = (self.first, self.last)
        if not len(offsets) or (offsets[0].as_py(), offsets[-1].as_py()) != endpoints:
            raise TimeFFormatError(f"axis {self.axis_id!r} has offsets disagreeing with its axis endpoints {endpoints}")
        if len(offsets) > 1 and pc.any(pc.less(offsets.slice(1), offsets.slice(0, len(offsets) - 1))).as_py():  # ty: ignore[unresolved-attribute]
            raise TimeFFormatError(f"axis {self.axis_id!r} has decreasing offsets")
        return offsets


class PackedValueReader:
    """Share a bounded cache of packed value groups between field handles."""

    def __init__(self, version: "DatasetVersion") -> None:
        self.version = version
        self._parquet = PayloadTableCache(_CACHE_FILES)

    def read(self, handle: TextHandle) -> str | list[str]:
        """Read packed text from its physical file.

        Returns:
            Native text or a string list.
        """
        return self._stored_value(handle).as_py()

    def read_offsets(self, handle: OffsetHandle) -> pa.Array:
        """Read a packed timestamp array from its physical file.

        Returns:
            The stored integer timestamps.
        """
        return self._stored_value(handle).values

    def _stored_value(self, handle: ValueHandle[object]) -> pa.Scalar:
        """Return the Arrow scalar a locator addresses.

        Returns:
            The list or string scalar at the handle's file and row.

        Raises:
            TimeFFormatError: If the locator or stored value is invalid.
        """
        table = self._parquet.read_table(self.version, handle.path)
        location = f"{handle.path}:{handle.row}"
        if handle.kind.value not in table.column_names:
            raise TimeFFormatError(f"unknown external value kind at {location}")
        if not 0 <= handle.row < len(table):
            raise TimeFFormatError(f"external value row out of bounds at {location}")
        value = table[handle.kind.value][handle.row]
        if not value.is_valid:
            raise TimeFFormatError(f"missing external {handle.kind} at {location}")
        return value

    def close(self) -> None:
        """Release cached payload tables."""
        self._parquet.close()

    @staticmethod
    def locators(
        connection: duckdb.DuckDBPyConnection, kind: ObjectKind, object_ids: Iterable[str]
    ) -> dict[ExternalObjectFieldKey, PackedValueLocator]:
        """Read locators for one batch of objects without opening payload files.

        Returns:
            Validated locators keyed by entity kind, public ID, field, and position.

        Raises:
            TimeFFormatError: If a catalog locator is invalid or duplicated.
        """
        identities = list(dict.fromkeys(object_ids))
        if not identities:
            # This batch contains no objects that need a locator lookup.
            return {}
        columns = ", ".join(EXTERNAL_VALUES.column_names)
        rows = connection.execute(
            f"SELECT {columns} FROM {EXTERNAL_VALUES.name} "  # noqa: S608 - fixed identifiers
            "WHERE object_kind = ? AND object_id IN (SELECT unnest(?))",
            [kind.value, identities],
        ).fetchall()
        locators: dict[ExternalObjectFieldKey, PackedValueLocator] = {}
        for row in rows:
            try:
                locator = PackedValueLocator.model_validate(dict(zip(EXTERNAL_VALUES.column_names, row, strict=True)))
            except ValueError as exc:
                raise TimeFFormatError(f"invalid external locator: {exc}") from exc
            key = locator.key
            if key in locators:
                raise TimeFFormatError(f"duplicate external locator for {key}")
            locators[key] = locator
        return locators

    def text_handle(self, locator: PackedValueLocator) -> TextHandle:
        """Return a handle that reads the packed text a locator addresses.

        Returns:
            A text handle that performs no payload I/O until read.
        """
        return TextHandle(self, locator.value_path, locator.row_index, locator.value_kind, locator.n_values)


@dataclass(frozen=True, slots=True, kw_only=True)
class StoredValueColumns:
    """Name the columns of one public Arrow table whose values can be packed."""

    kind: ObjectKind
    """Object kind whose locators address the packed values."""
    identity: str
    """Column holding each row's public object ID."""
    fields: Mapping[PackedField, Mapping[PackedValueKind, str]]
    """Per packed field, the column that holds each value kind."""
    position: str | None = None
    """Column holding the ordered position, for a field with several values per object."""


TASK_COLUMNS = StoredValueColumns(
    kind=ObjectKind.TASK,
    identity="task_id",
    fields={
        PackedField.PROMPT: {PackedValueKind.TEXT: "prompt"},
        PackedField.RATIONALE: {PackedValueKind.TEXT: "rationale"},
    },
)
"""Packable columns of the task table."""

TARGET_COLUMNS = StoredValueColumns(
    kind=ObjectKind.TASK,
    identity="task_id",
    fields={PackedField.TARGETS: {PackedValueKind.TEXT: "text_value"}},
    position="position",
)
"""Packable columns of the task-target table."""

ANNOTATION_COLUMNS = StoredValueColumns(
    kind=ObjectKind.ANNOTATION,
    identity="content_id",
    fields={PackedField.VALUE: {PackedValueKind.TEXT: "text_value", PackedValueKind.TEXT_LIST: "text_list_value"}},
)
"""Packable columns of the annotation table."""


def materialize_table(
    table: pa.Table, reader: PackedValueReader, connection: duckdb.DuckDBPyConnection, columns: StoredValueColumns
) -> pa.Table:
    """Resolve external text fields while retaining the public Arrow table schema.

    Returns:
        The original table when it has no external values, otherwise a table with native values.
    """
    if not len(table):
        return table
    identities = table[columns.identity].to_pylist()
    handles = {
        key: reader.text_handle(locator)
        for key, locator in reader.locators(connection, columns.kind, identities).items()
    }
    if not handles:
        return table
    positions = [0] * len(table) if columns.position is None else table[columns.position].to_pylist()
    changed: dict[str, list] = {}
    resolved: dict[ExternalObjectFieldKey, str | list[str]] = {}
    for index, (identity, position) in enumerate(zip(identities, positions, strict=True)):
        for field, value_columns in columns.fields.items():
            key = ExternalObjectFieldKey(object_kind=columns.kind, object_id=identity, field=field, position=position)
            handle = handles.get(key)
            if handle is None:
                continue
            column = value_columns[handle.kind]
            if column not in changed:
                changed[column] = table[column].to_pylist()
            if key not in resolved:
                resolved[key] = handle.read()
            changed[column][index] = resolved[key]
    for column, values in changed.items():
        index = table.schema.get_field_index(column)
        table = table.set_column(
            index, table.schema.field(index), pa.array(values, type=table.schema.field(index).type)
        )
    return table


class PackedValueWriter:
    """Pack large local values into Parquet files and record their catalog locations."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: list[str] = []
        self.rows: list[dict] = []
        self.locators: list[PackedValueLocator] = []
        self.bytes = 0

    def pack(
        self, kind: ObjectKind, identity: str, field: PackedField, value: object, position: int = 0
    ) -> InlineValue | ExternalValue:
        """Keep a value inline or register its external location.

        Returns:
            An explicit inline value or the locator of a packed value. Catalog writers choose
            the storage representation; this method never returns a placeholder.

        Raises:
            TimeFValidationError: If the external locator is invalid for this field.
        """
        ExternalObjectFieldKey(object_kind=kind, object_id=identity, field=field, position=position)
        classified = _classify_value(value)
        if isinstance(classified, InlineValue):
            return classified
        value = classified.value
        value_kind = classified.kind
        encoded = (
            value.to_numpy().tobytes()
            if isinstance(value, pa.Array)
            else json.dumps(value, ensure_ascii=False).encode()
        )
        if value_kind != PackedValueKind.OFFSETS and len(encoded) <= INLINE_LIMIT:
            return InlineValue(value)
        path = part_path(PAYLOAD_TEMPLATE, len(self.files))
        row = len(self.rows)
        length = len(value)
        digest = offsets_digest(value) if isinstance(value, pa.Array) else None
        try:
            locator = PackedValueLocator(
                object_kind=kind,
                object_id=identity,
                field=field,
                position=position,
                value_path=path,
                row_index=row,
                value_kind=value_kind,
                n_values=length,
                value_digest=digest,
            )
        except ValidationError as exc:
            raise TimeFValidationError(f"invalid external locator for {kind.value} {identity!r}: {exc}") from exc
        self.rows.append({value_kind.value: value.to_pylist() if isinstance(value, pa.Array) else value})
        self.bytes += len(encoded)
        self.locators.append(locator)
        if self.bytes >= _PACK_BYTES:
            self.flush()
        return ExternalValue(locator)

    def flush(self) -> None:
        """Write one packed group and release its decoded values."""
        if not self.rows:
            return
        path = part_path(PAYLOAD_TEMPLATE, len(self.files))
        (self.root / path).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(self.rows, schema=_SCHEMA), self.root / path, compression=DEFAULT_COMPRESSION
        )
        self.files.append(path)
        self.rows.clear()
        self.bytes = 0

    def finish(self, connection: duckdb.DuckDBPyConnection) -> None:
        """Flush payload files and store their explicit catalog locators."""
        self.flush()
        LOCATOR_TABLE.write(connection, self.locators)
