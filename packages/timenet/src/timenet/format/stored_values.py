"""Read and write text and string lists stored in Parquet payloads."""

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field as dataclass_field
from enum import StrEnum, unique
import json
from pathlib import Path
from typing import TYPE_CHECKING, Self, assert_never, cast

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, ValidationError, field_validator, model_validator

from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.format.constants import DEFAULT_COMPRESSION, PAYLOAD_TEMPLATE, check_relative_path, part_path
from timenet.format.control_schema import EXTERNAL_VALUES
from timenet.format.table_writer import ModelTable
from timenet.storage.parquet import PayloadTableCache
from timenet.types._model import TimeFModel
from timenet.types.object_kind import ObjectKind
from timenet.types.text import TextHandle, TextListHandle, read_value


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

_CACHE_BYTES = 64 * 2**20
"""Decoded payload bytes a reader keeps in memory.

The most recent file is kept even when it alone exceeds the budget.
"""

_SCHEMA = pa.schema([("text", pa.string()), ("texts", pa.list_(pa.field("element", pa.string(), nullable=False)))])
"""Exact schema of a payload file, which readers check once per file.

A row holds a string in ``text`` or a list in ``texts``, so both columns are nullable. A list's
strings are not.
"""


@unique
class PackedValueKind(StrEnum):
    """Represent the value shape stored at a packed payload location."""

    TEXT = "text"
    TEXT_LIST = "texts"


@unique
class PackedField(StrEnum):
    """Name the object fields that can store external values."""

    PROMPT = "prompt"
    RATIONALE = "rationale"
    TARGETS = "targets"
    VALUE = "value"


_FIELD_OWNERS = {
    PackedField.PROMPT: ObjectKind.TASK,
    PackedField.RATIONALE: ObjectKind.TASK,
    PackedField.TARGETS: ObjectKind.TASK,
    PackedField.VALUE: ObjectKind.ANNOTATION_CONTENT,
}
"""The object kind that owns each field: a task, or the content row that holds an annotation value."""


_FIELD_VALUE_KINDS = {
    PackedField.PROMPT: (PackedValueKind.TEXT,),
    PackedField.RATIONALE: (PackedValueKind.TEXT,),
    PackedField.TARGETS: (PackedValueKind.TEXT,),
    PackedField.VALUE: (PackedValueKind.TEXT, PackedValueKind.TEXT_LIST),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class ExternalObjectFieldKey:
    """Identify an external field on an object, including its position in an ordered field.

    The object is named the way the catalog names it: a task by its public id, an annotation's
    content by the storage id of its ``annotation_contents`` row.
    """

    object_kind: ObjectKind
    object_id: int
    field: PackedField
    position: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.field, PackedField):
            raise TimeFValidationError(f"external field must be a PackedField, got {self.field!r}")
        if _FIELD_OWNERS[self.field] != self.object_kind:
            raise TimeFValidationError(f"unsupported external field {self.object_kind.value}.{self.field.value}")
        if self.position < 0 or (self.field != PackedField.TARGETS and self.position != 0):
            raise TimeFValidationError(f"invalid position {self.position} for external field {self.field.value}")


@dataclass(frozen=True, slots=True, kw_only=True)
class PackedLocation:
    """Locate one value packed in a Parquet payload file of a release.

    The catalog records every field, so the value's kind and length are known without opening the
    payload file.
    """

    path: str
    """Payload file, relative to the release root."""
    row: int
    """Row of the value in the payload file."""
    kind: PackedValueKind
    """Kind of the value, which names the payload column that holds it."""
    length: int
    """Number of code points in a string, or of items in a string list."""


PayloadLocation = PackedLocation
"""Where a stored value's bytes live."""


@dataclass(frozen=True, slots=True, kw_only=True)
class PackedContentKey:
    """Name one packed value by its payload file's checksum and its place in the file, without a read.

    The checksum identifies the file's bytes, so equal keys mean equal values, and readers of one
    release, or of releases that share a payload file, give the same key. The key compares equal
    only to another ``PackedContentKey``, so it never equals the key of text in memory, which is a
    string or a tuple of strings.
    """

    checksum: str
    """The payload file's checksum, as the manifest records it."""
    row: int
    """Row of the value in the payload file."""
    kind: PackedValueKind
    """Kind of the value, which names the payload column that holds it."""


class PackedValueLocator(TimeFModel):
    """Validate the catalog location and value kind of one external field."""

    object_kind: ObjectKind
    object_id: int = Field(strict=True, gt=0)
    field: PackedField
    position: int = Field(strict=True, ge=0)
    value_path: str = Field(strict=True, min_length=1)
    row_index: int = Field(strict=True, ge=0)
    value_kind: PackedValueKind
    n_values: int = Field(strict=True, ge=0)

    @property
    def key(self) -> ExternalObjectFieldKey:
        """Return the validated object field addressed by this locator."""
        return ExternalObjectFieldKey(
            object_kind=self.object_kind, object_id=self.object_id, field=self.field, position=self.position
        )

    @property
    def location(self) -> PackedLocation:
        """Return where the value is packed."""
        return PackedLocation(path=self.value_path, row=self.row_index, kind=self.value_kind, length=self.n_values)

    @field_validator("value_path")
    @classmethod
    def _check_path(cls, value: str) -> str:
        return check_relative_path("external value_path", value)

    @model_validator(mode="after")
    def _check_field(self) -> Self:
        key = self.key
        if self.value_kind not in _FIELD_VALUE_KINDS[key.field]:
            raise TimeFValidationError(f"external field {key.field.value} cannot store {self.value_kind.value}")
        return self


LOCATOR_TABLE = ModelTable[PackedValueLocator](EXTERNAL_VALUES)


@dataclass(frozen=True, slots=True)
class CatalogValue:
    """Hold a value kept in its catalog row, including a legitimate absent value."""

    value: object


@dataclass(frozen=True, slots=True)
class _PackableValue:
    """Pair a supported payload value with its storage kind before deciding where it fits."""

    value: str | list[str]
    kind: PackedValueKind


def _classify_value(value: object) -> CatalogValue | _PackableValue:
    """Read text handles and distinguish payload candidates from values kept in the catalog row.

    Returns:
        A typed payload candidate or a catalog value, including an absent value.

    Raises:
        TimeFValidationError: If a string list has an unsupported element type.
    """
    value = read_value(value)
    if isinstance(value, str):
        return _PackableValue(value, PackedValueKind.TEXT)
    if isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise TimeFValidationError("packed string lists must contain only strings")
        return _PackableValue(cast("list[str]", value), PackedValueKind.TEXT_LIST)
    return CatalogValue(value)


@dataclass(frozen=True, slots=True)
class ExternalValue:
    """Identify a value written to a payload instead of the catalog."""

    locator: PackedValueLocator


def _check_length(length: int, location: PackedLocation) -> None:
    """Check a packed value read from its payload against the length its location records.

    Raises:
        TimeFFormatError: If the lengths differ.
    """
    if length != location.length:
        raise TimeFFormatError(
            f"external {location.kind} at {location.path}:{location.row} has length {length}, "
            f"but its locator records {location.length}"
        )


def _same_payload(handle: "StoredText | StoredTextList", other: object) -> bool:
    """Return whether ``other`` reads the same row of the same payload file as ``handle``."""
    return isinstance(other, StoredText | StoredTextList) and other._content_key() == handle._content_key()


def _stored_content_key(handle: "StoredText | StoredTextList") -> PackedContentKey:
    """Return a stored handle's content key, which names its payload file, row, and kind, without a read.

    Handles at one location of one payload file share a key, across readers too. So annotations
    read from one content row are written as one row again, and a task read by one reader finds the
    annotations on Records that another reader read.
    """
    return handle.store.content_key(handle._location)


@dataclass(frozen=True, slots=True, eq=False)
class StoredText(TextHandle):
    """Refer to one string packed in a Parquet payload.

    The handle holds the reader of the release's payloads and the string's location, not the string,
    so it is never loaded and every ``read()`` reads the payload. Handles at one location of one
    payload file are equal without a read.
    """

    store: "PackedValueReader" = dataclass_field(repr=False)
    # Required: without field(), the dataclass would take ValueHandle's None as the default.
    _location: PackedLocation = dataclass_field()

    @property
    def is_loaded(self) -> bool:
        """Return False, since every read goes to the payload."""
        return False

    @property
    def length(self) -> int:
        """Return the string's length, recorded in its location."""
        return self._location.length

    def read(self) -> str:
        """Return the string from its payload."""
        text: str = self.store.read_scalar(self._location).as_py()
        _check_length(len(text), self._location)
        return text

    def _content_key(self) -> Hashable:
        """Return the payload file's checksum, the row, and the kind, which name the string unread."""
        return _stored_content_key(self)

    __hash__ = None

    def __eq__(self, other: object) -> bool:
        # A slotted dataclass breaks zero-argument super() before Python 3.14, so name the base.
        return _same_payload(self, other) or TextHandle.__eq__(self, other)

    def __deepcopy__(self, memo: dict) -> Self:
        return self


@dataclass(frozen=True, slots=True, eq=False)
class StoredTextList(TextListHandle):
    """Refer to a list of strings packed in a Parquet payload.

    The handle holds the reader of the release's payloads and the list's location, not the list, so
    it is never loaded and every ``read()`` reads the payload. Handles at one location of one payload
    file are equal without a read.
    """

    store: "PackedValueReader" = dataclass_field(repr=False)
    # Required: without field(), the dataclass would take ValueHandle's None as the default.
    _location: PackedLocation = dataclass_field()

    @property
    def is_loaded(self) -> bool:
        """Return False, since every read goes to the payload."""
        return False

    @property
    def length(self) -> int:
        """Return the number of strings, recorded in the list's location."""
        return self._location.length

    def read(self) -> list[str]:
        """Return the list of strings from its payload."""
        texts: list[str] = self.store.read_scalar(self._location).as_py()
        _check_length(len(texts), self._location)
        return texts

    def _content_key(self) -> Hashable:
        """Return the payload file's checksum, the row, and the kind, which name the list unread."""
        return _stored_content_key(self)

    __hash__ = None

    def __eq__(self, other: object) -> bool:
        # A slotted dataclass breaks zero-argument super() before Python 3.14, so name the base.
        return _same_payload(self, other) or TextListHandle.__eq__(self, other)

    def __deepcopy__(self, memo: dict) -> Self:
        return self


class PackedValueReader:
    """Read one release's packed values through a bounded cache of decoded payload files."""

    def __init__(self, version: "DatasetVersion") -> None:
        self.version = version
        self._parquet = PayloadTableCache(_CACHE_BYTES)
        self._checksums = {part.path: part.checksum for part in version.manifest.files.payloads}

    def content_key(self, location: PackedLocation) -> PackedContentKey:
        """Return the key that names a packed value by its payload file's checksum, without a read.

        Returns:
            The checksum the manifest records for the location's file, with the row and kind.

        Raises:
            TimeFFormatError: If the manifest does not list the location's payload file.
        """
        checksum = self._checksums.get(location.path)
        if checksum is None:
            raise TimeFFormatError(f"payload file {location.path!r} is not listed in the manifest")
        return PackedContentKey(checksum=checksum, row=location.row, kind=location.kind)

    def read_scalar(self, location: PackedLocation) -> pa.Scalar:
        """Read a packed value from its physical file.

        Returns:
            The Arrow string or string-list scalar at the location's file and row.

        Raises:
            TimeFFormatError: If the payload file's schema, the location, or the stored value is invalid.
        """
        table = self._parquet.read_table(self.version, location.path, _SCHEMA)
        where = f"{location.path}:{location.row}"
        if not 0 <= location.row < len(table):
            raise TimeFFormatError(f"external value row out of bounds at {where}")
        value = table[location.kind.value][location.row]
        if not value.is_valid:
            raise TimeFFormatError(f"missing external {location.kind} at {where}")
        return value

    def handle(self, location: PackedLocation) -> StoredText | StoredTextList:
        """Return a handle of the location's value kind, without reading the payload.

        Returns:
            A ``StoredText`` for a string, or a ``StoredTextList`` for a list of strings.

        Raises:
            TimeFFormatError: If the manifest does not list the location's payload file.
        """  # noqa: DOC502 - raised by content_key
        self.content_key(location)
        kind = location.kind
        match kind:
            case PackedValueKind.TEXT:
                return StoredText(self, location)
            case PackedValueKind.TEXT_LIST:
                return StoredTextList(self, location)
            case _:
                assert_never(kind)

    def close(self) -> None:
        """Release cached payload tables."""
        self._parquet.close()

    def handles(
        self, connection: duckdb.DuckDBPyConnection, kind: ObjectKind, object_ids: Iterable[int]
    ) -> dict[ExternalObjectFieldKey, StoredText | StoredTextList]:
        """Return handles for one batch of objects' packed values without opening payload files.

        Args:
            connection: The release's open control database.
            kind: The kind of the objects that own the values.
            object_ids: The catalog ids of the objects, such as task ids or annotation content ids.

        Returns:
            Handles keyed by object kind, object id, field, and position.

        Raises:
            TimeFFormatError: If the locator table cannot be read, or a locator is invalid or duplicated.
        """
        identities = list(dict.fromkeys(object_ids))
        if not identities:
            # This batch contains no objects that need a locator lookup.
            return {}
        columns = ", ".join(EXTERNAL_VALUES.column_names)
        try:
            rows = connection.execute(
                f"SELECT {columns} FROM {EXTERNAL_VALUES.name} "  # noqa: S608 - fixed identifiers
                "WHERE object_kind = ? AND object_id IN (SELECT unnest(?))",
                [kind.value, identities],
            ).fetchall()
        except duckdb.Error as exc:
            # A release written before packed values has no locator table.
            raise TimeFFormatError(f"cannot read external values: {exc}") from exc
        handles: dict[ExternalObjectFieldKey, StoredText | StoredTextList] = {}
        for row in rows:
            try:
                locator = PackedValueLocator.model_validate(dict(zip(EXTERNAL_VALUES.column_names, row, strict=True)))
            except ValueError as exc:
                raise TimeFFormatError(f"invalid external locator: {exc}") from exc
            key = locator.key
            if key in handles:
                raise TimeFFormatError(f"duplicate external locator for {key}")
            handles[key] = self.handle(locator.location)
        return handles


@dataclass(frozen=True, slots=True, kw_only=True)
class StoredValueColumns:
    """Name the columns of one public Arrow table whose values can be packed."""

    kind: ObjectKind
    """Object kind whose locators address the packed values."""
    identity: str
    """Column holding each row's owner id: a task id, or an annotation content id."""
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
    kind=ObjectKind.ANNOTATION_CONTENT,
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
    handles = reader.handles(connection, columns.kind, identities)
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
            column = value_columns[handle._location.kind]
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
        self, kind: ObjectKind, object_id: int, field: PackedField, value: object, position: int = 0
    ) -> CatalogValue | ExternalValue:
        """Keep a value inline or register its external location.

        Args:
            kind: The kind of the object that owns the value.
            object_id: The object's id in this write's catalog: a task id, or the id of the
                annotation content row that holds the value.
            field: The object's field that holds the value.
            value: The value, a text handle, or any value that stays in the catalog.
            position: The value's position in an ordered field, such as a task's targets.

        Returns:
            An explicit catalog value or the locator of a packed value. Catalog writers choose
            the storage representation; this method never returns a placeholder.

        Raises:
            TimeFValidationError: If the external locator is invalid for this field.
        """
        ExternalObjectFieldKey(object_kind=kind, object_id=object_id, field=field, position=position)
        classified = _classify_value(value)
        if isinstance(classified, CatalogValue):
            return classified
        value = classified.value
        value_kind = classified.kind
        encoded = json.dumps(value, ensure_ascii=False).encode()
        if len(encoded) <= INLINE_LIMIT:
            return CatalogValue(value)
        path = part_path(PAYLOAD_TEMPLATE, len(self.files))
        row = len(self.rows)
        length = len(value)
        try:
            locator = PackedValueLocator(
                object_kind=kind,
                object_id=object_id,
                field=field,
                position=position,
                value_path=path,
                row_index=row,
                value_kind=value_kind,
                n_values=length,
            )
        except ValidationError as exc:
            raise TimeFValidationError(f"invalid external locator for {kind.value} {object_id!r}: {exc}") from exc
        self.rows.append({value_kind.value: value})
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

    def write_locators(self, connection: duckdb.DuckDBPyConnection, *, at_least: int = 0) -> None:
        """Store the locators packed so far in the catalog and release them.

        The catalog writer calls this once per batch inside its transaction, so a streamed source
        never holds all of its locators.

        Args:
            connection: The open write transaction.
            at_least: Keep the locators while there are fewer than this many.
        """
        if self.locators and len(self.locators) >= at_least:
            LOCATOR_TABLE.write(connection, self.locators)
            self.locators.clear()

    def finish(self, connection: duckdb.DuckDBPyConnection) -> None:
        """Flush payload files and store the remaining catalog locators."""
        self.flush()
        self.write_locators(connection)
