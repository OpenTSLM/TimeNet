"""Read and write text, string lists, and time offsets stored in Parquet payloads."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field as dataclass_field, replace
from enum import StrEnum, unique
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Self, assert_never, cast

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
from timenet.format.stored_array import ChunkLocation
from timenet.format.table_writer import ModelTable
from timenet.storage.parquet import PayloadTableCache
from timenet.types._model import TimeFModel
from timenet.types.handles import ArrayHandle
from timenet.types.object_kind import ObjectKind
from timenet.types.reference_keys import OwnerKey
from timenet.types.text import TextHandle, TextListHandle, read_value


if TYPE_CHECKING:
    from timenet.format.storage import StorageSources
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
"""Payload files a reader keeps decoded, about 32 MiB of text and time offsets at the packing target."""
_SCHEMA = pa.schema([("text", pa.string()), ("texts", pa.list_(pa.string())), ("time_offsets", pa.list_(pa.int64()))])


@unique
class PackedValueKind(StrEnum):
    """Represent the value shape stored at a packed payload location."""

    TEXT = "text"
    TEXT_LIST = "texts"
    TIME_OFFSETS = "time_offsets"


@unique
class PackedField(StrEnum):
    """Name the object fields that can store external values."""

    PROMPT = "prompt"
    RATIONALE = "rationale"
    TARGETS = "targets"
    VALUE = "value"
    TIME_OFFSETS = "time_offsets"


_FIELD_OWNERS = {
    PackedField.PROMPT: ObjectKind.TASK,
    PackedField.RATIONALE: ObjectKind.TASK,
    PackedField.TARGETS: ObjectKind.TASK,
    PackedField.VALUE: ObjectKind.ANNOTATION,
    PackedField.TIME_OFFSETS: ObjectKind.AXIS,
}


_FIELD_VALUE_KINDS = {
    PackedField.PROMPT: (PackedValueKind.TEXT,),
    PackedField.RATIONALE: (PackedValueKind.TEXT,),
    PackedField.TARGETS: (PackedValueKind.TEXT,),
    PackedField.VALUE: (PackedValueKind.TEXT, PackedValueKind.TEXT_LIST),
    PackedField.TIME_OFFSETS: (PackedValueKind.TIME_OFFSETS,),
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


@dataclass(frozen=True, slots=True, kw_only=True)
class PackedLocation:
    """Locate one value packed in a Parquet payload file of a release.

    The catalog records every field, so the value's release, kind, length, and digest are known
    without opening the payload file.
    """

    owner: OwnerKey | None = None
    """Release that stores the payload file, or ``None`` for the release whose catalog records it."""
    path: str
    """Payload file, relative to the root of the release that stores it."""
    row: int
    """Row of the value in the payload file."""
    kind: PackedValueKind
    """Kind of the value, which names the payload column that holds it."""
    length: int
    """Number of code points in a string, of items in a string list, or of time offsets."""
    digest: bytes | None = None
    """SHA-256 digest of time offsets, which identifies them without a read. ``None`` for text."""


PayloadLocation = PackedLocation | ChunkLocation
"""Where a stored value's bytes live: packed text or time offsets, or a signal array's chunks."""


class PackedValueLocator(TimeFModel):
    """Validate the catalog location and value kind of one external field."""

    object_kind: ObjectKind
    object_id: str = Field(strict=True, min_length=1)
    field: PackedField
    position: int = Field(strict=True, ge=0)
    owner_key: OwnerKey | None = Field(default=None, strict=True, min_length=1)
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

    @property
    def location(self) -> PackedLocation:
        """Return where the value is packed."""
        digest = self.value_digest
        return PackedLocation(
            owner=self.owner_key,
            path=self.value_path,
            row=self.row_index,
            kind=self.value_kind,
            length=self.n_values,
            digest=None if digest is None else bytes.fromhex(digest.removeprefix(CHECKSUM_PREFIX)),
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
        if (self.value_kind == PackedValueKind.TIME_OFFSETS) != (self.value_digest is not None):
            raise TimeFValidationError("external time offsets require a digest; text values must not store one")
        return self


LOCATOR_TABLE = ModelTable[PackedValueLocator](EXTERNAL_VALUES)


@dataclass(frozen=True, slots=True)
class CatalogValue:
    """Hold a value kept in its catalog row, including a legitimate absent value."""

    value: object


@dataclass(frozen=True, slots=True)
class _PackableValue:
    """Pair a supported payload value with its storage kind before deciding where it fits."""

    value: str | list[str] | pa.Array
    kind: PackedValueKind


def _classify_value(value: object) -> CatalogValue | _PackableValue:
    """Read text handles and distinguish payload candidates from values kept in the catalog row.

    Returns:
        A typed payload candidate or a catalog value, including an absent value.

    Raises:
        TimeFValidationError: If a string list or time offset array has an unsupported element type.
    """
    value = read_value(value)
    if isinstance(value, str):
        return _PackableValue(value, PackedValueKind.TEXT)
    if isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise TimeFValidationError("packed string lists must contain only strings")
        return _PackableValue(cast("list[str]", value), PackedValueKind.TEXT_LIST)
    if isinstance(value, pa.Array):
        if value.type != pa.int64() or value.null_count:
            raise TimeFValidationError("packed time offsets must be non-null int64 values")
        return _PackableValue(value, PackedValueKind.TIME_OFFSETS)
    return CatalogValue(value)


@dataclass(frozen=True, slots=True)
class ExternalValue:
    """Identify a value written to a payload instead of the catalog."""

    locator: PackedValueLocator


def offsets_digest(offsets: np.ndarray | pa.Array) -> bytes:
    """Return the SHA-256 digest of an irregular axis's time offsets.

    Every caller hashes the same bytes, contiguous little-endian int64, so one sequence of time
    offsets has one digest whether it is held as NumPy or as Arrow.

    Returns:
        The 32-byte digest.
    """
    values = offsets.to_numpy(zero_copy_only=False) if isinstance(offsets, pa.Array) else offsets
    canonical = np.ascontiguousarray(values, dtype="<i8")
    return hashlib.sha256(memoryview(canonical)).digest()


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
    """Return whether ``other`` reads the same payload row of the same reader as ``handle``."""
    return (
        isinstance(other, StoredText | StoredTextList)
        and other.store is handle.store
        and other._location == handle._location
    )


@dataclass(frozen=True, slots=True, eq=False)
class StoredText(TextHandle):
    """Refer to one string packed in a Parquet payload.

    The handle holds the reader of the release's payloads and the string's location, not the string,
    so it is never loaded and every ``read()`` reads the payload. Handles from one reader at the same
    location are equal without a read.
    """

    store: "PackedValueReader" = dataclass_field(repr=False)
    _packed_location: PackedLocation

    @property
    def _location(self) -> PackedLocation:
        """Return where the string is packed."""
        return self._packed_location

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
    it is never loaded and every ``read()`` reads the payload. Handles from one reader at the same
    location are equal without a read.
    """

    store: "PackedValueReader" = dataclass_field(repr=False)
    _packed_location: PackedLocation

    @property
    def _location(self) -> PackedLocation:
        """Return where the list is packed."""
        return self._packed_location

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

    __hash__ = None

    def __eq__(self, other: object) -> bool:
        # A slotted dataclass breaks zero-argument super() before Python 3.14, so name the base.
        return _same_payload(self, other) or TextListHandle.__eq__(self, other)

    def __deepcopy__(self, memo: dict) -> Self:
        return self


@dataclass(frozen=True, slots=True, eq=False)
class StoredOffsets(ArrayHandle):
    """Refer to the time offsets of one irregular axis, packed in a Parquet payload.

    The handle holds the reader of the release's payloads, the time offsets' location, and the axis's
    ID and endpoints. Its first ``read()`` checks the time offsets against the location and the axis
    and keeps a compact copy, so from then on the handle is loaded and reads need no payload access.
    The location records the number of time offsets and their digest, so both are known without a
    read.
    """

    store: "PackedValueReader" = dataclass_field(repr=False)
    _packed_location: PackedLocation
    axis_id: str
    """ID of the axis the time offsets belong to."""
    first: int
    """First time offset the axis declares, in microseconds."""
    last: int
    """Last time offset the axis declares, in microseconds."""
    _offsets: pa.Array | None = dataclass_field(default=None, init=False, repr=False)

    @property
    def _location(self) -> PackedLocation:
        """Return where the time offsets are packed."""
        return self._packed_location

    @property
    def is_loaded(self) -> bool:
        """Return whether a read has kept the time offsets."""
        return self._offsets is not None

    @property
    def length(self) -> int:
        """Return the number of time offsets, recorded in their location."""
        return self._location.length

    @property
    def digest(self) -> bytes:
        """Return the time offsets' SHA-256 digest, recorded in their location.

        The catalog stores a digest with every axis's time offsets, so only a location built by hand
        can lack one.

        Raises:
            TimeFFormatError: If the location records no digest.
        """
        digest = self._location.digest
        if digest is None:
            raise TimeFFormatError(f"axis {self.axis_id!r} has time offsets without a digest")
        return digest

    def read(self) -> pa.Array:
        """Return the time offsets, read and checked on the first call and kept.

        Returns:
            One int64 microsecond offset per step.
        """
        offsets = self._offsets
        if offsets is None:
            stored = self.store.read_scalar(self._location).values
            self._check(stored)
            # The stored array views its payload file's whole column, so keep a copy of this axis only.
            offsets = pa.concat_arrays([stored])
            object.__setattr__(self, "_offsets", offsets)  # noqa: PLC2801 - keeps the checked time offsets
        return offsets

    def _check(self, offsets: pa.Array) -> None:
        """Check time offsets read from the payload against their location and the axis.

        Raises:
            TimeFFormatError: If their count differs from the location's, or they contain nulls,
                disagree with the axis endpoints, or decrease.
        """
        if len(offsets) != self.length:
            raise TimeFFormatError(
                f"axis {self.axis_id!r} stores {len(offsets)} offsets but its location records {self.length}"
            )
        if offsets.null_count:
            raise TimeFFormatError(f"axis {self.axis_id!r} stores null offsets")
        endpoints = (self.first, self.last)
        if not len(offsets) or (offsets[0].as_py(), offsets[-1].as_py()) != endpoints:
            raise TimeFFormatError(f"axis {self.axis_id!r} has offsets disagreeing with its axis endpoints {endpoints}")
        if len(offsets) > 1 and pc.any(pc.less(offsets.slice(1), offsets.slice(0, len(offsets) - 1))).as_py():  # ty: ignore[unresolved-attribute]
            raise TimeFFormatError(f"axis {self.axis_id!r} has decreasing offsets")

    def __deepcopy__(self, memo: dict) -> Self:
        return self


class PackedValueReader:
    """Read one release's packed values through a bounded cache of decoded payload files.

    A value whose location names an owner is read from that payload source of the release.
    """

    def __init__(self, version: "DatasetVersion") -> None:
        self.version = version
        self._parquet = PayloadTableCache(_CACHE_FILES)

    def read_scalar(self, location: PackedLocation) -> pa.Scalar:
        """Read a packed value from the payload file of the release that stores it.

        Returns:
            The Arrow scalar at the location's file and row: a string, or a list of strings or time offsets.

        Raises:
            TimeFFormatError: If the location or stored value is invalid.
        """
        table = self._parquet.read_table(self.version.resolve_payload_source(location.owner), location.path)
        where = f"{location.path}:{location.row}"
        if location.kind.value not in table.column_names:
            raise TimeFFormatError(f"unknown external value kind at {where}")
        if not 0 <= location.row < len(table):
            raise TimeFFormatError(f"external value row out of bounds at {where}")
        value = table[location.kind.value][location.row]
        if not value.is_valid:
            raise TimeFFormatError(f"missing external {location.kind} at {where}")
        return value

    def handle(self, location: PackedLocation) -> StoredText | StoredTextList:
        """Return a text handle of the location's value kind, without reading the payload.

        Returns:
            A ``StoredText`` for a string, or a ``StoredTextList`` for a list of strings.

        Raises:
            TimeFValidationError: If the location holds time offsets, whose ``StoredOffsets`` also
                needs their axis.
        """
        kind = location.kind
        match kind:
            case PackedValueKind.TEXT:
                return StoredText(self, location)
            case PackedValueKind.TEXT_LIST:
                return StoredTextList(self, location)
            case PackedValueKind.TIME_OFFSETS:
                raise TimeFValidationError(
                    f"time offsets at {location.path}:{location.row} are read through StoredOffsets, not a text handle"
                )
            case _:
                assert_never(kind)

    def close(self) -> None:
        """Release cached payload tables."""
        self._parquet.close()

    @staticmethod
    def locations(
        connection: duckdb.DuckDBPyConnection, kind: ObjectKind, object_ids: Iterable[str]
    ) -> dict[ExternalObjectFieldKey, PackedLocation]:
        """Return where one batch of objects' values are packed, without opening payload files.

        Returns:
            Locations keyed by entity kind, public ID, field, and position.

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
        locations: dict[ExternalObjectFieldKey, PackedLocation] = {}
        for row in rows:
            try:
                locator = PackedValueLocator.model_validate(dict(zip(EXTERNAL_VALUES.column_names, row, strict=True)))
            except ValueError as exc:
                raise TimeFFormatError(f"invalid external locator: {exc}") from exc
            key = locator.key
            if key in locations:
                raise TimeFFormatError(f"duplicate external locator for {key}")
            locations[key] = locator.location
        return locations

    def handles(
        self, connection: duckdb.DuckDBPyConnection, kind: ObjectKind, object_ids: Iterable[str]
    ) -> dict[ExternalObjectFieldKey, StoredText | StoredTextList]:
        """Return text handles for one batch of objects' packed values without opening payload files.

        Returns:
            Handles keyed by entity kind, public ID, field, and position.
        """
        return {key: self.handle(location) for key, location in self.locations(connection, kind, object_ids).items()}


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
    """Pack new values into Parquet payload files, and reuse the payloads that store read values."""

    def __init__(self, root: Path, sources: "StorageSources") -> None:
        self.root = root
        self.sources = sources
        self.files: list[str] = []
        self.rows: list[dict] = []
        self.locators: list[PackedValueLocator] = []
        self.bytes = 0

    def pack(  # noqa: PLR0913 - a known digest spares hashing the time offsets twice
        self,
        kind: ObjectKind,
        identity: str,
        field: PackedField,
        value: object,
        position: int = 0,
        *,
        digest: bytes | None = None,
    ) -> CatalogValue | ExternalValue:
        """Keep a value inline, pack it, or reuse the payload that stores it.

        A stored text or time-offsets handle keeps its payload row, so its value is neither read nor
        copied. Its locator names the row in the release that stores it.

        Args:
            digest: The ``offsets_digest()`` of time offsets, when the caller has it already.
                Without it, the time offsets are hashed here. Text takes none.

        Returns:
            An explicit catalog value or the locator of a packed value. Catalog writers choose
            the storage representation; this method never returns a placeholder.

        Raises:
            TimeFValidationError: If the locator is invalid for this field, or a reused payload is
                stored by the output release or missing from the release that stores it.
        """  # noqa: DOC502 - raised by _add_locator and StorageSources.add
        key = ExternalObjectFieldKey(object_kind=kind, object_id=identity, field=field, position=position)
        if isinstance(value, StoredText | StoredTextList | StoredOffsets):
            stored = value._location
            owner = self.sources.add(value.store.version, stored)
            return self._add_locator(key, replace(stored, owner=owner))
        classified = _classify_value(value)
        if isinstance(classified, CatalogValue):
            return classified
        value = classified.value
        value_kind = classified.kind
        size = value.nbytes if isinstance(value, pa.Array) else len(json.dumps(value, ensure_ascii=False).encode())
        if value_kind != PackedValueKind.TIME_OFFSETS and size <= INLINE_LIMIT:
            return CatalogValue(value)
        if isinstance(value, pa.Array) and digest is None:
            digest = offsets_digest(value)
        external = self._add_locator(
            key,
            PackedLocation(
                path=part_path(PAYLOAD_TEMPLATE, len(self.files)),
                row=len(self.rows),
                kind=value_kind,
                length=len(value),
                digest=digest if isinstance(value, pa.Array) else None,
            ),
        )
        self.rows.append({value_kind.value: value.to_pylist() if isinstance(value, pa.Array) else value})
        self.bytes += size
        if self.bytes >= _PACK_BYTES:
            self.flush()
        return external

    def _add_locator(self, key: ExternalObjectFieldKey, location: PackedLocation) -> ExternalValue:
        """Record the catalog locator that names a field's packed value.

        Returns:
            The recorded locator.

        Raises:
            TimeFValidationError: If the location is invalid for this field.
        """
        digest = location.digest
        try:
            locator = PackedValueLocator(
                object_kind=key.object_kind,
                object_id=key.object_id,
                field=key.field,
                position=key.position,
                owner_key=location.owner,
                value_path=location.path,
                row_index=location.row,
                value_kind=location.kind,
                n_values=location.length,
                value_digest=None if digest is None else CHECKSUM_PREFIX + digest.hex(),
            )
        except ValidationError as exc:
            raise TimeFValidationError(
                f"invalid external locator for {key.object_kind.value} {key.object_id!r}: {exc}"
            ) from exc
        self.locators.append(locator)
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
