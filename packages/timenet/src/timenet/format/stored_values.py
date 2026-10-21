"""Read and write text, string lists, and timestamps stored in Parquet payloads."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum, unique
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import duckdb
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
from timenet.types.reference_keys import OwnerKey
from timenet.types.text import TextValue


if TYPE_CHECKING:
    from timenet.format.storage import StorageSources
    from timenet.registry.version import DatasetVersion


INLINE_LIMIT = 4096
_PACK_BYTES = 4 * 1024 * 1024
_CACHE_GROUPS = 8
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


@dataclass(frozen=True)
class ValueHandle:
    """A locator for one external value; constructing it performs no payload I/O."""

    store: "PackedValueReader"
    owner: OwnerKey | None
    path: str
    row: int
    kind: PackedValueKind
    length: int
    digest: str | None

    @property
    def value_type(self) -> Literal["str", "list"]:
        """Return the native text type without opening a payload file."""
        return "str" if self.kind == PackedValueKind.TEXT else "list"

    def read(self) -> Any:
        """Materialize this external value.

        Returns:
            Native text, a string list, or an Arrow timestamp array.
        """
        return self.store.read(self)

    def __call__(self) -> Any:
        """Return the materialized value through the callable loader interface."""
        return self.read()

    def __deepcopy__(self, memo: dict) -> "ValueHandle":
        return self

    def physical_version(self) -> "DatasetVersion":
        """Return the direct physical source without reading payload data."""
        return self.store.version.resolve_payload_source(self.owner)


@dataclass(frozen=True)
class OffsetHandle(ValueHandle):
    """An external timestamp locator with verified axis endpoints."""

    digest: str
    first: int
    last: int

    def read(self) -> pa.Array:
        """Return timestamps after checking length, endpoints, and ordering.

        Returns:
            Validated integer timestamps.

        Raises:
            TimeFFormatError: If timestamps disagree with the axis.
        """
        offsets = super().read()
        if len(offsets) != self.length or offsets.null_count:
            raise TimeFFormatError(f"invalid offset count or null offsets at {self.path}:{self.row}")
        if not len(offsets) or (offsets[0].as_py(), offsets[-1].as_py()) != (self.first, self.last):
            raise TimeFFormatError("stored offsets disagreeing with its axis endpoints")
        if len(offsets) > 1 and pc.any(pc.less(offsets.slice(1), offsets.slice(0, len(offsets) - 1))).as_py():  # ty: ignore[unresolved-attribute]
            raise TimeFFormatError("stored axis has decreasing offsets")
        return offsets


class PackedValueReader:
    """Share a bounded cache of packed value groups between field handles."""

    def __init__(self, version: "DatasetVersion") -> None:
        self.version = version
        self._parquet = PayloadTableCache(_CACHE_GROUPS)

    def read(self, handle: ValueHandle) -> Any:
        """Read a packed value from its physical file.

        Returns:
            Native text, string list, or an Arrow timestamp array.

        Raises:
            TimeFFormatError: If the locator or stored value is invalid.
        """
        table = self._parquet.read_table(handle.physical_version(), handle.path)
        location = f"{handle.path}:{handle.row}"
        if handle.kind.value not in table.column_names:
            raise TimeFFormatError(f"unknown external value kind at {location}")
        if not 0 <= handle.row < len(table):
            raise TimeFFormatError(f"external value row out of bounds at {location}")
        value = table[handle.kind.value][handle.row]
        if not value.is_valid:
            raise TimeFFormatError(f"missing external {handle.kind} at {location}")
        if handle.kind == PackedValueKind.OFFSETS:
            return value.values
        return value.as_py()

    def close(self) -> None:
        """Release cached payload tables."""
        self._parquet.close()

    def locators(
        self, connection: duckdb.DuckDBPyConnection, kind: ObjectKind, object_ids: Iterable[str]
    ) -> dict[ExternalObjectFieldKey, ValueHandle]:
        """Read locators for one batch of objects without opening payload files.

        Returns:
            Handles keyed by entity kind, public ID, field, and position.

        Raises:
            TimeFFormatError: If a catalog locator is invalid or duplicated.
        """
        identities = list(dict.fromkeys(object_ids))
        if not identities:
            # This batch contains no objects that need a locator lookup.
            return {}
        rows = connection.execute(
            "SELECT * FROM external_values WHERE object_kind = ? AND object_id IN (SELECT unnest(?))",
            [kind.value, identities],
        ).fetchall()
        handles: dict[ExternalObjectFieldKey, ValueHandle] = {}
        for row in rows:
            try:
                locator = PackedValueLocator.model_validate(dict(zip(EXTERNAL_VALUES.column_names, row, strict=True)))
            except ValueError as exc:
                raise TimeFFormatError(f"invalid external locator: {exc}") from exc
            key = locator.key
            if key in handles:
                raise TimeFFormatError(f"duplicate external locator for {key}")
            handles[key] = ValueHandle(
                self,
                locator.owner_key,
                locator.value_path,
                locator.row_index,
                locator.value_kind,
                locator.n_values,
                locator.value_digest,
            )
        return handles


def _stored_value_columns(table: pa.Table, kind: ObjectKind) -> tuple[str, dict[PackedField, tuple[str, ...]]]:
    """Identify the columns used to resolve stored values in a public Arrow table.

    Args:
        table: The annotation, task, or task-target table to materialize.
        kind: The object kind used to look up stored-value handles.

    Returns:
        The identity column and a mapping from stored field names to public value columns.
    """
    if kind == ObjectKind.ANNOTATION:
        return "content_id", {PackedField.VALUE: ("text_value", "text_list_value")}
    if "position" in table.column_names:
        return "task_id", {PackedField.TARGETS: ("text_value",)}
    return "task_id", {PackedField.PROMPT: ("prompt",), PackedField.RATIONALE: ("rationale",)}


def materialize_table(
    table: pa.Table, reader: PackedValueReader, connection: duckdb.DuckDBPyConnection, kind: ObjectKind
) -> pa.Table:
    """Resolve external text fields while retaining the public Arrow table schema.

    Returns:
        The original table when it has no external values, otherwise a table with native values.
    """
    if not len(table):
        return table
    identity_column, fields = _stored_value_columns(table, kind)
    identities = table[identity_column].to_pylist()
    handles = reader.locators(connection, kind, identities)
    if not handles:
        return table
    positions = table["position"].to_pylist() if "position" in table.column_names else [0] * len(table)
    changed: dict[str, list] = {}
    resolved: dict[ExternalObjectFieldKey, Any] = {}
    for index, (identity, position) in enumerate(zip(identities, positions, strict=True)):
        for field, columns in fields.items():
            key = ExternalObjectFieldKey(object_kind=kind, object_id=identity, field=field, position=position)
            handle = handles.get(key)
            if handle is None:
                continue
            column = columns[-1] if handle.kind == PackedValueKind.TEXT_LIST else columns[0]
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
    """Pack new values and preserve direct physical locations for inherited values."""

    def __init__(self, root: Path, sources: "StorageSources") -> None:
        self.root = root
        self.sources = sources
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
        if isinstance(value, ValueHandle):
            owner = self.sources.add(value.physical_version(), {value.path})
            path, row, value_kind, length, digest = value.path, value.row, value.kind, value.length, value.digest
        else:
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
            owner = None
            length = len(value)
            digest = (
                CHECKSUM_PREFIX + hashlib.sha256(encoded).hexdigest() if value_kind == PackedValueKind.OFFSETS else None
            )
            self.rows.append({value_kind.value: value.to_pylist() if isinstance(value, pa.Array) else value})
            self.bytes += len(encoded)
        try:
            locator = PackedValueLocator(
                object_kind=kind,
                object_id=identity,
                field=field,
                position=position,
                owner_key=owner,
                value_path=path,
                row_index=row,
                value_kind=value_kind,
                n_values=length,
                value_digest=digest,
            )
        except ValidationError as exc:
            raise TimeFValidationError(f"invalid external locator for {kind.value} {identity!r}: {exc}") from exc
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
