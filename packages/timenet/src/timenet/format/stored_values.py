"""Read and write text, string lists, and timestamps stored in Parquet payloads."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from timenet.errors import TimeFFormatError
from timenet.format.control_schema import TABLES
from timenet.storage.parquet import ParquetReadCache


if TYPE_CHECKING:
    from timenet.format.storage import StorageSources
    from timenet.registry.version import DatasetVersion


INLINE_LIMIT = 4096
_PACK_BYTES = 4 * 1024 * 1024
_CACHE_GROUPS = 8
_SCHEMA = pa.schema([("text", pa.string()), ("texts", pa.list_(pa.string())), ("offsets", pa.list_(pa.int64()))])


@dataclass(frozen=True, slots=True, kw_only=True)
class ExternalObjectFieldKey:
    """Identify an external field on an object, including its position in an ordered field."""

    object_kind: str
    object_id: str
    field: str
    position: int = 0


@dataclass(frozen=True)
class ValueHandle:
    """A locator for one external value; constructing it performs no payload I/O."""

    store: "PackedValueReader"
    owner: str | None
    path: str
    row: int
    kind: str
    length: int
    digest: str

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
        self._parquet = ParquetReadCache(_CACHE_GROUPS)

    def read(self, handle: ValueHandle) -> Any:
        """Read a packed value from its physical file.

        Returns:
            Native text, string list, or an Arrow timestamp array.

        Raises:
            TimeFFormatError: If the locator or stored value is invalid.
        """
        table = self._parquet.read_table(handle.physical_version(), handle.path)
        location = f"{handle.path}:{handle.row}"
        if handle.kind not in table.column_names:
            raise TimeFFormatError(f"unknown external value kind at {location}")
        if not 0 <= handle.row < len(table):
            raise TimeFFormatError(f"external value row out of bounds at {location}")
        value = table[handle.kind][handle.row]
        if not value.is_valid:
            raise TimeFFormatError(f"missing external {handle.kind} at {location}")
        return value.values if handle.kind == "offsets" else value.as_py()

    def close(self) -> None:
        """Release cached payload tables."""
        self._parquet.close()

    def locators(self, connection: duckdb.DuckDBPyConnection) -> dict[ExternalObjectFieldKey, ValueHandle]:
        """Read external locators without opening any external files.

        Returns:
            Handles keyed by entity kind, public ID, field, and position.
        """
        return {
            ExternalObjectFieldKey(object_kind=kind, object_id=identity, field=field, position=position): ValueHandle(
                self, owner, path, row, value_kind, length, digest
            )
            for kind, identity, field, position, owner, path, row, value_kind, length, digest in connection.execute(
                "SELECT * FROM external_values"
            ).fetchall()
        }


def _stored_value_columns(table: pa.Table, kind: str) -> tuple[str, dict[str, tuple[str, ...]]]:
    """Identify the columns used to resolve stored values in a public Arrow table.

    Args:
        table: The annotation, task, or task-target table to materialize.
        kind: The object kind used to look up stored-value handles.

    Returns:
        The identity column and a mapping from stored field names to public value columns.
    """
    if kind == "Annotation":
        return "content_id", {"value": ("text_value", "text_list_value")}
    if "position" in table.column_names:
        return "task_id", {"targets": ("text_value",)}
    return "task_id", {"prompt": ("prompt",), "rationale": ("rationale",)}


def materialize_table(table: pa.Table, handles: dict[ExternalObjectFieldKey, ValueHandle], kind: str) -> pa.Table:
    """Resolve external text fields while retaining the public Arrow table schema.

    Returns:
        The original table when it has no external values, otherwise a table with native values.
    """
    if not handles or not len(table):
        return table
    identity_column, fields = _stored_value_columns(table, kind)
    identities = table[identity_column].to_pylist()
    positions = table["position"].to_pylist() if "position" in table.column_names else [0] * len(table)
    changed: dict[str, list] = {}
    resolved: dict[ExternalObjectFieldKey, Any] = {}
    for index, (identity, position) in enumerate(zip(identities, positions, strict=True)):
        for field, columns in fields.items():
            key = ExternalObjectFieldKey(object_kind=kind, object_id=identity, field=field, position=position)
            handle = handles.get(key)
            if handle is None:
                continue
            column = columns[-1] if handle.kind == "texts" else columns[0]
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
        self.locators: list[dict] = []
        self.bytes = 0

    def pack(self, kind: str, identity: str, field: str, value: Any, position: int = 0) -> Any:
        """Return an inline value or register an external value and return its typed placeholder.

        Placeholders are paired with explicit locator rows; readers never expose them as values.

        Returns:
            The value to store in the catalog's native column.
        """
        if isinstance(value, ValueHandle):
            owner = self.sources.add(value.physical_version(), {value.path})
            path, row, value_kind, length, digest = value.path, value.row, value.kind, value.length, value.digest
        else:
            value_kind = "text" if isinstance(value, str) else "texts" if isinstance(value, list) else "offsets"
            if value_kind == "offsets" and not isinstance(value, pa.Array):
                return value
            encoded = (
                value.to_numpy().tobytes()
                if value_kind == "offsets"
                else json.dumps(value, ensure_ascii=False).encode()
            )
            if value_kind != "offsets" and len(encoded) <= INLINE_LIMIT:
                return value
            path = f"payloads/part-{len(self.files):08d}.parquet"
            row = len(self.rows)
            owner = None
            length = len(value)
            digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
            self.rows.append({value_kind: value.to_pylist() if value_kind == "offsets" else value})
            self.bytes += len(encoded)
        self.locators.append(
            {
                "object_kind": kind,
                "object_id": identity,
                "field": field,
                "position": position,
                "owner_key": owner,
                "value_path": path,
                "row_index": row,
                "value_kind": value_kind,
                "n_values": length,
                "value_digest": digest,
            }
        )
        if self.bytes >= _PACK_BYTES:
            self.flush()
        return [] if value_kind == "texts" else "" if value_kind == "text" else None

    def flush(self) -> None:
        """Write one packed group and release its decoded values."""
        if not self.rows:
            return
        path = f"payloads/part-{len(self.files):08d}.parquet"
        (self.root / path).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(self.rows, schema=_SCHEMA), self.root / path, compression="zstd")
        self.files.append(path)
        self.rows.clear()
        self.bytes = 0

    def finish(self, connection: duckdb.DuckDBPyConnection) -> None:
        """Flush payload files and store their explicit catalog locators."""
        self.flush()
        if self.locators:
            table = pa.Table.from_pylist(self.locators, schema=TABLES["external_values"].arrow_schema)
            connection.register("_external_values", table)
            try:
                connection.execute("INSERT INTO external_values SELECT * FROM _external_values")
            finally:
                connection.unregister("_external_values")
