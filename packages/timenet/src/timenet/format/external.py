"""Packed external values with lightweight handles allocated only for external fields."""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from timenet.errors import TimeFFormatError


if TYPE_CHECKING:
    from timenet.format.storage import StorageSources
    from timenet.registry.version import DatasetVersion


INLINE_LIMIT = 4096
_PACK_BYTES = 4 * 1024 * 1024
_CACHE_GROUPS = 8
_SCHEMA = pa.schema([("text", pa.string()), ("texts", pa.list_(pa.string())), ("offsets", pa.list_(pa.int64()))])


@dataclass(frozen=True)
class ValueHandle:
    """A locator for one external value; constructing it performs no payload I/O."""

    store: "ExternalReader"
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
        """Return the direct physical owner without reading payload data."""
        return self.store.version.owner(self.owner)


class ExternalReader:
    """Share a bounded cache of packed value groups between field handles."""

    def __init__(self, version: "DatasetVersion") -> None:
        self.version = version
        self.cache: OrderedDict[tuple[str, str], pa.Table] = OrderedDict()

    def __getstate__(self) -> dict:
        return {"version": self.version, "cache": OrderedDict()}

    def read(self, handle: ValueHandle) -> Any:
        """Read a packed value from its physical file.

        Returns:
            Native text, string list, or an Arrow timestamp array.

        Raises:
            TimeFFormatError: If the locator or stored value is invalid.
        """
        version = handle.physical_version()
        key = (version.root, handle.path)
        table = self.cache.get(key)
        if table is None:
            try:
                table = pq.read_table(version.path(handle.path), filesystem=version.filesystem)
            except (OSError, pa.ArrowException) as exc:
                raise TimeFFormatError(f"cannot read external value {key}: {exc}") from exc
            self.cache[key] = table
            while len(self.cache) > _CACHE_GROUPS:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        if not 0 <= handle.row < len(table) or handle.kind not in table.column_names:
            raise TimeFFormatError(f"invalid external locator {handle.path}:{handle.row}")
        value = table[handle.kind][handle.row]
        if not value.is_valid:
            raise TimeFFormatError(f"missing external {handle.kind} at {handle.path}:{handle.row}")
        return value.values if handle.kind == "offsets" else value.as_py()

    def locators(self, connection: duckdb.DuckDBPyConnection) -> dict[tuple[str, str, str, int], ValueHandle]:
        """Read external locators without opening any external files.

        Returns:
            Handles keyed by entity kind, public ID, field, and position.
        """
        return {
            (kind, identity, field, position): ValueHandle(self, owner, path, row, value_kind, length, digest)
            for kind, identity, field, position, owner, path, row, value_kind, length, digest in connection.execute(
                "SELECT * FROM external_values"
            ).fetchall()
        }


class ExternalWriter:
    """Pack new values and preserve locators for unchanged imported values."""

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
            from timenet.format.control_schema import TABLES  # noqa: PLC0415

            table = pa.Table.from_pylist(self.locators, schema=TABLES["external_values"].arrow_schema)
            connection.register("_external_values", table)
            try:
                connection.execute("INSERT INTO external_values SELECT * FROM _external_values")
            finally:
                connection.unregister("_external_values")
