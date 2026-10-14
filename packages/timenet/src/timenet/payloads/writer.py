"""Pack owned bulk content into bounded Parquet row groups."""

from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.errors import TimeFValidationError
from timenet.types.payload_references import PayloadChunk, PayloadReference
from timenet.types.payloads import Payload


_ROW_GROUP_BYTES = 16 * 2**20
_ARRAY_CHUNK_VALUES = 32_768


class PayloadStore:
    """Own the payload artifacts generated alongside one control database."""

    def __init__(self, root: Path, *, inline_bytes: int = 4096) -> None:
        self.root = root
        self.inline_bytes = inline_bytes
        self._text_refs: WeakKeyDictionary[Payload[Any], PayloadReference] = WeakKeyDictionary()
        self.files: list[str] = []
        self._writers: dict[str, pq.ParquetWriter] = {}
        self._buffers: dict[str, list[pa.Array]] = {}
        self._bytes: dict[str, int] = {}
        self._groups: dict[str, int] = {}

    def text(self, payload: Payload[Any]) -> tuple[Any, PayloadReference | None]:
        """Keep small content inline and pack larger text into shared artifacts.

        Returns:
            The inline value or a physical reference.

        Raises:
            TimeFValidationError: If a producer disagrees with its descriptor.
        """
        if payload.reference is not None:
            if payload.reference.source is None:
                raise TimeFValidationError("reused payload references must name a pinned owner")
            return None, payload.reference
        if payload in self._text_refs:
            return None, self._text_refs[payload]
        value = payload.read()
        actual = Payload.from_value(value).descriptor
        declared = payload.descriptor
        if actual.kind != declared.kind or any(
            getattr(declared, field) is not None and getattr(actual, field) != getattr(declared, field)
            for field in ("length", "encoded_size")
        ):
            raise TimeFValidationError("produced text does not match its payload descriptor")
        if actual.kind not in {"str", "list"} or (
            payload.storage == "auto" and (actual.encoded_size or 0) <= self.inline_bytes
        ):
            return value, None
        kind = pa.string() if actual.kind == "str" else pa.list_(pa.string())
        chunk = self._cell(actual.kind, pa.array([value], type=kind), 1)
        reference = PayloadReference(codec="text", chunks=(chunk,))
        self._text_refs[payload] = reference
        return None, reference

    def _cell(self, name: str, value: pa.Array, n_values: int) -> PayloadChunk:
        path = f"payloads/{name}.parquet"
        if name not in self._writers:
            target = self.root / path
            if target.exists():
                raise TimeFValidationError(f"payload artifact already exists at {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            self._writers[name] = pq.ParquetWriter(target, pa.schema([("values", value.type)]), compression="zstd")
            self.files.append(path)
            self._buffers[name] = []
            self._bytes[name] = 0
            self._groups[name] = 0
        chunk = PayloadChunk(path=path, major=self._groups[name], minor=len(self._buffers[name]), n_values=n_values)
        self._buffers[name].append(value)
        self._bytes[name] += value.nbytes
        if self._bytes[name] >= _ROW_GROUP_BYTES:
            self._flush(name)
        return chunk

    def offsets(self, offsets: pa.Array) -> PayloadReference:
        """Store int64 timestamp chunks without scalar Python conversion.

        Returns:
            The local physical reference.

        Raises:
            TimeFValidationError: If the array is empty, nullable, or not int64.
        """
        if offsets.type != pa.int64() or offsets.null_count or not len(offsets):
            raise TimeFValidationError("timestamp payloads require non-empty, non-null int64 arrays")
        chunks = []
        for start in range(0, len(offsets), _ARRAY_CHUNK_VALUES):
            values = offsets.slice(start, _ARRAY_CHUNK_VALUES)
            cell = pa.ListArray.from_arrays(pa.array([0, len(values)], type=pa.int32()), values)
            chunks.append(self._cell("offsets", cell, len(values)))
        return PayloadReference(codec="parquet", chunks=tuple(chunks))

    def _flush(self, name: str) -> None:
        values = self._buffers[name]
        if not values:
            return
        table = pa.Table.from_arrays([pa.concat_arrays(values)], names=["values"])
        self._writers[name].write_table(table, row_group_size=len(table))
        self._groups[name] += 1
        self._buffers[name] = []
        self._bytes[name] = 0

    def close(self) -> None:
        """Flush and close every generated artifact."""
        for name, writer in self._writers.items():
            self._flush(name)
            writer.close()
        self._writers.clear()

    def abort(self) -> None:
        """Remove only artifacts created by this store."""
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()
        for path in self.files:
            (self.root / path).unlink(missing_ok=True)
