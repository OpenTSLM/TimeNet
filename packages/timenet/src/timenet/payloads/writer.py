"""Pack owned bulk content into bounded Parquet row groups."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.errors import TimeFValidationError
from timenet.types.payload_references import PayloadChunk, PayloadReference


_ROW_GROUP_BYTES = 16 * 2**20
_ARRAY_CHUNK_VALUES = 32_768


class PayloadStore:
    """Own the payload artifacts generated alongside one control database."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: list[str] = []
        self._writers: dict[str, pq.ParquetWriter] = {}
        self._buffers: dict[str, list[pa.Array]] = {}
        self._bytes: dict[str, int] = {}
        self._groups: dict[str, int] = {}

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
