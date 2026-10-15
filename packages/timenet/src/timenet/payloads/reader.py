"""Resolve payloads through immutable physical locators without catalog lookups."""

from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.errors import TimeFFormatError
from timenet.format.checksums import stream_checksum
from timenet.format.constants import MANIFEST_FILE
from timenet.payloads.offsets import validate_offsets
from timenet.registry.version import DatasetVersion
from timenet.types.payload_references import PayloadReference, PayloadSource
from timenet.types.payloads import ArrayPayload, Payload, PayloadDescriptor
from timenet.types.references import DatasetRef
from timenet.values_backends.reader import BaseValuesReader, make_values_reader


_OFFSET_CACHE_BYTES = 64 * 2**20
_OFFSET_CACHE_ENTRIES = 64


@dataclass(frozen=True)
class _ArrayEncoding:
    dtype: str
    value_shape: tuple[int, ...]
    dimension_names: tuple[str, ...]
    nullable: bool
    categories: tuple[str, ...]


class PayloadResolver:
    """Share physical file caches across every payload in an opened release."""

    def __init__(self, version: DatasetVersion) -> None:
        self.version = version
        self._source: PayloadSource | None = None
        self._text_cache: OrderedDict[tuple[str, str, int], pa.Array] = OrderedDict()
        self._text_bytes = 0
        self._offset_cache: OrderedDict[str, pa.Array] = OrderedDict()
        self._offset_bytes = 0
        self._readers: dict[tuple[str, str], BaseValuesReader] = {}

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_readers"] = {}
        state["_offset_cache"] = OrderedDict()
        state["_offset_bytes"] = 0
        state["_text_cache"] = OrderedDict()
        state["_text_bytes"] = 0
        return state

    def source(self) -> PayloadSource:
        """Return the actual opened release and manifest checksum.

        Returns:
            The immutable source identity.
        """
        if self._source is None:
            with self.version.filesystem.open_input_file(self.version.path(MANIFEST_FILE)) as handle:
                checksum = stream_checksum(handle)
            self._source = PayloadSource(
                dataset=DatasetRef(
                    dataset_id=self.version.manifest.dataset_id, version=self.version.manifest.metadata.dataset_version
                ),
                manifest_checksum=checksum,
            )
        return self._source

    def resolve(self, reference: PayloadReference) -> DatasetVersion:
        """Resolve the reference's physical root without opening any control database.

        Returns:
            The physical source handle.

        Raises:
            TimeFFormatError: If the source is unavailable or a locator is absent from its manifest.
        """
        if reference.source is not None and reference.source != self.source():
            raise TimeFFormatError(f"no payload source resolver is attached for {reference.source.dataset}")
        owned = {part.path for part in self.version.manifest.files.all_files()}
        prefix = f"{reference.prefix}/" if reference.prefix else ""
        for chunk in reference.chunks:
            path = prefix + (f"{chunk.path}/zarr.json" if reference.codec == "zarr" else chunk.path)
            if path not in owned:
                raise TimeFFormatError(f"payload locator {path!r} is absent from its owner's manifest")
        return replace(self.version, root=self.version.root + (f"/{reference.prefix}" if reference.prefix else ""))

    def payload(self, descriptor: PayloadDescriptor, reference: PayloadReference) -> Payload[Any]:
        """Create an explicit payload handle and bind local ownership to this release.

        Returns:
            A lazy, directly located payload.

        Raises:
            TimeFFormatError: If an array's descriptor and chunk lengths disagree.
        """
        bound = reference if reference.source is not None else reference.model_copy(update={"source": self.source()})
        loader = StoredPayloadLoader(self, descriptor, bound)
        if descriptor.kind == "array":
            if sum(chunk.n_values for chunk in bound.chunks) != descriptor.length:
                raise TimeFFormatError("array payload chunks do not match its declared length")
            loader._spec()
            return ArrayPayload(descriptor, _loader=loader, _read_range=loader.read_steps, reference=bound)
        return Payload(descriptor, _loader=loader, reference=bound)

    def reader(self, version: DatasetVersion, codec: str) -> BaseValuesReader:
        """Return the source-specific cached array backend.

        Returns:
            The array decoder.
        """
        key = (version.root, codec)
        if key not in self._readers:
            self._readers[key] = make_values_reader(codec)
        return self._readers[key]

    def text(self, reference: PayloadReference, descriptor: PayloadDescriptor) -> Any:
        """Decode a packed text cell without consulting its catalog.

        Returns:
            The native text value.

        Raises:
            TimeFFormatError: If the locator or content disagrees with its descriptor.
        """
        if reference.codec != "text" or len(reference.chunks) != 1:
            raise TimeFFormatError("text payload requires one text-codec locator")
        version = self.resolve(reference)
        chunk = reference.chunks[0]
        key = (version.root, chunk.path, chunk.major)
        if key in self._text_cache:
            self._text_cache.move_to_end(key)
            values = self._text_cache[key]
        else:
            with version.filesystem.open_input_file(version.path(chunk.path)) as handle:
                values = (
                    pq.ParquetFile(handle).read_row_group(chunk.major, columns=["values"]).column(0).combine_chunks()
                )
            if values.nbytes <= _OFFSET_CACHE_BYTES:
                self._text_cache[key] = values
                self._text_bytes += values.nbytes
                while self._text_bytes > _OFFSET_CACHE_BYTES:
                    _, evicted = self._text_cache.popitem(last=False)
                    self._text_bytes -= evicted.nbytes
        if chunk.minor is None or chunk.minor >= len(values):
            raise TimeFFormatError("text payload has an invalid cell index")
        value = values[chunk.minor].as_py()
        actual = Payload.from_value(value).descriptor
        if actual.kind != descriptor.kind or any(
            getattr(descriptor, field) is not None and getattr(actual, field) != getattr(descriptor, field)
            for field in ("length", "encoded_size")
        ):
            raise TimeFFormatError("stored text does not match its payload descriptor")
        return value

    def close(self) -> None:
        """Close cached artifacts while retaining reusable source configuration."""
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()
        self._offset_cache.clear()
        self._offset_bytes = 0
        self._text_cache.clear()
        self._text_bytes = 0


@dataclass(frozen=True)
class StoredPayloadLoader:
    """A picklable payload reader with no dependency on a logical catalog."""

    resolver: PayloadResolver
    descriptor: PayloadDescriptor
    reference: PayloadReference

    def _spec(self) -> _ArrayEncoding:
        descriptor = self.descriptor
        if descriptor.dtype is None:
            raise TimeFFormatError("array payload has no physical dtype")
        return _ArrayEncoding(
            descriptor.dtype, descriptor.shape, descriptor.dimension_names, descriptor.nullable, descriptor.categories
        )

    def __call__(self) -> Any:
        """Read the payload through its physical array codec.

        Returns:
            The decoded content.

        Raises:
            TimeFFormatError: If the stored length or timestamp values are inconsistent.
        """
        if self.descriptor.kind != "array":
            return self.resolver.text(self.reference, self.descriptor)
        key = self.reference.model_dump_json()
        offsets = self.descriptor.first_us is not None
        if offsets and key in self.resolver._offset_cache:
            self.resolver._offset_cache.move_to_end(key)
            values = self.resolver._offset_cache[key]
            validate_offsets(values, self.descriptor)
            return values
        version = self.resolver.resolve(self.reference)
        rows = [chunk.as_backend_row() for chunk in self.reference.chunks]
        values = self.resolver.reader(version, self.reference.codec).load(version, rows, self._spec())
        if len(values) != self.descriptor.length:
            raise TimeFFormatError("array payload has a different length from its descriptor")
        if offsets:
            validate_offsets(values, self.descriptor)
            self.resolver._offset_cache[key] = values
            self.resolver._offset_bytes += values.nbytes
            while len(self.resolver._offset_cache) > 1 and (
                len(self.resolver._offset_cache) > _OFFSET_CACHE_ENTRIES
                or self.resolver._offset_bytes > _OFFSET_CACHE_BYTES
            ):
                _, evicted = self.resolver._offset_cache.popitem(last=False)
                self.resolver._offset_bytes -= evicted.nbytes
        return values

    def read_steps(self, start: int, stop: int) -> pa.Array:
        """Read a bounded array range from its actual physical owner.

        Returns:
            The selected values.
        """
        version = self.resolver.resolve(self.reference)
        rows = [chunk.as_backend_row() for chunk in self.reference.chunks]
        return self.resolver.reader(version, self.reference.codec).load_range(version, rows, start, stop, self._spec())
