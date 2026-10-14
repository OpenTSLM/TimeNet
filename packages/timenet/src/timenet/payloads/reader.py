"""Resolve payloads through immutable physical locators without catalog lookups."""

from collections import OrderedDict
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from timenet.errors import TimeFFormatError
from timenet.format.checksums import CHECKSUM_PREFIX, stream_checksum
from timenet.format.constants import MANIFEST_FILE
from timenet.manifest import Manifest
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
        self._versions: dict[PayloadSource, DatasetVersion] = {}
        self._owners: dict[int, tuple[PayloadSource, DatasetVersion]] = {}
        self._files: dict[str, frozenset[str]] = {}
        self._roots: dict[tuple[str, str], DatasetVersion] = {}
        self._text_cache: OrderedDict[tuple[str, str, int], pa.Array] = OrderedDict()
        self._text_bytes = 0
        self._offset_cache: OrderedDict[tuple[PayloadDescriptor, str], pa.Array] = OrderedDict()
        self._offset_bytes = 0
        self._readers: dict[tuple[str, str], BaseValuesReader] = {}

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_readers"] = {}
        state["_owners"] = {}
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

    def owner(self, source: PayloadSource | None) -> DatasetVersion:
        """Open an exact pinned physical owner without consulting its catalog.

        Returns:
            The immutable owner handle.

        Raises:
            TimeFFormatError: If the owner is missing, undeclared, or has a different manifest pin.
        """
        if source is None:
            return self.version
        cached = self._owners.get(id(source))
        if cached is not None and cached[0] is source:
            return cached[1]
        version = self.version
        if source is not None and source != self.source():
            if source not in self.version.manifest.payload_sources:
                raise TimeFFormatError(f"undeclared payload source {source.dataset}")
            if source not in self._versions:
                if self.version.payload_root is None:
                    raise TimeFFormatError(f"no payload root configured for {source.dataset}")
                root = f"{self.version.payload_root}/{source.dataset.dataset_id}/{source.dataset.version}"
                try:
                    with self.version.filesystem.open_input_file(f"{root}/{MANIFEST_FILE}") as handle:
                        content = handle.read()
                except OSError as exc:
                    raise TimeFFormatError(f"cannot open payload owner {source.dataset}: {exc}") from exc
                manifest = Manifest.from_artifact(content)
                if CHECKSUM_PREFIX + sha256(content).hexdigest() != source.manifest_checksum:
                    raise TimeFFormatError(f"payload owner manifest checksum mismatch for {source.dataset}")
                if (
                    manifest.dataset_id != source.dataset.dataset_id
                    or manifest.metadata.dataset_version != source.dataset.version
                ):
                    raise TimeFFormatError(f"payload owner identity mismatch for {source.dataset}")
                self._versions[source] = DatasetVersion(
                    manifest, self.version.filesystem, root, self.version.payload_root
                )
            version = self._versions[source]
        self._owners[id(source)] = (source, version)
        return version

    def resolve(self, reference: PayloadReference) -> DatasetVersion:
        """Resolve the reference's physical root without opening any control database.

        Returns:
            The physical source handle.

        Raises:
            TimeFFormatError: If the source is unavailable or a locator is absent from its manifest.
        """
        version = self.owner(reference.source)
        owned = self._files.get(version.root)
        if owned is None:
            owned = frozenset(part.path for part in version.manifest.files.all_files())
            self._files[version.root] = owned
        prefix = f"{reference.prefix}/" if reference.prefix else ""
        for chunk in reference.chunks:
            path = prefix + (f"{chunk.path}/zarr.json" if reference.codec == "zarr" else chunk.path)
            if path not in owned:
                raise TimeFFormatError(f"payload locator {path!r} is absent from its owner's manifest")
        if not reference.prefix:
            return version
        key = (version.root, reference.prefix)
        if key not in self._roots:
            self._roots[key] = replace(version, root=f"{version.root}/{reference.prefix}")
        return self._roots[key]

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
            return ArrayPayload(
                descriptor, _loader=loader, _read_range=loader.read_steps, reference=bound, _storage=self
            )
        return Payload(descriptor, _loader=loader, reference=bound, _storage=self)

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
        offsets = self.descriptor.first_us is not None
        key = (self.descriptor, self.reference.model_dump_json()) if offsets else None
        if key is not None and key in self.resolver._offset_cache:
            self.resolver._offset_cache.move_to_end(key)
            return self.resolver._offset_cache[key]
        version = self.resolver.resolve(self.reference)
        rows = [chunk.as_backend_row() for chunk in self.reference.chunks]
        values = self.resolver.reader(version, self.reference.codec).load(version, rows, self._spec())
        if len(values) != self.descriptor.length:
            raise TimeFFormatError("array payload has a different length from its descriptor")
        if key is not None:
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
