"""Resolve payloads through immutable physical locators without catalog lookups."""

from dataclasses import dataclass, replace
from typing import Any, cast

import pyarrow as pa

from timenet.errors import TimeFFormatError
from timenet.format.checksums import stream_checksum
from timenet.format.constants import MANIFEST_FILE
from timenet.registry.version import DatasetVersion
from timenet.types.payload_references import PayloadReference, PayloadSource
from timenet.types.payloads import ArrayPayload, Payload, PayloadDescriptor
from timenet.types.references import DatasetRef
from timenet.types.specs import TimeSeriesSpec
from timenet.types.wire import ValueDtype
from timenet.values_backends.reader import BaseValuesReader, make_values_reader


class PayloadResolver:
    """Share physical file caches across every payload in an opened release."""

    def __init__(self, version: DatasetVersion) -> None:
        self.version = version
        self._source: PayloadSource | None = None
        self._readers: dict[tuple[str, str], BaseValuesReader] = {}

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_readers"] = {}
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

    def close(self) -> None:
        """Close cached artifacts while retaining reusable source configuration."""
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()


@dataclass(frozen=True)
class StoredPayloadLoader:
    """A picklable payload reader with no dependency on a logical catalog."""

    resolver: PayloadResolver
    descriptor: PayloadDescriptor
    reference: PayloadReference

    def _spec(self) -> TimeSeriesSpec:
        descriptor = self.descriptor
        return TimeSeriesSpec(
            spec_type="payload",
            name="Payload",
            unit_value=None,
            dtype=cast("ValueDtype", descriptor.dtype),
            value_shape=descriptor.shape,
            dimension_names=descriptor.dimension_names,
            nullable=descriptor.nullable,
            categories=descriptor.categories,
        )

    def __call__(self) -> Any:
        """Read the payload through its physical array codec.

        Returns:
            The decoded content.
        """
        version = self.resolver.resolve(self.reference)
        rows = [chunk.as_backend_row() for chunk in self.reference.chunks]
        return self.resolver.reader(version, self.reference.codec).load(version, rows, self._spec())

    def read_steps(self, start: int, stop: int) -> pa.Array:
        """Read a bounded array range from its actual physical owner.

        Returns:
            The selected values.
        """
        version = self.resolver.resolve(self.reference)
        rows = [chunk.as_backend_row() for chunk in self.reference.chunks]
        return self.resolver.reader(version, self.reference.codec).load_range(version, rows, start, stop, self._spec())
