"""Store shared release identities and payload coordinates in typed catalog rows."""

from collections import defaultdict
from dataclasses import asdict
import json
from typing import Any

import duckdb

from timenet.errors import TimeFFormatError
from timenet.format.batches import TableBatch
from timenet.format.constants import check_relative_path
from timenet.format.control_schema import TABLES
from timenet.types.payload_references import PayloadChunk, PayloadReference, PayloadSource
from timenet.types.payloads import PayloadDescriptor
from timenet.types.references import DatasetRef, ObjectRef


class ReferenceCatalogWriter:
    """Normalize payload contracts, release identities, files, and lineage while writing."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self.batches = {
            name: TableBatch(connection, TABLES[name])
            for name in (
                "dataset_refs",
                "payload_stores",
                "payload_files",
                "payload_descriptors",
                "payloads",
                "payload_chunks",
                "origins",
            )
        }
        self.datasets: dict[tuple[str, str], int] = {}
        self.stores: dict[tuple[Any, ...], int] = {}
        self.files: dict[tuple[int, str], int] = {}
        self.descriptors: dict[PayloadDescriptor, int] = {}
        self.count = 0

    def dataset(self, reference: DatasetRef) -> int:
        """Return the key of a release, inserting it once."""
        identity = (reference.dataset_id, str(reference.version))
        if identity not in self.datasets:
            key = len(self.datasets) + 1
            self.datasets[identity] = key
            self.batches["dataset_refs"].add({"dataset_key": key, "dataset_id": identity[0], "version": identity[1]})
        return self.datasets[identity]

    def payload(  # noqa: PLR0913, PLR0917 - one catalog field and its descriptor/location
        self,
        kind: str,
        identity: str,
        field: str,
        position: int,
        descriptor: PayloadDescriptor,
        reference: PayloadReference,
    ) -> None:
        """Write one logical payload and its ordered physical coordinates."""
        source = reference.source
        dataset_key = None if source is None else self.dataset(source.dataset)
        checksum = None if source is None else source.manifest_checksum
        store_identity = (dataset_key, checksum, reference.codec, reference.prefix)
        if store_identity not in self.stores:
            key = len(self.stores) + 1
            self.stores[store_identity] = key
            self.batches["payload_stores"].add(
                {
                    "store_key": key,
                    "dataset_key": dataset_key,
                    "manifest_checksum": checksum,
                    "codec": reference.codec,
                    "prefix": reference.prefix,
                }
            )
        store_key = self.stores[store_identity]
        if descriptor not in self.descriptors:
            key = len(self.descriptors) + 1
            self.descriptors[descriptor] = key
            self.batches["payload_descriptors"].add(
                {"descriptor_key": key, "descriptor": json.dumps(asdict(descriptor), allow_nan=False)}
            )
        self.count += 1
        self.batches["payloads"].add(
            {
                "payload_key": self.count,
                "object_type": kind,
                "object_id": identity,
                "field": field,
                "position": position,
                "descriptor_key": self.descriptors[descriptor],
                "store_key": store_key,
            }
        )
        for index, chunk in enumerate(reference.chunks):
            file_identity = (store_key, chunk.path)
            if file_identity not in self.files:
                key = len(self.files) + 1
                self.files[file_identity] = key
                self.batches["payload_files"].add({"file_key": key, "store_key": store_key, "path": chunk.path})
            self.batches["payload_chunks"].add(
                {
                    "payload_key": self.count,
                    "position": index,
                    "file_key": self.files[file_identity],
                    "major": chunk.major,
                    "minor": chunk.minor,
                    "n_values": chunk.n_values,
                }
            )

    def origin(self, kind: str, identity: str, reference: ObjectRef) -> None:
        """Write lineage using the shared release key."""
        self.batches["origins"].add(
            {
                "object_type": kind,
                "object_id": identity,
                "dataset_key": self.dataset(reference.dataset),
                "origin_kind": reference.kind,
                "origin_id": reference.id,
            }
        )

    def flush(self) -> None:
        """Write all remaining reference rows."""
        for batch in self.batches.values():
            batch.flush()


class ReferenceCatalogReader:
    """Decode shared metadata once and reconstruct bounded payload selections."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self.connection = connection
        self.datasets = {
            key: DatasetRef.model_validate({"dataset_id": identity, "version": version})
            for key, identity, version in connection.execute("SELECT * FROM dataset_refs").fetchall()
        }
        self.stores: dict[int, tuple[PayloadSource | None, str, str]] = {}
        for key, dataset_key, checksum, codec, prefix in connection.execute("SELECT * FROM payload_stores").fetchall():
            if codec not in {"parquet", "zarr", "text"}:
                raise TimeFFormatError(f"payload store {key} has unsupported codec {codec!r}")
            check_relative_path("payload prefix", prefix)
            source = (
                None
                if dataset_key is None
                else PayloadSource(dataset=self.datasets[dataset_key], manifest_checksum=checksum)
            )
            if (dataset_key is None) != (checksum is None):
                raise TimeFFormatError(f"payload store {key} has incomplete ownership")
            self.stores[key] = (source, codec, prefix)
        self.files: dict[int, tuple[int, str]] = {}
        for key, store_key, path in connection.execute("SELECT * FROM payload_files").fetchall():
            check_relative_path("payload file", path)
            if store_key not in self.stores:
                raise TimeFFormatError(f"payload file {key} has no store")
            self.files[key] = (store_key, path)
        self.descriptors: dict[int, PayloadDescriptor] = {}

    def payloads(self, kind: str, identities: list[str], field: str | None = None) -> list[tuple]:
        """Return selected object fields, descriptors, and typed physical references.

        Returns:
            Rows of object ID, field, position, descriptor, and reference.

        Raises:
            TimeFFormatError: If a payload has invalid coordinates or missing shared metadata.
        """
        clause = "p.object_type=? AND p.object_id IN (SELECT unnest(?))"
        params: list[Any] = [kind, identities]
        if field is not None:
            clause += " AND p.field=? AND p.position=0"
            params.append(field)
        rows = self.connection.execute(
            f"SELECT p.object_id, p.field, p.position, p.payload_key, p.descriptor_key, p.store_key "  # noqa: S608 - only fixed clauses follow
            f"FROM payloads p WHERE {clause}",
            params,
        ).fetchall()
        missing = list({row[4] for row in rows} - self.descriptors.keys())
        for key, encoded in self.connection.execute(
            "SELECT descriptor_key, descriptor FROM payload_descriptors WHERE descriptor_key IN (SELECT unnest(?))",
            [missing],
        ).fetchall():
            data = json.loads(encoded)
            for name in ("shape", "categories", "dimension_names"):
                data[name] = tuple(data.get(name, ()))
            self.descriptors[key] = PayloadDescriptor(**data)
        chunks: dict[int, list[PayloadChunk]] = defaultdict(list)
        for payload_key, file_key, major, minor, n_values, expected_store in self.connection.execute(
            f"SELECT c.payload_key, c.file_key, c.major, c.minor, c.n_values, p.store_key FROM payload_chunks c "  # noqa: S608 - only fixed clauses follow
            f"JOIN payloads p USING (payload_key) WHERE {clause} ORDER BY c.position",
            params,
        ).fetchall():
            if file_key not in self.files:
                raise TimeFFormatError(f"payload {payload_key} references missing file {file_key}")
            if self.files[file_key][0] != expected_store:
                raise TimeFFormatError(f"payload {payload_key} has a chunk in another store")
            chunks[payload_key].append(
                PayloadChunk(path=self.files[file_key][1], major=major, minor=minor, n_values=n_values)
            )
        result = []
        for identity, field_name, position, key, descriptor_key, store_key in rows:
            if descriptor_key not in self.descriptors or store_key not in self.stores or not chunks[key]:
                raise TimeFFormatError(f"payload {key} has missing metadata or chunks")
            source, codec, prefix = self.stores[store_key]
            # The store and its chunks are already validated. Do not copy and revalidate
            # the same release identity for each Signal.
            reference = PayloadReference.model_construct(
                source=source, codec=codec, prefix=prefix, chunks=tuple(chunks[key])
            )
            result.append((identity, field_name, position, self.descriptors[descriptor_key], reference))
        return result

    def origins(self, identities: list[str]) -> dict[tuple[str, str], ObjectRef]:
        """Return lineage with shared validated release identities.

        Returns:
            References indexed by their local object kind and ID.

        Raises:
            TimeFFormatError: If lineage names an invalid object or absent release.
        """
        result = {}
        for kind, identity, key, origin_kind, origin_id in self.connection.execute(
            "SELECT * FROM origins WHERE object_id IN (SELECT unnest(?))", [identities]
        ).fetchall():
            if (
                key not in self.datasets
                or not origin_id
                or origin_kind
                not in {"record", "source", "signal", "axis", "annotation_content", "annotation_occurrence", "task"}
            ):
                raise TimeFFormatError(f"invalid lineage for {kind} {identity}")
            result[kind, identity] = ObjectRef.model_construct(
                dataset=self.datasets[key], kind=origin_kind, id=origin_id
            )
        return result
