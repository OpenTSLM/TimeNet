"""The :class:`Manifest`: the compiled ``manifest.json`` and its JSON codec.

The manifest is pure data with no file I/O; the writer and reader own reading/writing the file. Its
``schema`` block is a faithful serialization of :class:`~timenet.types.DatasetSchema` (flat descriptors),
so there is no separate set of "entry" types to keep in sync.
"""

from dataclasses import dataclass, field
import json
from typing import Any, ClassVar

from timenet.errors import InvalidManifestError
from timenet.format.constants import PART_STAT_KEYS
from timenet.manifest.counts import ManifestCounts
from timenet.manifest.files import ManifestFiles
from timenet.manifest.part_stats import PartStat
from timenet.types import (
    TASKS,
    AnnotationDescriptor,
    AnnotationType,
    DatasetMetadata,
    DatasetSchema,
    DataSource,
    Task,
    TaskType,
    TimeSeriesSpec,
    ureg,
)
from timenet.values_backends import SUPPORTED_VALUES_BACKENDS, ValuesBackend


@dataclass(frozen=True)
class Manifest:
    """The single source of truth a consumer reads to interpret a dataset version.

    Raises:
        InvalidManifestError: If ``timef_format_version`` is not a supported version.
    """

    SUPPORTED_FORMAT_VERSIONS: ClassVar[frozenset[int]] = frozenset({1})

    dataset_id: str
    """Denormalized copy of ``metadata.dataset_id``, readable without parsing metadata."""
    metadata: DatasetMetadata
    """Descriptive identity of the dataset (name, version, license, domains, tags)."""
    files: ManifestFiles
    """Relative paths to every data artifact, grouped by kind."""
    schema: DatasetSchema = field(default_factory=DatasetSchema)
    """Structural schema: time-series specs, annotations, and tasks."""
    counts: ManifestCounts = field(default_factory=ManifestCounts)
    """Row and entity counts recorded for quick inspection."""
    checksums: dict[str, str] = field(default_factory=dict)
    """Per-file checksums keyed by relative path, each ``sha256:`` prefixed."""
    id_encoding: dict[str, str] = field(default_factory=dict)
    """Logical id -> ``"uuid16"`` for ids stored as ``binary(16)``; absent entries are strings."""
    part_stats: dict[str, tuple[PartStat, ...]] = field(default_factory=dict)
    """Per-part skip metadata for samples, annotations, and the index; keyed by table name."""
    values_backend: str = ValuesBackend.PARQUET
    """Storage backend for the time-series values plane."""
    derived_from: dict[str, str] | None = None
    """Copy-on-write lineage (base version + operation), or ``None`` for a freshly built version."""
    timef_format_version: int = 1
    """TimeF manifest format version; must be in ``SUPPORTED_FORMAT_VERSIONS``."""

    def __post_init__(self) -> None:
        """Validate the format version, values backend, and denormalized ``dataset_id``.

        ``dataset_id`` is a top-level copy of ``metadata.dataset_id`` so a consumer can read the id
        without parsing the metadata block; the two must agree.

        Raises:
            InvalidManifestError: If ``timef_format_version`` is unsupported, or ``dataset_id`` does
                not match ``metadata.dataset_id``.
        """
        if self.timef_format_version not in self.SUPPORTED_FORMAT_VERSIONS:
            raise InvalidManifestError(
                f"unsupported timef_format_version {self.timef_format_version!r}; "
                f"supported: {sorted(self.SUPPORTED_FORMAT_VERSIONS)}"
            )
        if self.values_backend not in SUPPORTED_VALUES_BACKENDS:
            raise InvalidManifestError(
                f"unsupported values_backend {self.values_backend!r}; "
                f"supported: {', '.join(sorted(SUPPORTED_VALUES_BACKENDS))}"
            )
        if self.dataset_id != self.metadata.dataset_id:
            raise InvalidManifestError(
                f"manifest dataset_id {self.dataset_id!r} does not match "
                f"metadata.dataset_id {self.metadata.dataset_id!r}"
            )

    # ---- serialization -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest to a JSON-compatible dict with all keys present.

        Returns:
            The canonical dict form.
        """
        return {
            "timef_format_version": self.timef_format_version,
            "dataset_id": self.dataset_id,
            "metadata": _metadata_to_dict(self.metadata),
            "schema": _schema_to_dict(self.schema),
            "counts": _counts_to_dict(self.counts),
            "files": _files_to_dict(self.files),
            "checksums": dict(self.checksums),
            "id_encoding": dict(self.id_encoding),
            "part_stats": _part_stats_to_dict(self.part_stats),
            "values_backend": self.values_backend,
            "derived_from": dict(self.derived_from) if self.derived_from is not None else None,
        }

    def to_json(self) -> str:
        """Serialize the manifest to a pretty JSON string.

        Returns:
            The JSON text.
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Manifest":
        """Parse a manifest dict, tolerating missing optional blocks.

        Args:
            data: The manifest dict (e.g. from ``json.loads``).

        Returns:
            The parsed :class:`Manifest`.

        Raises:
            InvalidManifestError: If a required key is missing or a block is malformed.
        """
        for required in ("timef_format_version", "dataset_id", "metadata", "files"):
            if required not in data:
                raise InvalidManifestError(f"manifest missing required key {required!r}")
        checksums = _dict_block(data, "checksums")
        id_encoding = _dict_block(data, "id_encoding")
        derived_from = _optional_dict_block(data, "derived_from")
        files = _files_from_dict(data["files"])
        return cls(
            dataset_id=data["dataset_id"],
            metadata=_metadata_from_dict(data["metadata"]),
            files=files,
            schema=_schema_from_dict(data.get("schema", {})),
            counts=_counts_from_dict(data.get("counts", {})),
            checksums=checksums,
            id_encoding=id_encoding,
            part_stats=_part_stats_from_dict(data.get("part_stats", {}), files, id_encoding),
            values_backend=data.get("values_backend", ValuesBackend.PARQUET),
            derived_from=derived_from,
            timef_format_version=data["timef_format_version"],
        )

    @classmethod
    def from_json(cls, text: str) -> "Manifest":
        """Parse a manifest from a JSON string.

        Args:
            text: The JSON text.

        Returns:
            The parsed :class:`Manifest`.

        Raises:
            InvalidManifestError: If the text is not valid JSON or a block is malformed.
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InvalidManifestError(f"manifest is not valid JSON: {exc}") from exc
        return cls.from_dict(data)


def _dict_block(data: dict[str, Any], key: str) -> dict:
    """Coerce an optional manifest dict block to a ``dict``, naming it on failure.

    Args:
        data: The manifest dict.
        key: The block's key.

    Returns:
        The block as a ``dict`` (empty when absent).

    Raises:
        InvalidManifestError: If the block is present but not a mapping.
    """
    try:
        return dict(data.get(key, {}))
    except (ValueError, TypeError) as exc:
        raise InvalidManifestError(f"invalid manifest {key!r} block: {exc}") from exc


def _optional_dict_block(data: dict[str, Any], key: str) -> dict | None:
    """Coerce a nullable manifest dict block to a ``dict`` or ``None``, naming it on failure.

    Args:
        data: The manifest dict.
        key: The block's key.

    Returns:
        The block as a ``dict``, or ``None`` when the value is ``null``/absent.

    Raises:
        InvalidManifestError: If the block is present, non-null, and not a mapping.
    """
    value = data.get(key)
    if value is None:
        return None
    try:
        return dict(value)
    except (ValueError, TypeError) as exc:
        raise InvalidManifestError(f"invalid manifest {key!r} block: {exc}") from exc


def _str_tuple(value: Any, key: str) -> tuple[str, ...]:
    # tuple("abc") silently yields ("a", "b", "c"), so a bare string where a list is expected would be
    # accepted as corrupt data; require an actual list/tuple instead. The TypeError is caught by the
    # callers' except clauses and re-raised as InvalidManifestError.
    if not isinstance(value, list | tuple):
        raise TypeError(f"{key!r} must be a list, got {type(value).__name__}")
    return tuple(value)


def _metadata_to_dict(metadata: DatasetMetadata) -> dict[str, Any]:
    return {
        "dataset_id": metadata.dataset_id,
        "dataset_version": str(metadata.dataset_version),
        "name": metadata.name,
        "description": metadata.description,
        "license": str(metadata.license),
        "domains": [str(domain) for domain in metadata.domains],
        "tags": list(metadata.tags),
        "source_url": metadata.source_url,
        "yaml_schema_version": metadata.yaml_schema_version,
    }


def _metadata_from_dict(data: dict[str, Any]) -> DatasetMetadata:
    try:
        return DatasetMetadata.from_dict(data)
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise InvalidManifestError(f"invalid manifest 'metadata' block: {exc}") from exc


def _schema_to_dict(schema: DatasetSchema) -> dict[str, Any]:
    return {
        "time_series_specs": [
            {
                "spec_type": spec.spec_type,
                "name": spec.name,
                "unit_value": str(spec.unit_value),
                "data_source": (
                    {
                        "data_source_type": spec.data_source.data_source_type,
                        "name": spec.data_source.name,
                        "provider": spec.data_source.provider,
                    }
                    if spec.data_source is not None
                    else None
                ),
                "dtype": spec.dtype,
                "value_shape": list(spec.value_shape),
                "dimension_names": list(spec.dimension_names),
            }
            for spec in schema.time_series_specs
        ],
        "annotations": [
            {
                "key": ann.key,
                "annotation_type": str(ann.annotation_type),
                "value_type": ann.value_type,
                "unit": ann.unit,
                "description": ann.description,
            }
            for ann in schema.annotations
        ],
        "tasks": [{"task_type": str(task.task_type)} for task in schema.tasks],
    }


def _schema_from_dict(data: dict[str, Any]) -> DatasetSchema:
    try:
        specs = tuple(
            TimeSeriesSpec(
                spec_type=entry["spec_type"],
                name=entry["name"],
                unit_value=ureg.Unit(entry["unit_value"]),
                data_source=_data_source(entry.get("data_source")),
                dtype=entry.get("dtype", "float32"),
                value_shape=tuple(entry.get("value_shape", ())),
                dimension_names=tuple(entry.get("dimension_names", ())),
            )
            for entry in data.get("time_series_specs", ())
        )
        annotations = tuple(
            AnnotationDescriptor(
                key=entry["key"],
                annotation_type=AnnotationType(entry["annotation_type"]),
                value_type=entry.get("value_type"),
                unit=entry.get("unit"),
                description=entry.get("description"),
            )
            for entry in data.get("annotations", ())
        )
        tasks = tuple(_resolve_task(entry["task_type"]) for entry in data.get("tasks", ()))
        return DatasetSchema(
            time_series_specs=specs,
            annotations=annotations,
            tasks=tasks,
        )
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise InvalidManifestError(f"invalid manifest 'schema' block: {exc}") from exc


def _data_source(entry: dict[str, Any] | None) -> DataSource | None:
    """Rebuild a spec's data source from the record stored beside it.

    Args:
        entry: The stored ``data_source`` object, or ``None``.

    Returns:
        The data source, or ``None`` if the spec declares none.
    """
    if entry is None:
        return None
    return DataSource(data_source_type=entry["data_source_type"], name=entry["name"], provider=entry.get("provider"))


def _resolve_task(task_type: str) -> type[Task]:
    try:
        return TASKS[TaskType(task_type)]
    except ValueError as exc:
        raise ValueError(f"unknown task_type {task_type!r}") from exc


def _counts_to_dict(counts: ManifestCounts) -> dict[str, Any]:
    return {
        "samples": counts.samples,
        "annotations": counts.annotations,
        "tasks": dict(counts.tasks),
        "time_series_chunks": counts.time_series_chunks,
        "time_series_index_rows": counts.time_series_index_rows,
        "time_series_specs": dict(counts.time_series_specs),
    }


def _counts_from_dict(data: dict[str, Any]) -> ManifestCounts:
    try:
        return ManifestCounts(
            samples=data.get("samples", 0),
            annotations=data.get("annotations", 0),
            tasks=dict(data.get("tasks", {})),
            time_series_chunks=data.get("time_series_chunks", 0),
            time_series_index_rows=data.get("time_series_index_rows", 0),
            time_series_specs=dict(data.get("time_series_specs", {})),
        )
    except (ValueError, TypeError, AttributeError) as exc:
        raise InvalidManifestError(f"invalid manifest 'counts' block: {exc}") from exc


def _files_to_dict(files: ManifestFiles) -> dict[str, Any]:
    return {
        "samples": list(files.samples),
        "annotations": list(files.annotations),
        "time_series_index": list(files.time_series_index),
        "tasks": list(files.tasks),
        "time_series": list(files.time_series),
    }


def _files_from_dict(data: dict[str, Any]) -> ManifestFiles:
    try:
        return ManifestFiles(
            samples=_parts(data, "samples"),
            annotations=_parts(data, "annotations"),
            time_series_index=_parts(data, "time_series_index"),
            tasks=_str_tuple(data.get("tasks", ()), "tasks"),
            time_series=_str_tuple(data.get("time_series", ()), "time_series"),
        )
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise InvalidManifestError(f"invalid manifest 'files' block: {exc}") from exc


def _parts(data: dict[str, Any], key: str) -> tuple[str, ...]:
    """Read a required list-of-parts field, rejecting a bare string.

    Args:
        data: The manifest ``files`` block.
        key: The field to read.

    Returns:
        The field's parts as a tuple.

    Raises:
        TypeError: If the field is a string rather than a list of paths.
    """
    value = data[key]
    if isinstance(value, str):
        raise TypeError(f"'files.{key}' must be a list of parts, not a string")
    return tuple(value)


def _part_stats_to_dict(part_stats: dict[str, tuple[PartStat, ...]]) -> dict[str, Any]:
    return {
        table: [
            {
                "path": part.path,
                "n_rows": part.n_rows,
                "first_key": _key_to_json(part.first_key),
                "last_key": _key_to_json(part.last_key),
            }
            for part in parts
        ]
        for table, parts in part_stats.items()
    }


def _key_to_json(key: tuple[object, ...] | None) -> list[Any] | None:
    if key is None:
        return None
    return [value.hex() if isinstance(value, bytes) else value for value in key]


def _part_stats_from_dict(
    data: dict[str, Any], files: ManifestFiles, id_encoding: dict[str, str]
) -> dict[str, tuple[PartStat, ...]]:
    uuid16 = {name for name, encoding in id_encoding.items() if encoding == "uuid16"}
    result: dict[str, tuple[PartStat, ...]] = {}
    for table, entries in data.items():
        if table not in PART_STAT_KEYS:
            raise InvalidManifestError(f"unknown part_stats table {table!r}")
        logical = PART_STAT_KEYS[table]
        try:
            parts = tuple(
                PartStat(
                    path=entry["path"],
                    n_rows=entry["n_rows"],
                    first_key=_key_from_json(entry["first_key"], logical, uuid16),
                    last_key=_key_from_json(entry["last_key"], logical, uuid16),
                )
                for entry in entries
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise InvalidManifestError(f"invalid manifest 'part_stats' block: {exc}") from exc
        expected = getattr(files, table)
        if tuple(part.path for part in parts) != expected:
            raise InvalidManifestError(
                f"part_stats[{table!r}] paths {[part.path for part in parts]} "
                f"disagree with files.{table} {list(expected)}"
            )
        result[table] = parts
    return result


def _key_from_json(value: list[str] | None, logical: tuple[str, ...], uuid16: set[str]) -> tuple[object, ...] | None:
    if value is None:
        return None
    return tuple(bytes.fromhex(item) if name in uuid16 else item for item, name in zip(value, logical, strict=True))
