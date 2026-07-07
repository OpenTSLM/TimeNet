"""The :class:`Manifest`: the compiled ``manifest.json`` and its JSON codec.

The manifest is pure data with no file I/O; the writer and reader own reading/writing the file. Its
``schema`` block is a faithful serialization of :class:`~timenet.types.DatasetSchema` (flat descriptors),
so there is no separate set of "entry" types to keep in sync.
"""

from dataclasses import dataclass, field
import json
from typing import Any, ClassVar

from timenet.errors import InvalidManifestError
from timenet.manifest.counts import ManifestCounts
from timenet.manifest.files import ManifestFiles
from timenet.types import (
    TASKS,
    AnnotationDescriptor,
    AnnotationType,
    DatasetMetadata,
    DatasetSchema,
    DataSource,
    Domain,
    License,
    Task,
    TaskType,
    TimeSeriesSpec,
    Version,
    ureg,
)


@dataclass(frozen=True)
class Manifest:
    """The single source of truth a consumer reads to interpret a dataset version.

    Raises:
        InvalidManifestError: If ``timef_format_version`` is not a supported version.
    """

    SUPPORTED_FORMAT_VERSIONS: ClassVar[frozenset[int]] = frozenset({1})

    dataset_id: str
    metadata: DatasetMetadata
    files: ManifestFiles
    schema: DatasetSchema = field(default_factory=DatasetSchema)
    counts: ManifestCounts = field(default_factory=ManifestCounts)
    checksums: dict[str, str] = field(default_factory=dict)
    id_encoding: dict[str, str] = field(default_factory=dict)
    derived_from: dict[str, str] | None = None
    timef_format_version: int = 1

    def __post_init__(self) -> None:
        """Validate the format version and the denormalized ``dataset_id``.

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
        return cls(
            dataset_id=data["dataset_id"],
            metadata=_metadata_from_dict(data["metadata"]),
            files=_files_from_dict(data["files"]),
            schema=_schema_from_dict(data.get("schema", {})),
            counts=_counts_from_dict(data.get("counts", {})),
            checksums=checksums,
            id_encoding=id_encoding,
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
        return DatasetMetadata(
            dataset_id=data["dataset_id"],
            dataset_version=Version.parse(data["dataset_version"]),
            name=data["name"],
            description=data["description"],
            license=License(data["license"]),
            domains=tuple(Domain(domain) for domain in _str_tuple(data.get("domains", ()), "domains")),
            tags=_str_tuple(data.get("tags", ()), "tags"),
            source_url=data.get("source_url"),
            yaml_schema_version=data.get("yaml_schema_version", 1),
        )
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise InvalidManifestError(f"invalid manifest 'metadata' block: {exc}") from exc


def _schema_to_dict(schema: DatasetSchema) -> dict[str, Any]:
    return {
        "time_series_specs": [
            {
                "spec_type": spec.spec_type,
                "name": spec.name,
                "unit_sampling_rate": str(spec.unit_sampling_rate),
                "unit_timestamp": str(spec.unit_timestamp),
                "unit_value": str(spec.unit_value),
                "data_source": spec.data_source.data_source_type if spec.data_source else None,
            }
            for spec in schema.time_series_specs
        ],
        "data_sources": [
            {"data_source_type": ds.data_source_type, "name": ds.name, "provider": ds.provider}
            for ds in schema.data_sources
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
        data_sources = tuple(
            DataSource(
                data_source_type=entry["data_source_type"],
                name=entry["name"],
                provider=entry.get("provider"),
            )
            for entry in data.get("data_sources", ())
        )
        by_type = {ds.data_source_type: ds for ds in data_sources}
        specs = tuple(
            TimeSeriesSpec(
                spec_type=entry["spec_type"],
                name=entry["name"],
                unit_sampling_rate=ureg.Unit(entry["unit_sampling_rate"]),
                unit_timestamp=ureg.Unit(entry["unit_timestamp"]),
                unit_value=ureg.Unit(entry["unit_value"]),
                data_source=_resolve_data_source(entry.get("data_source"), by_type),
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
            data_sources=data_sources,
            annotations=annotations,
            tasks=tasks,
        )
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise InvalidManifestError(f"invalid manifest 'schema' block: {exc}") from exc


def _resolve_data_source(data_source_type: str | None, by_type: dict[str, DataSource]) -> DataSource | None:
    if data_source_type is None:
        return None
    if data_source_type not in by_type:
        raise ValueError(f"spec references unknown data_source {data_source_type!r}")
    return by_type[data_source_type]


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
        "samples": files.samples,
        "annotations": files.annotations,
        "time_series_index": files.time_series_index,
        "tasks": list(files.tasks),
        "time_series": list(files.time_series),
    }


def _files_from_dict(data: dict[str, Any]) -> ManifestFiles:
    try:
        return ManifestFiles(
            samples=data["samples"],
            annotations=data["annotations"],
            time_series_index=data["time_series_index"],
            tasks=_str_tuple(data.get("tasks", ()), "tasks"),
            time_series=_str_tuple(data.get("time_series", ()), "time_series"),
        )
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise InvalidManifestError(f"invalid manifest 'files' block: {exc}") from exc
