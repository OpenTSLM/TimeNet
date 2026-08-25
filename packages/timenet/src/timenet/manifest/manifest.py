"""The :class:`Manifest`: the compiled ``manifest.json`` file and its JSON codec.

The manifest is pure data. It has no file I/O. The writer and the reader do the read and write
operations for the file. The ``schema`` block is a direct serialization of
:class:`~timenet.types.DatasetSchema` (flat descriptors). No separate set of "entry" types exists to
keep in sync.
"""

from dataclasses import dataclass, field
import json
from typing import Any, ClassVar

from timenet.errors import TimeNetInvalidManifestError
from timenet.format.constants import check_relative_path
from timenet.manifest.counts import ManifestCounts
from timenet.manifest.files import FilePart, ManifestFiles
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
        TimeNetInvalidManifestError: If ``timef_format_version`` is not a supported version.
    """

    SUPPORTED_FORMAT_VERSIONS: ClassVar[frozenset[int]] = frozenset({1})

    dataset_id: str
    """A denormalized copy of ``metadata.dataset_id``. A reader can get the id without parsing metadata."""
    metadata: DatasetMetadata
    """Descriptive identity of the dataset (name, version, license, domains, tags)."""
    files: ManifestFiles
    """Descriptor (path, checksum, size) for every data artifact, grouped by kind."""
    schema: DatasetSchema = field(default_factory=DatasetSchema)
    """Structural schema: time-series specs, annotations, and tasks."""
    counts: ManifestCounts = field(default_factory=ManifestCounts)
    """Row and entity counts recorded for quick inspection."""
    id_encoding: dict[str, str] = field(default_factory=dict)
    """Logical id -> ``"uuid16"`` for ids stored as ``binary(16)``. An id not listed here is a string."""
    values_backend: str = ValuesBackend.PARQUET
    """Storage backend for the time-series values plane."""
    value_encoding: dict[str, str] = field(default_factory=dict)
    """``spec_type`` -> the values-column encoding that its shards carry.

    This field exists only for provenance. Parquet already records the applied encoding in each
    file's footer, so a reader does not need this field. The field lets a builder see what a build
    chose without opening a shard. The field is empty for a backend with no such choice.
    """
    derived_from: dict[str, str] | None = None
    """Copy-on-write lineage (base version and operation), or ``None`` for a newly built version."""
    build_env: dict[str, Any] = field(default_factory=dict)
    """The Python version and package set that produced this version.

    Provenance only: nothing reads it to interpret the data. It is here so a builder can answer what
    produced a dataset version without re-deriving it from a build log.
    """
    timef_format_version: int = 1
    """The TimeF manifest format version. The value must be in ``SUPPORTED_FORMAT_VERSIONS``."""

    def __post_init__(self) -> None:
        """Validate the format version, the values backend, and the denormalized ``dataset_id``.

        ``dataset_id`` is a top-level copy of ``metadata.dataset_id``, so a consumer can read the id
        without parsing the metadata block. The two values must match.

        Raises:
            TimeNetInvalidManifestError: If ``timef_format_version`` is unsupported, or ``dataset_id`` does
                not match ``metadata.dataset_id``.
        """
        if self.timef_format_version not in self.SUPPORTED_FORMAT_VERSIONS:
            raise TimeNetInvalidManifestError(
                f"unsupported timef_format_version {self.timef_format_version!r}; "
                f"supported: {sorted(self.SUPPORTED_FORMAT_VERSIONS)}"
            )
        if self.values_backend not in SUPPORTED_VALUES_BACKENDS:
            raise TimeNetInvalidManifestError(
                f"unsupported values_backend {self.values_backend!r}; "
                f"supported: {', '.join(sorted(SUPPORTED_VALUES_BACKENDS))}"
            )
        if self.dataset_id != self.metadata.dataset_id:
            raise TimeNetInvalidManifestError(
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
            "id_encoding": dict(self.id_encoding),
            "values_backend": self.values_backend,
            "value_encoding": dict(self.value_encoding),
            "derived_from": dict(self.derived_from) if self.derived_from is not None else None,
            "build_env": dict(self.build_env) if self.build_env is not None else None,
        }

    def to_json(self) -> str:
        """Serialize the manifest to a pretty JSON string.

        Returns:
            The JSON text.
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Manifest":
        """Parse a manifest dict. Allow missing optional blocks.

        Args:
            data: The manifest dict, for example the output of ``json.loads``.

        Returns:
            The parsed :class:`Manifest`.

        Raises:
            TimeNetInvalidManifestError: If a required key is missing or a block is invalid.
        """
        for required in ("timef_format_version", "dataset_id", "metadata", "files"):
            if required not in data:
                raise TimeNetInvalidManifestError(f"manifest missing required key {required!r}")
        id_encoding = _dict_block(data, "id_encoding")
        derived_from = _optional_dict_block(data, "derived_from")
        return cls(
            dataset_id=data["dataset_id"],
            metadata=_metadata_from_dict(data["metadata"]),
            files=_files_from_dict(data["files"]),
            schema=_schema_from_dict(data.get("schema", {})),
            counts=_counts_from_dict(data.get("counts", {})),
            id_encoding=id_encoding,
            values_backend=data.get("values_backend", ValuesBackend.PARQUET),
            value_encoding=_dict_block(data, "value_encoding"),
            derived_from=derived_from,
            build_env=data.get("build_env", {}),
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
            TimeNetInvalidManifestError: If the text is not valid JSON or a block is invalid.
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TimeNetInvalidManifestError(f"manifest is not valid JSON: {exc}") from exc
        return cls.from_dict(data)


def _dict_block(data: dict[str, Any], key: str) -> dict:
    """Convert an optional manifest dict block to a ``dict``. Name the block in the error on failure.

    Args:
        data: The manifest dict.
        key: The block's key.

    Returns:
        The block as a ``dict``. The result is empty when the block is absent.

    Raises:
        TimeNetInvalidManifestError: If the block is present but is not a mapping.
    """
    try:
        return dict(data.get(key, {}))
    except (ValueError, TypeError) as exc:
        raise TimeNetInvalidManifestError(f"invalid manifest {key!r} block: {exc}") from exc


def _optional_dict_block(data: dict[str, Any], key: str) -> dict | None:
    """Convert a nullable manifest dict block to a ``dict`` or ``None``. Name the block in the error on failure.

    Args:
        data: The manifest dict.
        key: The block's key.

    Returns:
        The block as a ``dict``, or ``None`` when the value is ``null`` or absent.

    Raises:
        TimeNetInvalidManifestError: If the block is present, is not null, and is not a mapping.
    """
    value = data.get(key)
    if value is None:
        return None
    try:
        return dict(value)
    except (ValueError, TypeError) as exc:
        raise TimeNetInvalidManifestError(f"invalid manifest {key!r} block: {exc}") from exc


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
        raise TimeNetInvalidManifestError(f"invalid manifest 'metadata' block: {exc}") from exc


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
        raise TimeNetInvalidManifestError(f"invalid manifest 'schema' block: {exc}") from exc


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
        "registered_annotations": counts.registered_annotations,
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
            registered_annotations=data.get("registered_annotations", 0),
            tasks=dict(data.get("tasks", {})),
            time_series_chunks=data.get("time_series_chunks", 0),
            time_series_index_rows=data.get("time_series_index_rows", 0),
            time_series_specs=dict(data.get("time_series_specs", {})),
        )
    except (ValueError, TypeError, AttributeError) as exc:
        raise TimeNetInvalidManifestError(f"invalid manifest 'counts' block: {exc}") from exc


def _files_to_dict(files: ManifestFiles) -> dict[str, Any]:
    return {
        "samples": [_part_to_dict(part) for part in files.samples],
        "annotations": [_part_to_dict(part) for part in files.annotations],
        "time_series_index": [_part_to_dict(part) for part in files.time_series_index],
        "tasks": [_part_to_dict(part) for part in files.tasks],
        "time_series": [_part_to_dict(part) for part in files.time_series],
    }


def _part_to_dict(part: FilePart) -> dict[str, Any]:
    return {"path": part.path, "checksum": part.checksum, "size": part.size}


def _files_from_dict(data: dict[str, Any]) -> ManifestFiles:
    try:
        return ManifestFiles(
            samples=_parts(data["samples"], "samples"),
            annotations=_parts(data["annotations"], "annotations"),
            time_series_index=_parts(data["time_series_index"], "time_series_index"),
            tasks=_parts(data.get("tasks", ()), "tasks"),
            time_series=_parts(data.get("time_series", ()), "time_series"),
        )
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise TimeNetInvalidManifestError(f"invalid manifest 'files' block: {exc}") from exc


def _parts(value: Any, key: str) -> tuple[FilePart, ...]:
    """Read one ``files`` field. The field is a list of ``{path, checksum, size}`` objects. Reject a bare string.

    Args:
        value: The field's value from the manifest ``files`` block.
        key: The field's name. The name appears in error messages.

    Returns:
        The field's parts as a tuple of :class:`FilePart`. A missing ``path``, ``checksum``, or
        ``size`` raises ``KeyError``. The caller re-raises this error as ``TimeNetInvalidManifestError``.

    Raises:
        TypeError: If the field is a string, or an entry is not an object.
    """
    if isinstance(value, str):
        raise TypeError(f"'files.{key}' must be a list of file entries, not a string")
    return tuple(_part_from_dict(entry, key) for entry in value)


def _part_from_dict(entry: Any, key: str) -> FilePart:
    if not isinstance(entry, dict):
        raise TypeError(f"'files.{key}' entries must be objects, got {type(entry).__name__}")
    path, checksum, size = entry["path"], entry["checksum"], entry["size"]
    if not isinstance(path, str) or not path:
        raise TypeError(f"'files.{key}' path must be a non-empty string, got {path!r}")
    check_relative_path(f"'files.{key}' path", path)
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        raise ValueError(f"'files.{key}' checksum must be 'sha256:<hex>', got {checksum!r}")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise TypeError(f"'files.{key}' size must be a non-negative integer, got {size!r}")
    return FilePart(path=path, checksum=checksum, size=size)
