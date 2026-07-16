"""Dataset-level descriptive identity and derived type declaration."""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

from timenet.types.annotations import AnnotationDescriptor
from timenet.types.domains import Domain
from timenet.types.licenses import License
from timenet.types.specs import DataSource, TimeSeriesSpec
from timenet.types.tasks import Task
from timenet.types.version import Version


# A HuggingFace-style ``org/name`` pair: exactly one slash, no leading/trailing/empty segment.
_DATASET_ID = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class DatasetMetadata:
    """A dataset's descriptive identity: who it is, not what it emits.

    Authored in the dataset card. ``dataset_version`` is the upstream source's semantic version;
    ``yaml_schema_version`` is the card's own field-schema version. ``dataset_id`` is an ``org/name``
    pair (HuggingFace style, exactly one slash); ids are case-sensitive, so avoid casing-only
    differences on case-insensitive filesystems.
    """

    dataset_id: str
    """HuggingFace-style ``org/name`` pair; case-sensitive, exactly one slash."""
    dataset_version: Version
    """Semantic version of the upstream source data."""
    name: str
    """Human-readable display name."""
    description: str
    """Free-text description of the dataset."""
    license: License
    """Legal license of the source data, as an SPDX-style identifier."""
    domains: tuple[Domain, ...] = ()
    """Kinds of data the dataset contains."""
    tags: tuple[str, ...] = ()
    """Free-form tags for search and grouping."""
    source_url: str | None = None
    """Link to the dataset's origin, if any."""
    yaml_schema_version: int = 1
    """Version of the card's own field schema."""

    def __post_init__(self) -> None:
        """Validate the ``dataset_id`` shape.

        No segment may start with ``.``: dataset ids are joined into filesystem paths, so a ``.`` /
        ``..`` segment could escape the registry/storage root, and a leading-dot name (e.g. ``.git``)
        writes to disk but is skipped by discovery, which drops hidden directories.

        Raises:
            ValueError: If ``dataset_id`` is not a single-slash ``org/name`` pair, or has a segment
                that starts with ``.``.
        """
        if not _DATASET_ID.match(self.dataset_id) or any(part.startswith(".") for part in self.dataset_id.split("/")):
            raise ValueError(
                f"dataset_id must be 'org/name' (letters, digits, ., _, -; exactly one slash; no "
                f"segment may start with '.'), got {self.dataset_id!r}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetMetadata":
        """Build metadata from a plain mapping of card fields.

        Enum and version fields arrive as strings (``license``, ``domains``, ``dataset_version``) and are
        coerced here; unmodeled keys are ignored. Shared by :meth:`from_yaml` and the manifest codec so
        the mapping lives in one place. Coercion may raise ``KeyError`` (missing field) or ``ValueError``
        (bad license/domain/version/id); callers wrap these in their own error type.

        Args:
            data: A mapping with the card fields (strings for the enums and the version).

        Returns:
            The constructed :class:`DatasetMetadata`.
        """
        return cls(
            dataset_id=data["dataset_id"],
            dataset_version=Version.parse(data["dataset_version"]),
            name=data["name"],
            description=data["description"],
            license=License(data["license"]),
            domains=tuple(Domain(domain) for domain in data.get("domains", ())),
            tags=tuple(data.get("tags", ())),
            source_url=data.get("source_url"),
            yaml_schema_version=data.get("yaml_schema_version", 1),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetMetadata":
        """Load and validate a dataset card YAML into metadata.

        The card is validated against the packaged ``dataset-card.schema.json`` (so authoring mistakes
        surface with clear, aggregated messages) before construction. PyYAML and jsonschema are optional
        and imported lazily; install the ``timenet[curation]`` extra to use this.

        Args:
            path: Path to the card YAML file.

        Returns:
            The constructed :class:`DatasetMetadata`.

        Raises:
            InvalidCardError: If the curation extra is missing, or the card is unreadable, is not a
                mapping, fails schema validation, or has an invalid field value.
        """
        from timenet.errors import InvalidCardError

        card_path = Path(path)
        try:
            import jsonschema
            import yaml
        except ModuleNotFoundError as exc:
            raise InvalidCardError(
                "reading a dataset card needs PyYAML and jsonschema; install the 'timenet[curation]' extra"
            ) from exc

        from timenet.schemas import DATASET_CARD_SCHEMA

        try:
            raw = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise InvalidCardError(f"could not read dataset card {card_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise InvalidCardError(f"dataset card {card_path} must be a YAML mapping, got {type(raw).__name__}")

        errors = sorted(
            jsonschema.Draft202012Validator(DATASET_CARD_SCHEMA).iter_errors(raw),
            key=lambda error: list(error.path),
        )
        if errors:
            detail = "; ".join(
                f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}" for error in errors
            )
            raise InvalidCardError(f"dataset card {card_path} failed validation: {detail}")

        try:
            return cls.from_dict(raw)
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            raise InvalidCardError(f"dataset card {card_path} is invalid: {exc}") from exc


@dataclass(frozen=True)
class DatasetSchema:
    """A dataset's type declaration, derived from its data (never hand-authored).

    Holds flat descriptor instances for specs / data sources / annotations, and the real built-in
    :class:`~timenet.types.tasks.Task` subclasses (resolved against the registry, not reconstructed).
    """

    time_series_specs: tuple[TimeSeriesSpec, ...] = ()
    """Descriptors for the dataset's measurement modalities."""
    data_sources: tuple[DataSource, ...] = ()
    """Origins that produced the modalities, keyed by ``data_source_type``."""
    annotations: tuple[AnnotationDescriptor, ...] = ()
    """Type-level descriptors for the dataset's annotation keys."""
    tasks: tuple[type[Task], ...] = field(default=())
    """Built-in ``Task`` subclasses the dataset declares, resolved from the registry."""

    def __post_init__(self) -> None:
        """Reject a spec whose data source is absent from ``data_sources`` or is registered ambiguously.

        The schema is always derived from data, so every ``spec.data_source`` is expected to appear in
        ``data_sources``. Enforcing it keeps the manifest codec a lossless round-trip: serialization
        stores only a spec's ``data_source_type`` tag and rebuilds the rest from ``data_sources`` keyed
        by that tag. That is only faithful when a spec's source matches the registered one by value and
        each tag maps to exactly one source, so both are checked here.

        Raises:
            ValueError: If two data sources share a ``data_source_type`` with different fields, or a
                spec references a data source not present (by value) in ``data_sources``.
        """
        by_type: dict[str, DataSource] = {}
        for source in self.data_sources:
            existing = by_type.get(source.data_source_type)
            if existing is not None and existing != source:
                raise ValueError(
                    f"data_sources has conflicting entries for data_source_type "
                    f"{source.data_source_type!r}: {existing!r} and {source!r}"
                )
            by_type[source.data_source_type] = source
        for spec in self.time_series_specs:
            source = spec.data_source
            if source is not None and by_type.get(source.data_source_type) != source:
                raise ValueError(
                    f"time_series_spec {spec.spec_type!r} references data source {source!r} not in data_sources"
                )
