"""Dataset-level descriptive identity and derived type declaration."""

from dataclasses import dataclass, field
import re

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
    dataset_version: Version
    name: str
    description: str
    license: License
    domains: tuple[Domain, ...] = ()
    tags: tuple[str, ...] = ()
    source_url: str | None = None
    yaml_schema_version: int = 1

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


@dataclass(frozen=True)
class DatasetSchema:
    """A dataset's type declaration, derived from its data (never hand-authored).

    Holds flat descriptor instances for specs / data sources / annotations, and the real built-in
    :class:`~timenet.types.tasks.Task` subclasses (resolved against the registry, not reconstructed).
    """

    time_series_specs: tuple[TimeSeriesSpec, ...] = ()
    data_sources: tuple[DataSource, ...] = ()
    annotations: tuple[AnnotationDescriptor, ...] = ()
    tasks: tuple[type[Task], ...] = field(default=())

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
