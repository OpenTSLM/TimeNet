"""Dataset-level descriptive identity and derived type declaration."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Self, TypeVar

from pydantic import (
    Field,
    PlainSerializer,
    PlainValidator,
    StrictStr,
    WithJsonSchema,
    field_validator,
    model_validator,
)

from timenet.errors import TimeFValidationError
from timenet.types._model import TimeFModel
from timenet.types.access import Access
from timenet.types.annotations import AnnotationDescriptor
from timenet.types.domains import Domain
from timenet.types.licenses import License
from timenet.types.metadata_fields import DatasetId, NonEmptyText, SemanticVersion
from timenet.types.reference_keys import ParentAlias
from timenet.types.references import DatasetRef
from timenet.types.specs import TimeSeriesSpec
from timenet.types.tasks import TASKS, Task, TaskType


if TYPE_CHECKING:
    from timenet.registry.base import BaseRegistry


T = TypeVar("T")


class DatasetMetadata(TimeFModel):
    """Dataset identity, release version, source information, and access requirements.

    The dataset card defines these fields. ``dataset_version`` identifies the TimeNet release.
    ``source_revision`` identifies the original source release, tag, or commit.
    ``yaml_schema_version`` identifies the card schema. ``dataset_id`` is a case-sensitive
    ``org/name`` pair with exactly one slash. Casing-only differences can cause conflicts on
    case-insensitive filesystems.
    """

    dataset_id: DatasetId
    """HuggingFace-style ``org/name`` pair, case-sensitive, exactly one slash."""
    dataset_version: SemanticVersion
    """Semantic version of the compiled TimeNet release."""
    parents: dict[ParentAlias, DatasetRef] = Field(default_factory=dict)
    """Exact parent releases indexed by connector-local aliases."""
    source_revision: StrictStr | None = None
    """Upstream release, tag, or commit, separate from the compiled dataset version."""
    name: NonEmptyText
    """Human-readable display name."""
    description: NonEmptyText
    """Free-text description of the dataset."""
    license: License
    """Legal license of the source data, as an SPDX-style identifier."""
    domains: tuple[Domain, ...] = ()
    """Kinds of data the dataset contains."""
    tags: tuple[StrictStr, ...] = ()
    """Free-form tags for search and grouping."""
    source_url: StrictStr | None = None
    """Link to the dataset's origin, if any."""
    license_url: StrictStr | None = None
    """Where to read the full license text. Required and nonblank when ``license`` is :attr:`License.OTHER`."""
    citation: StrictStr | None = None
    """How to cite the dataset, when the source asks for attribution."""
    access: Access = Access.OPEN
    """How a user obtains the data (see :class:`Access`). ``OPEN`` needs nothing.

    An import into a child dataset does not change it. The level that applies to a release with
    parents is :func:`~timenet.types.access.effective_access`.
    """
    access_url: StrictStr | None = None
    """Where to obtain access (the DUA or credentialing page). Required when ``access`` is not ``OPEN``."""
    yaml_schema_version: Annotated[int, Field(strict=True, ge=1, le=1)] = 1
    """Version of the card's own field schema."""

    @field_validator("parents", mode="before", json_schema_input_type=dict[ParentAlias, DatasetRef | str])
    @classmethod
    def _parse_parents(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: DatasetRef.parse(ref) if isinstance(ref, str) else ref for key, ref in value.items()}
        return value

    @field_validator("name", "description")
    @classmethod
    def _non_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def _dependent_urls(self) -> Self:
        if self.license is License.OTHER and (self.license_url is None or not self.license_url.strip()):
            raise TimeFValidationError("license_url is required when license is License.OTHER")
        if self.access is not Access.OPEN and not self.access_url:
            raise ValueError(f"access_url is required when access is {self.access.value!r}")
        return self

    @model_validator(mode="after")
    def _one_alias_per_parent(self) -> Self:
        # A child imports a release under one alias only, so its card declares the release once.
        aliases: dict[str, list[str]] = {}
        for alias, ref in self.parents.items():
            aliases.setdefault(str(ref), []).append(alias)
        for release, names in aliases.items():
            if len(names) > 1:
                raise ValueError(
                    f"{self.dataset_id} declares the parent {release} under the parent aliases {sorted(names)}; "
                    "declare each parent release under one alias only"
                )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetMetadata":
        """Load and validate a dataset card YAML into metadata.

        Authoring mistakes surface as clear, aggregated messages instead of a stack trace from deep
        inside coercion. PyYAML is optional and imported lazily. Install the ``timenet[build]`` extra
        to use this method.

        Args:
            path: Path to the card YAML file.

        Returns:
            The constructed :class:`DatasetMetadata`.

        Raises:
            TimeNetInvalidCardError: If the build extra is missing, or the card is unreadable or is
                not a mapping.
        """
        from timenet.errors import TimeNetInvalidCardError  # ruff: ignore[import-outside-top-level]

        card_path = Path(path)
        try:
            import yaml  # ruff: ignore[import-outside-top-level]
        except ModuleNotFoundError as exc:
            raise TimeNetInvalidCardError(
                "reading a dataset card needs PyYAML; install the 'timenet[build]' extra"
            ) from exc

        try:
            raw = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise TimeNetInvalidCardError(f"could not read dataset card {card_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise TimeNetInvalidCardError(f"dataset card {card_path} must be a YAML mapping, got {type(raw).__name__}")

        return cls.model_validate(raw)


def _resolve_task(value: Any) -> type[Task]:
    """Resolve a manifest task declaration to its public task class.

    Returns:
        The built-in task class.

    Raises:
        ValueError: If the declaration does not name a built-in task.
    """
    if isinstance(value, type) and issubclass(value, Task):
        return value
    if not isinstance(value, dict):
        raise ValueError("task declaration must be a Task class or an object with task_type")
    try:
        task_type = TaskType(value["task_type"])
        return TASKS[task_type]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unknown task_type {value.get('task_type')!r}") from exc


TaskDeclaration = Annotated[
    type[Task],
    PlainValidator(_resolve_task, json_schema_input_type=dict[str, str]),
    PlainSerializer(lambda task: {"task_type": str(task.task_type)}, return_type=dict[str, str]),
    WithJsonSchema(
        {
            "type": "object",
            "properties": {"task_type": {"type": "string", "enum": [item.value for item in TaskType]}},
            "required": ["task_type"],
            "additionalProperties": False,
        }
    ),
]
"""A built-in task class encoded as a ``task_type`` declaration."""


class DatasetSchema(TimeFModel):
    """A dataset's type declaration, derived from its data (never hand-authored).

    It holds flat descriptor instances for specs and annotations. It also holds the real built-in
    :class:`~timenet.types.tasks.Task` subclasses, resolved against the registry instead of reconstructed.
    """

    time_series_specs: tuple[TimeSeriesSpec, ...] = ()
    """Descriptors for the dataset's measurement modalities."""
    annotations: tuple[AnnotationDescriptor, ...] = ()
    """Type-level descriptors for the dataset's annotation keys."""
    tasks: tuple[TaskDeclaration, ...] = ()
    """Built-in ``Task`` subclasses the dataset declares, resolved from the registry."""


@dataclass(frozen=True)
class ListedRelease:
    """A card of a dataset listing, with the registry that lists it."""

    card: DatasetMetadata
    """The card of the release."""
    registry: "BaseRegistry | None"
    """The registry that holds the manifest of the release, or ``None`` for the card of an installed connector."""


@dataclass(frozen=True)
class DatasetFilter:
    """The filters of a dataset search. Filters combine with AND, and an empty filter matches every dataset."""

    query: tuple[str, ...] = ()
    """Terms of which any must appear in the name, description, or tags, ignoring case."""
    domain: tuple[Domain, ...] = ()
    """Domains of which the dataset must have any."""
    task: tuple[type[Task], ...] = ()
    """Task classes of which the schema must declare any."""
    license: tuple[License, ...] = ()
    """Licenses of which the dataset must have any."""
    time_series_spec: tuple[str, ...] = ()
    """``spec_type`` values that the schema must declare all of."""
    dataset_id: tuple[str, ...] = ()
    """IDs of which the dataset must have any."""
    tag: tuple[str, ...] = ()
    """Tags that the dataset must declare all of."""
    limit: int = 100
    """The maximum number of results."""

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise TimeFValidationError(f"limit must be 0 or more, got {self.limit}")

    @classmethod
    def of(  # ruff: ignore[too-many-arguments]
        cls,
        *,
        query: str | list[str] | None = None,
        domain: Domain | list[Domain] | None = None,
        task: type[Task] | list[type[Task]] | None = None,
        license: License | list[License] | None = None,
        time_series_spec: str | list[str] | None = None,
        dataset_id: str | list[str] | None = None,
        tag: str | list[str] | None = None,
        limit: int = 100,
    ) -> Self:
        """Build a filter from one value, a list, or None for each filter.

        Args:
            query: Match any term in the name, description, or tags, ignoring case.
            domain: Match any of these domains.
            task: Match any of these task classes in the schema.
            license: Match any of these licenses.
            time_series_spec: Match all of these ``spec_type`` values.
            dataset_id: Match any of these IDs.
            tag: Match all of these tags.
            limit: The maximum number of results.

        Returns:
            The filter. ``None`` or an empty list sets no restriction.

        Raises:
            TimeFValidationError: If ``limit`` is negative.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by __post_init__
        return cls(
            query=_as_tuple(query),
            domain=_as_tuple(domain),
            task=_as_tuple(task),
            license=_as_tuple(license),
            time_series_spec=_as_tuple(time_series_spec),
            dataset_id=_as_tuple(dataset_id),
            tag=_as_tuple(tag),
            limit=limit,
        )

    @property
    def needs_schema(self) -> bool:
        """Whether a ``task`` or ``time_series_spec`` filter is set. Both read the schema of a manifest."""
        return bool(self.task or self.time_series_spec)

    def matches_card(self, card: DatasetMetadata) -> bool:
        """Return whether a card matches the ``query``, ``domain``, ``license``, ``dataset_id`` and ``tag`` filters."""
        if self.dataset_id and card.dataset_id not in self.dataset_id:
            return False
        if self.license and card.license not in self.license:
            return False
        if self.domain and not set(card.domains) & set(self.domain):
            return False
        if not set(self.tag) <= set(card.tags):
            return False
        text = " ".join([card.name, card.description, *card.tags]).lower()
        return not self.query or any(term.lower() in text for term in self.query)

    def matches_schema(self, schema: DatasetSchema) -> bool:
        """Return whether a schema matches the ``task`` and ``time_series_spec`` filters."""
        if self.task and not set(self.task) & set(schema.tasks):
            return False
        return set(self.time_series_spec) <= {spec.spec_type for spec in schema.time_series_specs}

    def select(self, listed: Iterable[ListedRelease]) -> list[DatasetMetadata]:
        """Return the cards of the listed releases that match, in order, with at most ``limit`` entries.

        A ``task`` or ``time_series_spec`` filter reads the schema of a release only if its card matches. A release
        without a registry has no schema, so it never matches these filters.

        Args:
            listed: The releases to filter.

        Returns:
            The matching cards.
        """
        results: list[DatasetMetadata] = []
        for release in listed:
            if len(results) >= self.limit:
                break
            if not self.matches_card(release.card):
                continue
            if self.needs_schema:
                if release.registry is None:
                    continue
                manifest = release.registry.get_manifest(release.card.dataset_id, str(release.card.dataset_version))
                if not self.matches_schema(manifest.dataset_schema):
                    continue
            results.append(release.card)
        return results


def _as_tuple(value: T | list[T] | None) -> tuple[T, ...]:
    """Return a filter value as a tuple: empty for ``None``, the items of a list, or the one value."""
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(value)
    return (value,)
