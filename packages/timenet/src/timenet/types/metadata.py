"""Dataset-level descriptive identity and derived type declaration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Self

from pydantic import Field, StrictStr, field_validator, model_validator

from timenet.types._model import TimeFModel
from timenet.types.access import Access
from timenet.types.annotations import AnnotationDescriptor
from timenet.types.domains import Domain
from timenet.types.licenses import License
from timenet.types.metadata_fields import DatasetId, NonEmptyText, SemanticVersion
from timenet.types.specs import TimeSeriesSpec
from timenet.types.tasks import Task


class DatasetMetadata(TimeFModel):
    """A dataset's descriptive identity: who it is, not what it emits.

    The dataset card holds these fields. ``dataset_version`` is the semantic version of the upstream
    source. ``yaml_schema_version`` is the version of the card's own field schema. ``dataset_id`` is an
    ``org/name`` pair in HuggingFace style with exactly one slash. Ids are case-sensitive, so avoid
    casing-only differences on case-insensitive filesystems.
    """

    dataset_id: DatasetId
    """HuggingFace-style ``org/name`` pair, case-sensitive, exactly one slash."""
    dataset_version: SemanticVersion
    """Semantic version of the upstream source data."""
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
    """Where to read the full license text. Required when ``license`` is :attr:`License.OTHER`."""
    citation: StrictStr | None = None
    """How to cite the dataset, when the source asks for attribution."""
    access: Access = Access.OPEN
    """How a user obtains the data (see :class:`Access`). ``OPEN`` needs nothing."""
    access_url: StrictStr | None = None
    """Where to obtain access (the DUA or credentialing page). Required when ``access`` is not ``OPEN``."""
    yaml_schema_version: Annotated[int, Field(strict=True, ge=1, le=1)] = 1
    """Version of the card's own field schema."""

    @field_validator("name", "description")
    @classmethod
    def _non_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def _dependent_urls(self) -> Self:
        if self.license is License.OTHER and not self.license_url:
            raise ValueError("license_url is required when license is License.OTHER")
        if self.access is not Access.OPEN and not self.access_url:
            raise ValueError(f"access_url is required when access is {self.access.value!r}")
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
        from timenet.errors import TimeNetInvalidCardError  # noqa: PLC0415

        card_path = Path(path)
        try:
            import yaml  # noqa: PLC0415
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


@dataclass(frozen=True)
class DatasetSchema:
    """A dataset's type declaration, derived from its data (never hand-authored).

    It holds flat descriptor instances for specs and annotations. It also holds the real built-in
    :class:`~timenet.types.tasks.Task` subclasses, resolved against the registry instead of reconstructed.
    """

    time_series_specs: tuple[TimeSeriesSpec, ...] = ()
    """Descriptors for the dataset's measurement modalities."""
    annotations: tuple[AnnotationDescriptor, ...] = ()
    """Type-level descriptors for the dataset's annotation keys."""
    tasks: tuple[type[Task], ...] = field(default=())
    """Built-in ``Task`` subclasses the dataset declares, resolved from the registry."""
