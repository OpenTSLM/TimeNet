"""Dataset-level descriptive identity and derived type declaration."""

from pathlib import Path
from typing import Annotated, Any, Self

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
from timenet.types.references import DatasetRef
from timenet.types.specs import TimeSeriesSpec
from timenet.types.tasks import TASKS, Task, TaskType


class DatasetMetadata(TimeFModel):
    """A dataset's descriptive identity: who it is, not what it emits."""

    dataset_id: DatasetId
    """HuggingFace-style ``org/name`` pair, case-sensitive, exactly one slash."""
    dataset_version: SemanticVersion
    """Semantic version of this immutable TimeNet dataset release."""
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
    source_revision: StrictStr | None = None
    """Upstream release or revision from which this TimeNet release was built."""
    license_url: StrictStr | None = None
    """Where to read the full license text. Required when ``license`` is :attr:`License.OTHER`."""
    citation: StrictStr | None = None
    """How to cite the dataset, when the source asks for attribution."""
    access: Access = Access.OPEN
    """How a user obtains the data (see :class:`Access`). ``OPEN`` needs nothing."""
    access_url: StrictStr | None = None
    """Where to obtain access. Required when ``access`` is not ``OPEN``."""
    parents: tuple[DatasetRef, ...] = ()
    """Exact parent versions, addressed by their full dataset IDs."""
    yaml_schema_version: Annotated[int, Field(strict=True, ge=1, le=1)] = 1
    """Version of the card's own field schema."""

    @field_validator("name", "description")
    @classmethod
    def _non_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def _dependent_fields(self) -> Self:
        if self.license is License.OTHER and not self.license_url:
            raise ValueError("license_url is required when license is License.OTHER")
        if self.access is not Access.OPEN and not self.access_url:
            raise ValueError(f"access_url is required when access is {self.access.value!r}")
        parent_ids = [parent.dataset_id for parent in self.parents]
        if len(parent_ids) != len(set(parent_ids)):
            raise TimeFValidationError("parent dataset IDs must be unique")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetMetadata":
        """Load and validate a dataset card YAML into metadata.

        Authoring mistakes surface as aggregated validation messages. PyYAML is imported lazily,
        so install the ``timenet[build]`` extra to use this method.

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
    """A dataset's type declaration, derived from its data (never hand-authored)."""

    time_series_specs: tuple[TimeSeriesSpec, ...] = ()
    """Descriptors for the dataset's measurement modalities."""
    annotations: tuple[AnnotationDescriptor, ...] = ()
    """Type-level descriptors for the dataset's annotation keys."""
    tasks: tuple[TaskDeclaration, ...] = ()
    """Built-in ``Task`` subclasses the dataset declares, resolved from the registry."""
