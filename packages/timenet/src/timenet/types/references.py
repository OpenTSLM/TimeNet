"""Exact dataset releases and source-object provenance."""

from typing import Any, Literal

from pydantic import GetJsonSchemaHandler, model_serializer, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

from timenet.errors import TimeFValidationError
from timenet.types._model import TimeFModel
from timenet.types.metadata_fields import DatasetId, NonEmptyText, SemanticVersion


class DatasetRef(TimeFModel):
    """An exact TimeNet release, encoded as ``org/name@version``."""

    dataset_id: DatasetId
    version: SemanticVersion

    @model_validator(mode="before")
    @classmethod
    def _parse(cls, value: Any) -> Any:
        if isinstance(value, str):
            if value.count("@") != 1:
                raise TimeFValidationError("dataset reference must be org/name@version")
            dataset_id, version = value.split("@")
            return {"dataset_id": dataset_id, "version": version}
        return value

    @model_serializer(when_used="json")
    def _serialize(self) -> str:
        return str(self)

    @classmethod
    def __get_pydantic_json_schema__(  # noqa: PLW3201 - Pydantic schema hook
        cls, _schema: CoreSchema, _handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        r"""Describe an exact release's wire representation.

        The pattern matches ``org/name@major.minor.patch``. ``^`` and ``$`` anchor
        the reference at the start and end of the string. Both ``org`` and ``name``
        start with a letter, digit, underscore, or hyphen. Their remaining characters
        may also include dots. ``/`` and ``@`` separate the identity components.
        Each ``\d+`` accepts one or more version digits, and ``\.`` matches a
        literal dot. Version ranges and labels such as ``latest`` do not match.

        Returns:
            The reference string schema.
        """
        return {
            "type": "string",
            "pattern": r"^[A-Za-z0-9_-][A-Za-z0-9._-]*/[A-Za-z0-9_-][A-Za-z0-9._-]*@\d+\.\d+\.\d+$",
        }

    def __str__(self) -> str:
        return f"{self.dataset_id}@{self.version}"


ObjectKind = Literal["record", "source", "signal", "axis", "annotation_content", "annotation_occurrence", "task"]


class ObjectRef(TimeFModel):
    """A source object used as provenance, without asserting unchanged content."""

    dataset: DatasetRef
    kind: ObjectKind
    id: NonEmptyText


class TaskRef(TimeFModel):
    """The identity of a task in the same catalog, without retaining its body."""

    id: NonEmptyText


class AnnotationContentRef(TimeFModel):
    """An explicit reference to annotation content in the same catalog."""

    id: NonEmptyText


def validate_task_graph(parents: dict[str, set[str]]) -> None:
    """Reject missing parents and cycles using only task identities.

    Raises:
        TimeFValidationError: If a derivation graph contains a cycle or missing task.
    """
    children: dict[str, set[str]] = {identity: set() for identity in parents}
    counts = {identity: len(values) for identity, values in parents.items()}
    for identity, values in parents.items():
        for parent in values:
            if parent == identity:
                raise TimeFValidationError(f"task {identity!r} depends on itself")
            if parent not in parents:
                raise TimeFValidationError(f"task {identity!r} depends on missing task {parent!r}")
            children[parent].add(identity)
    ready = [identity for identity, count in counts.items() if count == 0]
    while ready:
        for child in children[ready.pop()]:
            counts[child] -= 1
            if counts[child] == 0:
                ready.append(child)
    cyclic = [identity for identity, count in counts.items() if count]
    if cyclic:
        raise TimeFValidationError(f"task dependencies contain a cycle through {sorted(cyclic)}")
