"""Stable references to dataset versions and objects in composition graphs."""

from dataclasses import dataclass
from enum import StrEnum, unique
import re
from urllib.parse import quote, unquote

from timenet.errors import TimeFValidationError
from timenet.types.version import Version


_ALIAS = re.compile(r"^[a-z][a-z0-9_-]*$")


@dataclass(frozen=True, order=True)
class DatasetRef:
    """An exact, immutable dataset-version reference."""

    dataset_id: str
    version: Version

    def __post_init__(self) -> None:
        """Coerce the version and validate the dataset id."""
        from timenet.types.metadata import validate_dataset_id  # noqa: PLC0415

        validate_dataset_id(self.dataset_id)
        if isinstance(self.version, str):
            object.__setattr__(self, "version", Version.parse(self.version))

    @classmethod
    def parse(cls, value: str) -> "DatasetRef":
        """Parse an ``org/name@version`` reference.

        Args:
            value: Canonical dataset reference.

        Returns:
            The parsed reference.

        Raises:
            TimeFValidationError: If the reference has no exact version.
        """
        dataset_id, separator, version = value.rpartition("@")
        if not separator or not dataset_id or not version:
            raise TimeFValidationError(f"dataset reference must be 'org/name@major.minor.patch', got {value!r}")
        return cls(dataset_id=dataset_id, version=Version.parse(version))

    def __str__(self) -> str:
        """Return the canonical ``org/name@version`` representation."""
        return f"{self.dataset_id}@{self.version}"


@dataclass(frozen=True)
class ParentDataset:
    """A named, exact parent declared by a dataset card."""

    alias: str
    dataset: DatasetRef

    def __post_init__(self) -> None:
        """Validate that the alias is safe for APIs and storage.

        Raises:
            TimeFValidationError: If the alias is not a safe identifier.
        """
        if not _ALIAS.fullmatch(self.alias):
            raise TimeFValidationError(
                "parent alias must start with a lowercase letter and contain only "
                f"lowercase letters, digits, '_' or '-', got {self.alias!r}"
            )


@unique
class ObjectKind(StrEnum):
    """Kinds of TimeF objects that can cross dataset boundaries."""

    RECORD = "record"
    SOURCE = "source"
    SIGNAL = "signal"
    AXIS = "axis"
    ANNOTATION_CONTENT = "annotation_content"
    ANNOTATION_OCCURRENCE = "annotation_occurrence"
    TASK = "task"


@dataclass(frozen=True, order=True)
class ObjectRef:
    """A globally unambiguous reference to one object in one dataset version."""

    dataset: DatasetRef
    kind: ObjectKind
    object_id: str

    def __post_init__(self) -> None:
        """Coerce the object kind and require a non-empty id.

        Raises:
            TimeFValidationError: If the object kind or id is invalid.
        """
        if isinstance(self.kind, str):
            object.__setattr__(self, "kind", ObjectKind(self.kind))
        if not isinstance(self.object_id, str) or not self.object_id:
            raise TimeFValidationError("object_id must be a non-empty string")

    @classmethod
    def parse(cls, value: str) -> "ObjectRef":
        """Parse an ``org/name@version#kind:percent-encoded-id`` reference.

        Args:
            value: Canonical object reference.

        Returns:
            The parsed reference.

        Raises:
            TimeFValidationError: If the reference is malformed.
        """
        dataset_text, hash_separator, object_text = value.partition("#")
        kind_text, colon_separator, object_id = object_text.partition(":")
        if not hash_separator or not colon_separator or not object_id:
            raise TimeFValidationError(
                f"object reference must be 'org/name@major.minor.patch#kind:object-id', got {value!r}"
            )
        try:
            kind = ObjectKind(kind_text)
        except ValueError as exc:
            raise TimeFValidationError(f"unknown object kind {kind_text!r}") from exc
        return cls(
            dataset=DatasetRef.parse(dataset_text),
            kind=kind,
            object_id=unquote(object_id),
        )

    def __str__(self) -> str:
        """Return the canonical representation."""
        return f"{self.dataset}#{self.kind.value}:{quote(self.object_id, safe='')}"
