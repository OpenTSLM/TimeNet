"""Dataset-level descriptive identity and derived type declaration."""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

from timenet.errors import TimeFValidationError
from timenet.types.access import Access
from timenet.types.annotations import AnnotationDescriptor
from timenet.types.domains import Domain
from timenet.types.licenses import License
from timenet.types.specs import TimeSeriesSpec
from timenet.types.tasks import Task
from timenet.types.version import Version


# A HuggingFace-style ``org/name`` pair: exactly one slash, no leading/trailing/empty segment.
_DATASET_ID = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def validate_dataset_id(dataset_id: str) -> None:
    """Check that a dataset id is a safe ``org/name`` pair.

    The registry, the writer, and the download cache join ids into filesystem paths. This check is the
    only gate that stops an id from naming a location outside its root.

    Args:
        dataset_id: The id to check.

    Raises:
        TimeFValidationError: If the id is not a single-slash ``org/name`` pair, or has a segment
            starting with ``.``.
    """
    if not _DATASET_ID.match(dataset_id) or any(part.startswith(".") for part in dataset_id.split("/")):
        raise TimeFValidationError(
            f"dataset_id must be 'org/name' (letters, digits, ., _, -; exactly one slash; no "
            f"segment may start with '.'), got {dataset_id!r}"
        )


def _str_tuple(value: Any, key: str) -> tuple[str, ...]:
    # tuple("abc") returns ("a", "b", "c"). A bare string where callers expect a list passes as
    # corrupt data. Require an actual list or tuple instead. Callers wrap the TypeError.
    if not isinstance(value, list | tuple):
        raise TypeError(f"{key!r} must be a list, got {type(value).__name__}")
    return tuple(value)


@dataclass(frozen=True)
class DatasetMetadata:
    """A dataset's descriptive identity: who it is, not what it emits.

    The dataset card holds these fields. ``dataset_version`` is the semantic version of the upstream
    source. ``yaml_schema_version`` is the version of the card's own field schema. ``dataset_id`` is an
    ``org/name`` pair in HuggingFace style with exactly one slash. Ids are case-sensitive, so avoid
    casing-only differences on case-insensitive filesystems.
    """

    dataset_id: str
    """HuggingFace-style ``org/name`` pair, case-sensitive, exactly one slash."""
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
    license_url: str | None = None
    """Where to read the full license text. Required when ``license`` is :attr:`License.OTHER`."""
    citation: str | None = None
    """How to cite the dataset, when the source asks for attribution."""
    access: Access = Access.OPEN
    """How a user obtains the data (see :class:`Access`). ``OPEN`` needs nothing."""
    access_url: str | None = None
    """Where to obtain access (the DUA or credentialing page). Required when ``access`` is not ``OPEN``."""
    yaml_schema_version: int = 1
    """Version of the card's own field schema."""

    def __post_init__(self) -> None:
        """Validate and coerce typed fields.

        Coerces ``license``, ``domains``, and ``dataset_version`` from plain strings to their enum
        or value types when passed as strings, matching what :meth:`from_dict` does.

        Raises:
            TimeFValidationError: If a typed field cannot be coerced, ``license`` is
                ``License.OTHER`` without a ``license_url``, or ``access`` is not open without an
                ``access_url``.
        """
        validate_dataset_id(self.dataset_id)
        if isinstance(self.license, str) and not isinstance(self.license, License):
            try:
                object.__setattr__(self, "license", License(self.license))
            except ValueError as exc:
                raise TimeFValidationError(f"invalid license {self.license!r}") from exc
        if isinstance(self.dataset_version, str):
            object.__setattr__(self, "dataset_version", Version.parse(self.dataset_version))
        if self.domains and not isinstance(self.domains[0], Domain):
            try:
                object.__setattr__(self, "domains", tuple(Domain(d) for d in self.domains))
            except ValueError as exc:
                raise TimeFValidationError(f"invalid domain in {self.domains!r}") from exc
        if self.license is License.OTHER and not self.license_url:
            raise TimeFValidationError("license_url is required when license is License.OTHER")
        if self.access is not Access.OPEN and not self.access_url:
            raise TimeFValidationError(f"access_url is required when access is {self.access.value!r}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetMetadata":
        """Build metadata from a plain mapping of card fields.

        The enum and version fields arrive as strings (``license``, ``domains``, ``dataset_version``).
        This method coerces them and ignores unmodeled keys. :meth:`from_yaml` and the manifest codec
        share it, so the mapping lives in one place. Coercion can raise ``KeyError`` for a missing field
        or ``ValueError`` for a bad license, domain, or version. Callers wrap these in their own error
        type.

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
            domains=tuple(Domain(domain) for domain in _str_tuple(data.get("domains", ()), "domains")),
            tags=_str_tuple(data.get("tags", ()), "tags"),
            source_url=data.get("source_url"),
            license_url=data.get("license_url"),
            citation=data.get("citation"),
            access=Access(data.get("access", Access.OPEN)),
            access_url=data.get("access_url"),
            yaml_schema_version=data.get("yaml_schema_version", 1),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetMetadata":
        """Load and validate a dataset card YAML into metadata.

        This method validates the card against the packaged ``dataset-card.schema.json`` before
        construction. Authoring mistakes then surface as clear, aggregated messages instead of a stack
        trace from deep inside coercion. PyYAML and jsonschema are optional. This method imports them
        lazily, so the types package does not depend on them. Install the ``timenet[build]`` extra
        to use this.

        Args:
            path: Path to the card YAML file.

        Returns:
            The constructed :class:`DatasetMetadata`.

        Raises:
            TimeNetInvalidCardError: If the build extra is missing, or the card is unreadable, is not a
                mapping, fails schema validation, or has an invalid field value.
        """
        from timenet.errors import TimeNetInvalidCardError  # noqa: PLC0415

        card_path = Path(path)
        try:
            import jsonschema  # noqa: PLC0415
            import yaml  # noqa: PLC0415
        except ModuleNotFoundError as exc:
            raise TimeNetInvalidCardError(
                "reading a dataset card needs PyYAML and jsonschema; install the 'timenet[build]' extra"
            ) from exc

        from timenet.schemas import DATASET_CARD_SCHEMA  # noqa: PLC0415

        try:
            raw = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise TimeNetInvalidCardError(f"could not read dataset card {card_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise TimeNetInvalidCardError(f"dataset card {card_path} must be a YAML mapping, got {type(raw).__name__}")

        errors = sorted(
            jsonschema.Draft202012Validator(DATASET_CARD_SCHEMA).iter_errors(raw),
            key=lambda error: list(error.path),
        )
        if errors:
            detail = "; ".join(
                f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}" for error in errors
            )
            raise TimeNetInvalidCardError(f"dataset card {card_path} failed validation: {detail}")

        try:
            return cls.from_dict(raw)
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            raise TimeNetInvalidCardError(f"dataset card {card_path} is invalid: {exc}") from exc


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
