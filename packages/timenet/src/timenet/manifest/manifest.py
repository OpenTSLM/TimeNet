"""The :class:`Manifest`: the compiled ``manifest.json`` file and its JSON codec."""

from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import Field, StrictStr, ValidationError, model_validator
from typing_extensions import TypedDict

from timenet.errors import TimeFFormatError, TimeNetManifestError
from timenet.manifest.counts import ManifestCounts
from timenet.manifest.files import ManifestFiles
from timenet.manifest.payload_source import PayloadSource
from timenet.types import DatasetMetadata, DatasetSchema
from timenet.types._model import TimeFModel
from timenet.types.metadata_fields import DatasetId
from timenet.types.parent_dataset_card import ParentDatasetCard
from timenet.types.reference_keys import ParentAlias, SourceKeyField
from timenet.types.wire import ValueEncoding
from timenet.values_backends import ValuesBackend


class BuildEnvironment(TypedDict, total=False):
    """Build-provenance fields recorded in a manifest."""

    python: StrictStr
    packages: dict[StrictStr, StrictStr]


class Manifest(TimeFModel):
    """The single source of truth a consumer reads to interpret a dataset version."""

    dataset_id: DatasetId
    """A denormalized copy of ``metadata.dataset_id``."""
    metadata: DatasetMetadata
    """Descriptive identity of the dataset."""
    files: ManifestFiles
    """Descriptor for every data artifact, grouped by kind."""
    dataset_schema: Annotated[DatasetSchema, Field(alias="schema")] = Field(default_factory=DatasetSchema)
    """Structural schema: time-series specs, annotations, and tasks."""
    counts: ManifestCounts = Field(default_factory=ManifestCounts)
    """Row and entity counts recorded for quick inspection."""
    values_backend: ValuesBackend = ValuesBackend.PARQUET
    """Storage backend for the time-series values plane."""
    value_encoding: dict[StrictStr, ValueEncoding] = Field(default_factory=dict)
    """``spec_type`` to the values-column encoding used by its shards."""
    parent_cards: dict[ParentAlias, ParentDatasetCard] = Field(default_factory=dict)
    """Copies of the cards of the declared parents, indexed by parent alias."""
    lineage: tuple[ParentDatasetCard, ...] = ()
    """Flattened provenance snapshots; never traversed to read data."""
    payload_sources: dict[SourceKeyField, PayloadSource] = Field(default_factory=dict)
    """Every release that stores a payload file this release reads, by source key, pinned to those
    files. The writer flattens them, so a source's own sources are not listed unless this release
    reads their files too."""
    build_env: BuildEnvironment = Field(default_factory=dict)
    """The Python version and package set that produced this version."""
    timef_format_version: Annotated[int, Field(strict=True, ge=1, le=1)]
    """The TimeF manifest format version."""

    @model_validator(mode="after")
    def _matching_dataset_id(self) -> Self:
        if self.dataset_id != self.metadata.dataset_id:
            raise ValueError(
                f"manifest dataset_id {self.dataset_id!r} does not match "
                f"metadata.dataset_id {self.metadata.dataset_id!r}"
            )
        return self

    @model_validator(mode="after")
    def _source_keys_name_their_release(self) -> Self:
        for key, source in self.payload_sources.items():
            if key != str(source.release):
                raise ValueError(f"payload source key {key!r} does not name its release {source.release}")
        return self

    @model_validator(mode="after")
    def _one_parent_card_per_release(self) -> Self:
        aliases: dict[str, list[str]] = {}
        for alias, card in self.parent_cards.items():
            aliases.setdefault(str(card.source), []).append(alias)
        for release, names in aliases.items():
            if len(names) > 1:
                raise ValueError(
                    f"parent_cards hold the parent {release} under the parent aliases {sorted(names)}; "
                    "a manifest holds each parent release under one alias only"
                )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest to a validated JSON-compatible dict with all keys present.

        Returns:
            The canonical dict form.
        """
        # Frozen models still contain mutable dictionaries; validate before publishing them.
        validated = type(self).model_validate(self)
        return validated.model_dump(mode="json", by_alias=True)

    def to_json(self) -> str:
        """Serialize the manifest to a pretty JSON string.

        Returns:
            The JSON text.
        """
        # Apply the same validation as to_dict() to catch changes in nested dictionaries.
        validated = type(self).model_validate(self)
        return validated.model_dump_json(indent=2, by_alias=True)


def same_release(a: Manifest, b: Manifest) -> bool:
    """Compare two manifests, ignoring their build environments.

    Args:
        a: One manifest.
        b: The other manifest.

    Returns:
        Whether the manifests are equal apart from ``build_env``.
    """
    return a.model_copy(update={"build_env": {}}) == b.model_copy(update={"build_env": {}})


def parse_manifest(text: str | bytes, where: str) -> Manifest:
    """Parse the JSON text of a stored manifest.

    Args:
        text: The JSON text.
        where: The source location to include in errors.

    Returns:
        The manifest.

    Raises:
        TimeNetManifestError: If the text is not a valid manifest.
    """
    try:
        return Manifest.model_validate_json(text)
    except ValidationError as error:
        first = error.errors()[0]
        location = ".".join(str(part) for part in first["loc"])
        message = first["msg"].removeprefix("Value error, ")
        raise TimeNetManifestError(where, f"{location}: {message}" if location else message) from error


def read_manifest(path: Path) -> Manifest:
    """Read a ``manifest.json`` file.

    Args:
        path: The file.

    Returns:
        The manifest.

    Raises:
        TimeFFormatError: If the file cannot be read.
        TimeNetManifestError: If the file is not a valid manifest.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by parse_manifest
    try:
        text = path.read_bytes()
    except OSError as error:
        raise TimeFFormatError(f"cannot read the manifest {path}: {error.strerror or error}") from error
    return parse_manifest(text, str(path))
