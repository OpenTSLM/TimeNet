"""The :class:`Manifest`: the compiled ``manifest.json`` file and its JSON codec."""

import json
from typing import Annotated, Any, Self

from pydantic import Field, StrictStr, ValidationError, model_validator
from typing_extensions import TypedDict

from timenet.errors import TimeFFormatError
from timenet.format.constants import TIMEF_FORMAT_VERSION
from timenet.manifest.counts import ManifestCounts
from timenet.manifest.files import ManifestFiles
from timenet.types import DatasetMetadata, DatasetSchema
from timenet.types._model import TimeFModel
from timenet.types.metadata_fields import DatasetId
from timenet.types.payload_references import PayloadSource
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
    payload_sources: tuple[PayloadSource, ...] = ()
    """Flat pinned physical owners needed by this catalog, independent of build inputs."""
    counts: ManifestCounts = Field(default_factory=ManifestCounts)
    """Row and entity counts recorded for quick inspection."""
    values_backend: ValuesBackend = ValuesBackend.PARQUET
    """Storage backend for the time-series values plane."""
    value_encoding: dict[StrictStr, ValueEncoding] = Field(default_factory=dict)
    """``spec_type`` to the values-column encoding used by its shards."""
    build_env: BuildEnvironment = Field(default_factory=dict)
    """The Python version and package set that produced this version."""
    timef_format_version: Annotated[int, Field(strict=True, ge=3, le=3)]
    """The TimeF manifest format version."""

    @model_validator(mode="after")
    def _matching_dataset_id(self) -> Self:
        if self.dataset_id != self.metadata.dataset_id:
            raise ValueError(
                f"manifest dataset_id {self.dataset_id!r} does not match "
                f"metadata.dataset_id {self.metadata.dataset_id!r}"
            )
        sources = [source.dataset for source in self.payload_sources]
        if len(sources) != len(set(sources)):
            raise ValueError("payload sources must have unique release identities")
        if any(
            source.dataset.dataset_id == self.dataset_id and source.dataset.version == self.metadata.dataset_version
            for source in self.payload_sources
        ):
            raise ValueError("a release cannot depend on itself as a foreign payload source")
        return self

    @classmethod
    def from_artifact(cls, value: str | bytes | dict[str, Any]) -> "Manifest":
        """Read an on-disk manifest with explicit format and corruption errors.

        Returns:
            The validated manifest.

        Raises:
            TimeFFormatError: If the format is unsupported or the artifact is corrupt.
        """
        try:
            data = json.loads(value) if isinstance(value, str | bytes) else value
        except ValueError as exc:
            raise TimeFFormatError(f"invalid TimeF manifest JSON: {exc}") from exc
        if (
            not isinstance(data, dict)
            or type(data.get("timef_format_version")) is not int
            or data["timef_format_version"] != TIMEF_FORMAT_VERSION
        ):
            found = data.get("timef_format_version") if isinstance(data, dict) else None
            raise TimeFFormatError(
                f"unsupported TimeF format version {found!r}; rebuild the artifact with format {TIMEF_FORMAT_VERSION}"
            )
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise TimeFFormatError(f"invalid TimeF manifest: {exc}") from exc

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
