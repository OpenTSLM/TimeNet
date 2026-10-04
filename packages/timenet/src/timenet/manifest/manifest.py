"""The :class:`Manifest`: the compiled ``manifest.json`` file and its JSON codec."""

from typing import Annotated, Any, Self

from pydantic import Field, StrictStr, model_validator
from typing_extensions import TypedDict

from timenet.manifest.counts import ManifestCounts
from timenet.manifest.files import ManifestFiles
from timenet.types import DatasetMetadata, DatasetSchema
from timenet.types._metadata_model import DatasetId
from timenet.types._model import TimeFModel
from timenet.types._wire import ValueEncoding
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest to a validated JSON-compatible dict with all keys present.

        Returns:
            The canonical dict form.
        """
        validated = type(self).model_validate(self)
        return validated.model_dump(mode="json", by_alias=True)

    def to_json(self) -> str:
        """Serialize the manifest to a pretty JSON string.

        Returns:
            The JSON text.
        """
        validated = type(self).model_validate(self)
        return validated.model_dump_json(indent=2, by_alias=True)
