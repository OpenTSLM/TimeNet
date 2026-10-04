"""Private Pydantic models for hosted-registry API responses."""

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator

from timenet.format.constants import check_relative_path
from timenet.types import Access, DatasetMetadata, Domain, License
from timenet.types._metadata_model import DatasetId, NonEmptyText, SemanticVersion


class _ApiModel(BaseModel):
    """Strict known fields with forward-compatible unknown fields."""

    model_config = ConfigDict(extra="ignore", frozen=True)


class _DatasetSummaryModel(_ApiModel):
    """One dataset summary from the catalog API."""

    dataset_id: DatasetId
    version: SemanticVersion
    name: NonEmptyText
    description: NonEmptyText
    license: License
    license_url: StrictStr | None = None
    domains: list[Domain] = Field(default_factory=list)
    tags: list[StrictStr] = Field(default_factory=list)
    access: Access = Access.OPEN
    access_url: StrictStr | None = None

    def requires_manifest(self) -> bool:
        """Return whether dependent metadata is absent from this summary."""
        return (self.license is License.OTHER and not self.license_url) or (
            self.access is not Access.OPEN and not self.access_url
        )

    def metadata(self) -> DatasetMetadata:
        """Convert a complete summary to public dataset metadata.

        Returns:
            The summary's public metadata representation.
        """
        return DatasetMetadata(
            dataset_id=self.dataset_id,
            dataset_version=self.version,
            name=self.name,
            description=self.description,
            license=self.license,
            license_url=self.license_url,
            domains=tuple(self.domains),
            tags=tuple(self.tags),
            access=self.access,
            access_url=self.access_url,
        )


class _DatasetListModel(_ApiModel):
    """Catalog list response."""

    datasets: list[_DatasetSummaryModel]


class _DatasetDetailModel(_DatasetSummaryModel):
    """Dataset detail response used to resolve the latest version."""


class _PublishResponseModel(_ApiModel):
    """Publish response listing the files the service expects."""

    files: list[StrictStr]

    @field_validator("files")
    @classmethod
    def _safe_paths(cls, value: list[str]) -> list[str]:
        for path in value:
            if not path:
                raise ValueError("publish paths must be non-empty")
            check_relative_path("publish path", path)
        return value

    @model_validator(mode="after")
    def _unique_paths(self) -> Self:
        if len(set(self.files)) != len(self.files):
            raise ValueError("publish paths must be unique")
        return self


HttpUrlString = Annotated[StrictStr, Field(pattern=r"^https?://[^\s]+$")]


class _UploadGrantModel(_ApiModel):
    """Presigned upload grant response."""

    url: HttpUrlString
    headers: dict[StrictStr, StrictStr]
