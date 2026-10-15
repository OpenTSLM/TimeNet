"""Physical payload locations independent of logical object identities."""

from typing import Annotated, Literal

from pydantic import AfterValidator, Field, StrictStr, field_validator

from timenet.format.constants import check_relative_path
from timenet.types._model import TimeFModel
from timenet.types.references import DatasetRef
from timenet.types.wire import Checksum


def _relative_path(value: str) -> str:
    check_relative_path("payload path", value)
    return value


class PayloadSource(TimeFModel):
    """An immutable physical owner pinned by its manifest digest."""

    dataset: DatasetRef
    manifest_checksum: Checksum


class PayloadChunk(TimeFModel):
    """One physical chunk or text cell in an immutable artifact."""

    path: Annotated[StrictStr, AfterValidator(_relative_path)]
    major: Annotated[int, Field(strict=True, ge=0)]
    minor: Annotated[int | None, Field(strict=True, ge=0)] = None
    n_values: Annotated[int, Field(strict=True, ge=1)]

    def as_backend_row(self) -> dict[str, object]:
        """Return the existing array-backend coordinates.

        Returns:
            A backend locator row.
        """
        return {
            "chunk_file": self.path,
            "chunk_major_idx": self.major,
            "chunk_minor_idx": self.minor,
            "n_values": self.n_values,
        }


class PayloadReference(TimeFModel):
    """A direct storage location. An absent source means local ownership on disk."""

    source: PayloadSource | None = None
    codec: Literal["parquet", "zarr", "text"]
    prefix: StrictStr = ""
    chunks: Annotated[tuple[PayloadChunk, ...], Field(min_length=1)]

    @field_validator("prefix")
    @classmethod
    def _prefix(cls, value: str) -> str:
        if value:
            check_relative_path("payload prefix", value)
        return value
