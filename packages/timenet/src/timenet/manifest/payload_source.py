"""Pins on the payload files that a release reads from other releases."""

from typing import Annotated, Self

from pydantic import Field, model_validator

from timenet.manifest.files import FilePart
from timenet.types._model import TimeFModel
from timenet.types.references import DatasetRef


class PayloadSource(TimeFModel):
    """A release whose payload files this release reads, pinned to the files it reuses.

    The pins copy the source's own manifest records, so checking that the source is still the
    release this one was written against needs only the two manifests, no hashing.
    """

    release: DatasetRef
    """The exact release that stores the files."""
    files: Annotated[tuple[FilePart, ...], Field(min_length=1)]
    """The source's manifest records for the files this release reads, in path order."""

    @model_validator(mode="after")
    def _distinct_paths(self) -> Self:
        paths = [part.path for part in self.files]
        if len(set(paths)) != len(paths):
            raise ValueError(f"payload source {self.release} pins one file more than once")
        return self
