"""The provenance record: where and when each release in the local registry was downloaded or built.

:class:`~timenet.registry.LocalRegistry` writes and reads the record of each release. The record stays in the
local registry when the release is copied elsewhere.
"""

from datetime import UTC, datetime
from enum import Enum
from urllib.parse import urlsplit, urlunsplit

from pydantic import AwareDatetime, BaseModel, ConfigDict, field_validator


class OriginKind(Enum):
    """How a release reached the local registry."""

    DOWNLOAD = "download"
    """A copy from a registry."""
    BUILD = "build"
    """A build by an installed connector."""


class Origin(BaseModel):
    """The kind, source, and time of a stored release's download or build."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: OriginKind
    """How the release reached the local registry."""
    source: str
    """The registry location, or the name and version of the backend package."""
    at: AwareDatetime
    """The time that the release reached the local registry, in UTC."""

    @field_validator("at")
    @classmethod
    def _in_utc(cls, value: datetime) -> datetime:
        """Return the time in UTC."""
        return value.astimezone(UTC)


def without_credentials(location: str) -> str:
    """Remove the user info, the query and the fragment of a URL.

    A string that is not a URL with a host, such as a path, comes back unchanged.

    Args:
        location: A URL, a path, or another name.

    Returns:
        The location without credentials.
    """
    parts = urlsplit(location)
    if not (parts.scheme and parts.netloc):
        return location
    host = parts.netloc.rpartition("@")[2]
    return urlunsplit((parts.scheme, host, parts.path, "", ""))
