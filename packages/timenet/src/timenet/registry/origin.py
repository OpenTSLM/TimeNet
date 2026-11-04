"""The provenance record: where and when each release in the local registry was copied or built.

Each release stores the record in ``.origin.json``. The record stays in the local registry when the release
is copied elsewhere.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal
from urllib.parse import urlsplit, urlunsplit
import uuid

from pydantic import AwareDatetime, BaseModel, ConfigDict, ValidationError

from timenet.errors import TimeFFormatError, TimeNetDatasetNotFoundError
from timenet.format.constants import MANIFEST_FILE
from timenet.registry._paths import release_dir
from timenet.registry.local import LocalRegistry
from timenet.types.references import DatasetRef


ORIGIN_FILE = ".origin.json"


@dataclass(frozen=True)
class Origin:
    """The source and time of a stored release's copy or build.

    Attributes:
        origin: ``"copy"`` for a registry copy or ``"build"`` for a backend build.
        source: The registry location or backend package name and version.
        at: The time the release reached the local registry, in UTC.
    """

    origin: Literal["copy", "build"]
    source: str
    at: datetime


class _Record(BaseModel):
    """The JSON record with only the ``origin``, ``source``, and ``at`` fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    origin: Literal["copy", "build"]
    source: str
    at: AwareDatetime


def record_origin(store: LocalRegistry, release: DatasetRef, origin: Literal["copy", "build"], source: str) -> None:
    """Record a stored release's source and the current time.

    The record replaces an earlier one. A URL in ``source`` loses its user info, query and fragment, so no
    credential lands in the record.

    Args:
        store: The local registry that holds the release.
        release: The release.
        origin: ``"copy"`` or ``"build"``.
        source: The registry location or backend package name and version.

    Raises:
        TimeNetDatasetNotFoundError: If the local registry does not hold the release.
    """
    directory = release_dir(store.root, release)
    if not (directory / MANIFEST_FILE).is_file():
        raise TimeNetDatasetNotFoundError(f"the local registry holds no {release}, so it gets no provenance record")
    record = _Record(origin=origin, source=without_credentials(source), at=datetime.now(UTC).replace(microsecond=0))
    # Write next to the record, then rename, so a reader never sees half a record.
    staging = directory / f"{ORIGIN_FILE}.tmp-{uuid.uuid4().hex}"
    try:
        staging.write_text(record.model_dump_json() + "\n")
        staging.replace(directory / ORIGIN_FILE)
    finally:
        staging.unlink(missing_ok=True)


def read_origin(store: LocalRegistry, release: DatasetRef) -> Origin | None:
    """Read the provenance record of a release in the local registry.

    A release without a record counts as a build.

    Args:
        store: The local registry.
        release: The release.

    Returns:
        The record, or ``None`` if it is missing.

    Raises:
        TimeFFormatError: If the record is invalid.
    """
    path = release_dir(store.root, release) / ORIGIN_FILE
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    try:
        record = _Record.model_validate_json(text)
    except ValidationError as error:
        problems = "; ".join(f"{'.'.join(map(str, item['loc'])) or 'record'}: {item['msg']}" for item in error.errors())
        raise TimeFFormatError(
            f"{path} is not a valid provenance record ({problems}). Delete it to count the release as built."
        ) from error
    return Origin(record.origin, record.source, record.at.astimezone(UTC))


def without_credentials(location: str) -> str:
    """Remove user information, the query, and the fragment from a URL.

    Return strings unchanged unless they have both a URL scheme and a host.

    Args:
        location: A URL, path, or name.

    Returns:
        The URL without user information, query, or fragment, or the original non-URL string.
    """
    parts = urlsplit(location)
    if not (parts.scheme and parts.netloc):
        return location
    host = parts.netloc.rpartition("@")[2]
    return urlunsplit((parts.scheme, host, parts.path, "", ""))
