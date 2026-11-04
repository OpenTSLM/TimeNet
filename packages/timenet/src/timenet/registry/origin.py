"""The provenance record: where a release in the store comes from.

The record is the file ``.origin.json`` in the release directory, and a copy of the release never
carries it. It says whether the release is a copy or a build, where it comes from, and when.
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
    """Where a release in the store comes from.

    Attributes:
        origin: ``"copy"`` for a copy from a registry, ``"build"`` for a build by a backend.
        source: The location of the registry, or the backend package and its version.
        at: When the store got the release, in UTC.
    """

    origin: Literal["copy", "build"]
    source: str
    at: datetime


class _Record(BaseModel):
    """The JSON form of :class:`Origin`, with exactly its three keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    origin: Literal["copy", "build"]
    source: str
    at: AwareDatetime


def record_origin(store: LocalRegistry, release: DatasetRef, origin: Literal["copy", "build"], source: str) -> None:
    """Write the provenance record of a release in the store, with the current time.

    The record replaces an earlier one. A URL in ``source`` loses its user info, query and fragment,
    so no credential lands in the record.

    Args:
        store: The local registry that holds the release.
        release: The release.
        origin: ``"copy"`` or ``"build"``.
        source: The location of the registry, or the backend package and its version.

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

    Args:
        store: The local registry.
        release: The release.

    Returns:
        The record, or ``None`` if the release has none: such a release counts as built.

    Raises:
        TimeFFormatError: If the record is damaged.
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
