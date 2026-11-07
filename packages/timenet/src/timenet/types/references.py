"""Exact dataset releases used by declared parents and payload sources."""

from pydantic import ValidationError

from timenet.errors import TimeFValidationError
from timenet.types._model import TimeFModel
from timenet.types.metadata_fields import DatasetId, SemanticVersion
from timenet.types.version import Version


LATEST_VERSION = "latest"
"""Reference identifier that selects the newest committed dataset version across all sources."""


def is_latest(version: str | None) -> bool:
    """Return whether a requested version selects the newest committed release.

    ``None``, an empty string, and ``"latest"`` all mean the request pins no version.

    Returns:
        Whether the request leaves the version unpinned.
    """
    return version in {None, "", LATEST_VERSION}


def split_ref(ref: str) -> tuple[str, str | None]:
    """Split a dataset reference into its id and requested version.

    ``"org/id@1.0.0"`` -> ``("org/id", "1.0.0")``. ``"org/id@latest"`` -> ``("org/id", "latest")``. ``"org/id"`` ->
    ``("org/id", None)``.

    Args:
        ref: A dataset id, optionally suffixed with ``@<version>`` or ``@latest``.

    Returns:
        The bare dataset id and the version string, ``"latest"``, or ``None`` without a version.

    Raises:
        TimeFValidationError: If ``ref`` has more than one ``@``, an empty id or version, or a version
            part that is neither ``latest`` nor a valid ``major.minor.patch``.
    """
    if ref.count("@") > 1:
        raise TimeFValidationError(f"dataset ref must contain at most one '@', got {ref!r}")
    dataset_id, separator, version = ref.partition("@")
    if not dataset_id:
        raise TimeFValidationError(f"dataset ref has an empty id: {ref!r}")
    if not separator:
        return dataset_id, None
    if not version:
        raise TimeFValidationError(f"dataset ref has an empty version: {ref!r}")
    if version != LATEST_VERSION:
        Version.parse(version)  # Validate the pin. This raises TimeFValidationError on a malformed version.
    return dataset_id, version


class DatasetRef(TimeFModel):
    """An exact release, independent of its registry location."""

    dataset_id: DatasetId
    version: SemanticVersion

    def __str__(self) -> str:
        return f"{self.dataset_id}@{self.version}"

    @classmethod
    def parse(cls, value: str) -> "DatasetRef":
        """Parse an exact ``org/name@version`` reference.

        Returns:
            The validated reference.

        Raises:
            TimeFValidationError: If the reference is malformed or does not contain an exact version.
        """
        dataset_id, version = split_ref(value)
        if is_latest(version):
            raise TimeFValidationError(f"an exact dataset release is required: {value!r}")
        try:
            return cls.model_validate({"dataset_id": dataset_id, "version": version})
        except ValidationError as error:
            raise TimeFValidationError(f"invalid dataset reference {value!r}: {error}") from error
