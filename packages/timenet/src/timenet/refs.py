"""Parse dataset references of the form ``org/id@version``.

A reference pins a version with ``@`` (``chengsenwang/tsqa@1.0.0``); with no ``@`` (or ``@latest``) it
resolves to the latest committed version.
"""

from timenet.errors import TimeFValidationError
from timenet.types import Version


def split_ref(ref: str) -> tuple[str, str | None]:
    """Split a dataset reference into its id and pinned version.

    ``"org/id@1.0.0"`` -> ``("org/id", "1.0.0")``; ``"org/id@latest"`` or ``"org/id"`` -> ``("org/id", None)``.

    Args:
        ref: A dataset id, optionally suffixed with ``@<version>`` or ``@latest``.

    Returns:
        The bare dataset id and the pinned version string, or ``None`` for latest.

    Raises:
        TimeFValidationError: If ``ref`` has more than one ``@``, an empty id or version, or a version
            part that is neither ``latest`` nor a valid ``major.minor.patch``.
    """
    if ref.count("@") > 1:
        raise TimeFValidationError(f"dataset ref must contain at most one '@', got {ref!r}")
    dataset_id, separator, version = ref.partition("@")
    if not dataset_id:
        raise TimeFValidationError(f"dataset ref has an empty id: {ref!r}")
    if not separator or version == "latest":
        return dataset_id, None
    if not version:
        raise TimeFValidationError(f"dataset ref has an empty version: {ref!r}")
    Version.parse(version)  # validate the pin; raises TimeFValidationError on a malformed version
    return dataset_id, version
