"""The :func:`open_registry` factory: dispatch a URI to the right registry backend."""

from pathlib import Path

from timenet._registry_uri import registry_location
from timenet.registry.local import LocalRegistry
from timenet.registry.remote import RemoteRegistry
from timenet.registry.s3 import S3Registry
from timenet.registry.writable import WritableRegistry


def open_registry(
    uri: str | Path, *, local_registry: str | Path | None = None, label: str = "registry"
) -> WritableRegistry:
    """Open a registry from a URI or path.

    The scheme selects the backend. ``http(s)://`` and ``timenet://`` open a :class:`RemoteRegistry`,
    and ``timenet://`` is an alias for the hosted :data:`TIMENET_REGISTRY_URL`. ``s3://`` opens an
    :class:`S3Registry`. ``file://`` or a plain path opens a :class:`LocalRegistry`.

    Args:
        uri: A URL, ``timenet://`` / ``s3://`` / ``file://`` URI, or local path.
        local_registry: A directory of complete releases that a remote backend reuses. The local backend ignores
            it. With ``None``, a remote backend reads every file remotely.
        label: The name of the value in an error message, such as ``sources[0]``.

    Returns:
        The matching registry backend.

    Raises:
        TimeNetRegistryError: If ``uri`` is blank or carries a scheme no backend handles, if a
            ``timenet://`` URI carries a path, or if ``uri`` is a ``file://`` URI with a host component
            (the host would be dropped without notice) or without a path.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by registry_location)
    location = registry_location(uri, label=label)
    if isinstance(location, Path):
        return LocalRegistry(location)
    if location.startswith("s3://"):
        return S3Registry(location, local_registry=local_registry)
    return RemoteRegistry(location, local_registry=local_registry)
