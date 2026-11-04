"""Producer-side shortcuts to build and load a dataset from local code.

:func:`build` runs a connector through the engine into the shared local registry. :func:`load` runs
build and reads the result back. The writer and reader stacks are heavy, so they load lazily. This
keeps the import of :mod:`timenet_connectors` cheap.
"""

from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from timenet.dataset import TimeFDataset
    from timenet.registry import BaseRegistry


def build(  # ruff: ignore[too-many-arguments]
    dataset_id: str,
    *,
    version: str | None = None,
    out: str | Path | None = None,
    force: bool = False,
    keep_cache: bool = False,
    fetch_from: "str | BaseRegistry | None" = None,
    build_parents: bool = False,
) -> Path:
    # ruff: ignore[docstring-extraneous-exception] (raised by build_release, not directly here)
    """Build a dataset release into a local registry by its dataset ID.

    The build reuses a release that the registry already holds, unless you set ``force``. The
    connector runs in an environment built from its ``requirements.txt``, or in this interpreter if
    ``TIMENET_ISOLATION=off``.

    Args:
        dataset_id: The dataset ID, exactly as the card of the connector declares it (``org/name``).
        version: The version that the card must declare. ``None`` accepts any version.
        out: The local registry directory. Defaults to the shared one that ``TimeNet`` reads.
        force: Build the release again, even if the registry already holds it.
        keep_cache: Keep the raw download cache. A successful build deletes it by default.
        fetch_from: The registry, or its URI, to copy missing parent releases from.
        build_parents: Build missing parent releases from their installed connectors.

    Returns:
        The version directory of the release.

    Raises:
        TimeNetDatasetNotFoundError: If no installed connector builds ``dataset_id``.
        TimeNetVersionUnavailableError: If ``version`` is set and the card declares another version.
        TimeNetMissingParentError: If a parent release is missing. Its hint names the two keywords.
    """
    from timenet.builders import BuildOptions, declared  # ruff: ignore[import-outside-top-level]
    from timenet.coordinator.build import build_release  # ruff: ignore[import-outside-top-level]
    from timenet.errors import TimeNetMissingParentError  # ruff: ignore[import-outside-top-level]
    from timenet.registry import (  # ruff: ignore[import-outside-top-level]
        LocalRegistry,
        default_registry_path,
        open_registry,
    )
    from timenet.registry._paths import release_dir  # ruff: ignore[import-outside-top-level, import-private-name]

    release = declared(dataset_id).release
    root = Path(out).expanduser() if out is not None else default_registry_path()
    try:
        build_release(
            f"{dataset_id}@{version}" if version else dataset_id,
            store=LocalRegistry(root),
            fetch_from=open_registry(fetch_from) if isinstance(fetch_from, str) else fetch_from,
            build_parents=build_parents,
            force=force,
            options=BuildOptions(keep_cache=keep_cache),
        )
    except TimeNetMissingParentError as error:
        error.hint = "Copy it with fetch_from=<registry> or build it with build_parents=True."
        raise
    return release_dir(root, release)


def load(
    dataset_id: str,
    version: str | None = None,
    *,
    fetch_from: "str | BaseRegistry | None" = None,
    build_parents: bool = False,
) -> "TimeFDataset":
    """Build a dataset if needed, then load it into memory.

    This runs :func:`build`, then :meth:`timenet.client.TimeNet.load`.

    Args:
        dataset_id: The dataset ID (``org/name``).
        version: The version to load, which the card must declare, or ``None`` for the latest.
        fetch_from: The registry, or its URI, to copy missing parents and payload sources from.
        build_parents: Build missing parent releases from their installed connectors.

    Returns:
        The loaded dataset.
    """
    from timenet.client import TimeNet  # ruff: ignore[import-outside-top-level]

    build(dataset_id, version=version, fetch_from=fetch_from, build_parents=build_parents)
    return TimeNet().load(dataset_id, version, fetch_from=fetch_from)
