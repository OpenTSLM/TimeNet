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

    This is the producer-side one-liner over the build coordinator,
    :func:`timenet.coordinator.build.build_release`. The build reuses a release that the registry
    already holds, unless the caller sets ``force``. The default output is the shared local registry
    (:func:`timenet.registry.default_registry_path`). The SDK reads from that directory, so a build
    here loads at once with ``TimeNet().load(dataset_id)``.

    Like ``timenet-build build``, this runs the connector in an environment built from its
    ``requirements.txt``. Set ``TIMENET_ISOLATION=off`` to run it in this interpreter instead, which
    is what you want while writing a connector. Missing parent releases fail the build before any
    download, unless ``fetch_from`` or ``build_parents`` obtains them.

    Args:
        dataset_id: The dataset ID, exactly as the card of the connector declares it (``org/name``).
        version: The expected dataset version. A connector builds only the version that its card
            declares, so this value is a guard. If it is another version, the build stops before
            anything runs. ``None`` builds the version that the card declares.
        out: Output registry directory. Defaults to the shared local registry.
        force: Build the release again, even if the registry already holds it.
        keep_cache: Keep the raw download cache. A successful build deletes it by default.
        fetch_from: The registry, or its URI, to copy missing parent releases from.
        build_parents: Build missing parent releases from their installed connectors.

    Returns:
        The version directory of the release.

    Raises:
        TimeNetDatasetNotFoundError: If no installed connector builds ``dataset_id``.
        TimeNetVersionUnavailableError: If the caller sets ``version`` and the card declares another
            version. The error is also a :class:`~timenet.errors.TimeFValidationError`.
        TimeNetMissingParentError: If a parent release is missing, with a hint that names the
            keywords.
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

    This is the one-call sugar over :func:`build` plus :meth:`timenet.client.TimeNet.load`. For a
    clear producer and consumer split, call :func:`build` and ``TimeNet().load`` yourself.

    Args:
        dataset_id: The dataset id (``org/name``).
        version: The version string, or ``None`` for the latest. :func:`build` checks this value
            against the version that the card of the connector declares, and rejects another one.
        fetch_from: The registry, or its URI, that the build and the read copy missing parent
            releases and payload sources from.
        build_parents: Build missing parent releases from their installed connectors.

    Returns:
        The loaded dataset.
    """
    from timenet.client import TimeNet  # ruff: ignore[import-outside-top-level]

    build(dataset_id, version=version, fetch_from=fetch_from, build_parents=build_parents)
    return TimeNet().load(dataset_id, version, fetch_from=fetch_from)
