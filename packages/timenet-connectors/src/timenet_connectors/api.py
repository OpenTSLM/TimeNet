"""Build and load datasets with installed connectors."""

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
    # ruff: ignore[docstring-extraneous-exception] (raised by the plan and its run, not directly here)
    """Build a dataset release in a local registry.

    The connector runs in an environment built from its ``requirements.txt``.
    Set ``TIMENET_ISOLATION=off`` to use the current interpreter.

    Args:
        dataset_id: The ``org/name`` ID declared in the connector's card.
        version: The required card version, or ``None`` to accept any version.
        out: The output directory, or ``None`` to use ``TIMENET_LOCAL_REGISTRY``.
        force: Rebuild an existing release instead of reusing it.
        keep_cache: Keep raw downloads after a successful build.
        fetch_from: A registry or registry URI to copy missing parent releases from.
        build_parents: Build missing parent releases from their installed connectors.

    Returns:
        The release's version directory.

    Raises:
        TimeNetDatasetNotFoundError: If no installed connector builds ``dataset_id``.
        TimeNetVersionUnavailableError: If the card version differs from ``version``.
        TimeNetMissingParentError: If a required parent release is missing.
    """
    from timenet.builders import BuildOptions, declared  # ruff: ignore[import-outside-top-level]
    from timenet.config import settings  # ruff: ignore[import-outside-top-level]
    from timenet.coordinator.build import plan_release_build, run_plan  # ruff: ignore[import-outside-top-level]
    from timenet.errors import TimeNetMissingParentError  # ruff: ignore[import-outside-top-level]
    from timenet.registry import LocalRegistry, open_registry  # ruff: ignore[import-outside-top-level]
    from timenet.registry._paths import release_dir  # ruff: ignore[import-outside-top-level, import-private-name]
    from timenet.sources import InstalledConnectors  # ruff: ignore[import-outside-top-level]

    release = declared(dataset_id).release
    # TODO(adrian): Remove temporary glue which maps a missing out onto the local registry of the default
    # list until timenet.build() replaces this function.
    root = Path(out).expanduser() if out is not None else settings().local_registry_dir
    source = open_registry(fetch_from) if isinstance(fetch_from, str) else fetch_from
    store = LocalRegistry(root)
    try:
        plan = plan_release_build(
            f"{dataset_id}@{version}" if version else dataset_id,
            store=store,
            # TODO(adrian): Remove temporary glue which maps fetch_from and build_parents onto a
            # list of sources until the producer API replaces this function.
            parents_from=[
                *([source] if source is not None else []),
                *([InstalledConnectors()] if build_parents else []),
            ],
            force=force,
        )
        run_plan(plan, store, options=BuildOptions(keep_cache=keep_cache))
    except TimeNetMissingParentError as error:
        error.hint = _missing_hint(fetch_from=fetch_from, build_parents=build_parents)
        raise
    return release_dir(root, release)


def _missing_hint(*, fetch_from: "str | BaseRegistry | None", build_parents: bool) -> str | None:
    """Suggest options to obtain a missing release.

    Args:
        fetch_from: The selected registry or URI, or ``None``.
        build_parents: Whether parent builds are enabled.

    Returns:
        A sentence naming unused options, or ``None`` if both are enabled.
    """
    if fetch_from is None and not build_parents:
        return "Copy it with fetch_from=<registry> or build it with build_parents=True."
    if fetch_from is None:
        return "Copy it with fetch_from=<registry>."
    if not build_parents:
        return "Build it with build_parents=True."
    return None


def load(
    dataset_id: str,
    version: str | None = None,
    *,
    fetch_from: "str | BaseRegistry | None" = None,
    build_parents: bool = False,
) -> "TimeFDataset":
    """Build a dataset if needed and load its values into memory.

    The build writes to the default local registry.

    Args:
        dataset_id: The dataset ID (``org/name``).
        version: The version declared in the connector's card, or ``None`` for the latest.
        fetch_from: A registry or URI to copy missing dependencies from.
        build_parents: Build missing parent releases from their installed connectors.

    Returns:
        The dataset with all stored values in memory.
    """
    from timenet.client import TimeNet  # ruff: ignore[import-outside-top-level]

    build(dataset_id, version=version, fetch_from=fetch_from, build_parents=build_parents)
    return TimeNet().load(dataset_id, version)
