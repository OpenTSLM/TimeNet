"""Build releases into a local registry, and publish them to other registries.

Each argument of :func:`build`, :func:`plan_build` and :func:`publish` that you leave out comes from
the environment at the time of the call, as in the default list of sources of ``TimeNet()``. So a
default build lands where a default client reads.
"""

from collections.abc import Sequence
from pathlib import Path

from timenet.builders import BuildOptions
from timenet.client import as_release_sources, default_local_registry, default_sources
from timenet.config import check_local_registry
from timenet.coordinator.build import plan_release_build, run_plan
from timenet.coordinator.plan import Plan
from timenet.errors import TimeFValidationError, TimeNetDatasetNotFoundError
from timenet.progress import ProgressCallback
from timenet.registry import (
    BaseRegistry,
    LocalRegistry,
    WritableRegistry,
    copy_release,
    local_registry_path,
    open_registry,
)
from timenet.sources import NotFound, NotServed, ReleaseSource, resolve_release
from timenet.types.references import DatasetRef, split_ref


def build(  # ruff: ignore[too-many-arguments]
    dataset_id: str,
    *,
    local_registry: str | Path | LocalRegistry | None = None,
    parents_from: Sequence[ReleaseSource | str | Path] | None = None,
    force: bool = False,
    open_access_only: bool = False,
    keep_cache: bool = False,
    progress: ProgressCallback | None = None,
) -> Plan:
    """Build a release into a local registry with the installed connector that claims it.

    Each parent and payload source that the local registry lacks comes from the first source in
    ``parents_from`` that provides it. The build runs in an environment made from the requirements of
    the connector, unless ``TIMENET_ISOLATION`` is ``off``.

    Args:
        dataset_id: The dataset ID, as ``org/name``, or one exact release, as ``org/name@1.2.0``.
        local_registry: The local registry: a :class:`~timenet.registry.LocalRegistry`, a local path, or a
            ``file://`` URI. ``None`` means ``TIMENET_LOCAL_REGISTRY``.
        parents_from: Where missing parents and payload sources come from, in order after the local registry:
            registries, their URLs or paths, and ``InstalledConnectors()``. ``None`` means
            ``TIMENET_REMOTE_REGISTRY``, then ``InstalledConnectors()`` unless ``TIMENET_AUTO_BUILD``
            is false.
        force: Build the release again, even if the local registry holds it.
        open_access_only: Refuse a plan that needs a release without open access, before any step runs.
        keep_cache: Keep the raw downloads in ``TIMENET_CACHE`` after the build.
        progress: Receives the progress events of each copy and each build.

    Returns:
        The plan that ran: its steps in the order in which they ran.

    Raises:
        TimeFValidationError: If ``local_registry`` is not local or overlaps ``TIMENET_CACHE``, or ``parents_from``
            is not a list.
        TimeNetRegistryError: If ``local_registry`` is a URL of a remote registry, or an item of ``parents_from``
            cannot be opened as a registry.
        TimeNetDatasetNotFoundError: If no installed connector builds the dataset.
        TimeNetVersionUnavailableError: If the requested version differs from the connector's version. Nothing
            runs then.
        TimeNetMissingParentError: If no source provides a required parent or payload source. Nothing runs then.
        TimeNetAccessError: If a source denies access to a parent and no later source provides it, or
            ``open_access_only`` is set and the plan needs a release without open access. Nothing runs then.
        TimeNetVersionConflictError: If a release to copy has other files than the same version in the local registry.
        TimeNetFileError: If a stored release is damaged and no registry in ``parents_from`` provides it.
        TimeNetBuildError: If a build fails.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the coordinator and local_registry_path
    target, parents = _build_inputs(local_registry, parents_from)
    plan = plan_release_build(
        dataset_id, store=target, parents_from=parents, force=force, open_access_only=open_access_only
    )
    run_plan(plan, target, options=BuildOptions(keep_cache=keep_cache, progress=progress))
    return plan


def plan_build(
    dataset_id: str,
    *,
    local_registry: str | Path | LocalRegistry | None = None,
    parents_from: Sequence[ReleaseSource | str | Path] | None = None,
    force: bool = False,
    open_access_only: bool = False,
) -> Plan:
    """Return the plan that :func:`build` runs with the same arguments, and run nothing.

    The call reads cards and manifests.

    Args:
        dataset_id: The dataset ID (``org/name``) or exact release (``org/name@1.2.0``).
        local_registry: The local registry, with the same options and default as :func:`build`.
        parents_from: Sources for missing parents and payloads, with the same options and default as :func:`build`.
        force: Plan a new build even if the local registry holds the release.
        open_access_only: Refuse a plan that needs a release without open access.

    Returns:
        The plan: its steps in the order in which they run.

    Raises:
        TimeFValidationError: If ``local_registry`` is not local or overlaps ``TIMENET_CACHE``, or ``parents_from``
            is not a list.
        TimeNetRegistryError: If ``local_registry`` is a URL of a remote registry, or an item of ``parents_from``
            cannot be opened as a registry.
        TimeNetDatasetNotFoundError: If no installed connector builds the dataset.
        TimeNetVersionUnavailableError: If the requested version differs from the connector's version.
        TimeNetMissingParentError: If no source provides a required parent or payload source.
        TimeNetAccessError: If ``open_access_only`` is set and the plan needs a release without open access.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the coordinator and local_registry_path
    target, parents = _build_inputs(local_registry, parents_from)
    return plan_release_build(
        dataset_id, store=target, parents_from=parents, force=force, open_access_only=open_access_only
    )


def publish(
    dataset_id: str,
    *,
    dest: str | Path | WritableRegistry,
    source: str | Path | BaseRegistry | None = None,
    progress: ProgressCallback | None = None,
) -> DatasetRef:
    """Copy a complete release and the payload sources that it needs to another registry.

    A published version cannot change, so the call copies nothing if ``dest`` holds the same files,
    and raises an error if it holds other files. A local ``dest`` with a missing or damaged file gets
    the release again.

    Args:
        dataset_id: The dataset ID, as ``org/name`` for the latest version in ``source``, or one exact
            release, as ``org/name@1.2.0``.
        dest: The registry to copy to, or its URL or path.
        source: The registry to copy from, or its URL or path. ``None`` means the local registry,
            ``TIMENET_LOCAL_REGISTRY``.
        progress: Receives the progress events of the copy.

    Returns:
        The release that ``dest`` now holds.

    Raises:
        TimeNetRegistryError: If ``dest`` or ``source`` cannot be opened as a registry.
        TimeNetDatasetNotFoundError: If ``source`` does not hold the release.
        TimeNetAccessError: If ``source`` lists the release but does not serve its files. Also if a
            release of the copy is not open, and ``dest`` or ``source`` requires open access.
        TimeNetVersionConflictError: If ``dest`` holds the release, or a payload source that it
            needs, with other files. Nothing is copied then.
        TimeNetMissingParentError: If a payload source is neither in ``source`` nor complete in
            ``dest``.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by open_registry and copy_release
    target = open_registry(dest, label="dest") if isinstance(dest, (str, Path)) else dest
    if source is None:
        origin = default_local_registry()
    else:
        origin = open_registry(source, label="source") if isinstance(source, (str, Path)) else source
    bare_id, version = split_ref(dataset_id)
    answer = resolve_release([origin], bare_id, version)
    if isinstance(answer, NotServed):
        raise answer.access_error(bare_id)
    if isinstance(answer, NotFound):
        wanted = bare_id if version is None else f"{bare_id}@{version}"
        raise TimeNetDatasetNotFoundError(f"{wanted} is not in {origin}.")
    release = DatasetRef(dataset_id=bare_id, version=answer.version)
    copy_release(release, origin, target, progress=progress)
    return release


def _build_inputs(
    store: str | Path | LocalRegistry | None, parents_from: Sequence[ReleaseSource | str | Path] | None
) -> tuple[LocalRegistry, list[ReleaseSource]]:
    """Pick the local registry and the sources of the parents of a build, from the arguments or the environment.

    Args:
        store: The value of ``local_registry``.
        parents_from: The value of ``parents_from``.

    Returns:
        The local registry, and the sources of the releases that it lacks, in order.

    Raises:
        TimeFValidationError: If ``store`` is not local or overlaps ``TIMENET_CACHE``, or ``parents_from``
            is not a list.
        TimeNetRegistryError: If ``store`` is a URL of a remote registry, or an item of ``parents_from``
            cannot be opened as a registry.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by local_registry_path and open_registry
    if store is None:
        target = default_local_registry()
    elif isinstance(store, (str, Path)):
        target = LocalRegistry(local_registry_path(store, label="local_registry"), store=True)
    elif isinstance(store, LocalRegistry):
        target = store
    else:
        raise TimeFValidationError(
            "local_registry takes a local registry, because a build always lands locally. Publish the result with "
            "timenet.publish()."
        )
    if store is not None:
        check_local_registry(target.root)
    if parents_from is None:
        # The first entry of the default list is the local registry, and the build already has its own.
        return target, default_sources(local_registry=target.root)[1:]
    return target, as_release_sources(parents_from, keyword="parents_from", store=target.root)
