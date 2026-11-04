"""Build releases into a local registry, and publish them to other registries.

Each argument of :func:`build`, :func:`plan_build` and :func:`publish` that you leave out comes from
the environment at the time of the call, as in the default list of sources of ``TimeNet()``. So a
default build lands where a default client reads.
"""

from collections.abc import Sequence
from pathlib import Path

from timenet._registry_uri import registry_location
from timenet.builders import BuildOptions, declared
from timenet.client import as_release_sources, default_local_registry, default_sources
from timenet.coordinator.build import run_plan
from timenet.coordinator.plan import Plan, plan_steps
from timenet.errors import TimeFValidationError, TimeNetDatasetNotFoundError
from timenet.progress import ProgressCallback
from timenet.registry import BaseRegistry, LocalRegistry, WritableRegistry, copy_release, open_registry
from timenet.registry._paths import release_dir
from timenet.sources import NotFound, NotServed, ReleaseSource, resolve_release
from timenet.types.references import DatasetRef, is_latest, split_ref


def build(  # ruff: ignore[too-many-arguments]
    dataset_id: str,
    *,
    local_registry: str | Path | LocalRegistry | None = None,
    sources: Sequence[ReleaseSource | str | Path] | None = None,
    force: bool = False,
    open_access_only: bool = False,
    keep_cache: bool = False,
    progress: ProgressCallback | None = None,
) -> Path:
    """Build a dataset release into a local registry with its installed connector.

    Missing parents and payload sources come from the first source in ``sources`` that provides them.
    The build uses a separate environment with the connector's dependencies unless ``TIMENET_ISOLATION`` is ``off``.

    Args:
        dataset_id: The dataset ID (``org/name``) or exact release (``org/name@1.2.0``).
        local_registry: The writable local registry to build into: a ``LocalRegistry``, a local path, or a
            ``file://`` URI, or ``None`` for ``TIMENET_LOCAL_REGISTRY``.
        sources: Sources, registry URLs, or paths for missing releases, or ``None`` for
            :func:`~timenet.client.default_sources` with ``local_registry`` first.
        force: Build the release again, even if the local registry holds it.
        open_access_only: Refuse a plan that needs a release without open access, before any step runs.
        keep_cache: Keep the raw downloads in ``TIMENET_CACHE`` after the build.
        progress: Receives the progress events of each copy and each build.

    Returns:
        The release directory in the local registry.

    Raises:
        TimeFValidationError: If ``local_registry`` is not a writable local registry, or ``sources`` is invalid.
        TimeNetRegistryError: If a registry in ``sources`` fails to open.
        TimeNetDatasetNotFoundError: If no installed connector builds the dataset.
        TimeNetVersionUnavailableError: If the requested version differs from the connector's version. Nothing
            runs then.
        TimeNetMissingParentError: If no source provides a required parent or payload source. Nothing runs then.
        TimeNetAccessError: If a source denies access to a parent and no later source provides it, or
            ``open_access_only`` is set and the plan needs a release without open access. Nothing runs then.
        TimeNetVersionConflictError: If a release to copy has other files than the same version in the local registry.
        TimeNetFileError: If a stored release is damaged and no registry in ``sources`` provides it.
        TimeNetBuildError: If a build fails.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the coordinator and open_registry
    target = _build_target(local_registry)
    # The planner leaves the local registry out of the sources of its copies and builds.
    parents = (
        default_sources(local_registry=target.root)
        if sources is None
        else as_release_sources(sources, keyword="sources")
    )
    bare_id, version = split_ref(dataset_id)
    # A connector builds only the version of its card, so "latest" asks for that version.
    builder = declared(bare_id, None if is_latest(version) else version)
    release, steps = plan_steps(
        builder, target, parents_from=parents, force_build=force, open_access_only=open_access_only
    )
    run_plan(Plan(tuple(steps)), target, options=BuildOptions(keep_cache=keep_cache, progress=progress))
    return release_dir(target.root, release)


def publish(
    dataset_id: str | DatasetRef,
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
        dataset_id: ``org/name`` for the latest version in ``source``, ``org/name@1.2.0`` for an exact release, or
            a ``DatasetRef``, which needs no lookup in ``source``.
        dest: The registry to copy to, or its URL or path.
        source: The registry to copy from, or its URL or path. ``None`` means the local registry,
            ``TIMENET_LOCAL_REGISTRY``.
        progress: Receives the progress events of the copy.

    Returns:
        The release that ``dest`` now holds.

    Raises:
        TimeNetRegistryError: If ``dest`` or ``source`` cannot be opened as a registry, or ``dest`` is read-only.
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
    if isinstance(dataset_id, DatasetRef):
        # The caller names the exact release, so the copy starts at once.
        copy_release(dataset_id, origin, target, progress=progress)
        return dataset_id
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


def _build_target(local_registry: str | Path | LocalRegistry | None) -> LocalRegistry:
    """Open the local registry that a build writes into.

    Args:
        local_registry: A local registry, path, or ``file://`` URI, or ``None`` for the configured one.

    Returns:
        The writable local registry.

    Raises:
        TimeFValidationError: If ``local_registry`` is remote, read-only, or not a registry.
        TimeNetRegistryError: If ``local_registry`` is blank or has an unsupported scheme.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by registry_location
    if local_registry is None:
        return default_local_registry()
    if isinstance(local_registry, (str, Path)):
        location = registry_location(local_registry, label="local_registry")
        if isinstance(location, Path):
            return LocalRegistry(location)
    elif isinstance(local_registry, LocalRegistry) and not local_registry.read_only:
        return local_registry
    raise TimeFValidationError(
        "local_registry takes a writable local registry, because a build writes into it. Publish the result to "
        "another registry with timenet.publish()."
    )
