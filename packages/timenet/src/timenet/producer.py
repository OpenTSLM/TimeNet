"""Build dataset releases in a local registry and publish them to other registries.

The default local and remote registries come from ``TIMENET_LOCAL_REGISTRY`` and ``TIMENET_REMOTE_REGISTRY``.
``TIMENET_AUTO_BUILD`` controls whether installed connectors supply missing parents by default.
"""

from collections.abc import Sequence
from pathlib import Path

from timenet._hints import Facts, Operation, RequestedRelease, explain
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
from timenet.types.references import DatasetRef, is_latest


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
        open_access_only: Require open access for every release before running the plan.
        keep_cache: Keep the raw downloads in ``TIMENET_CACHE`` after the build.
        progress: A callback for copy and build progress events.

    Returns:
        The release directory in the local registry.

    Raises:
        TimeFValidationError: If ``local_registry`` is not a writable local registry, or ``sources`` is invalid.
        TimeNetRegistryError: If a registry in ``sources`` fails to open.
        TimeNetDatasetNotFoundError: If no installed connector builds the dataset.
        TimeNetVersionUnavailableError: If the requested version differs from the connector's version.
        TimeNetMissingParentError: If no source provides a required parent or payload source.
        TimeNetAccessError: If no source grants access to a parent or a release violates ``open_access_only``.
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
    wanted = RequestedRelease.parse(dataset_id)
    with explain(Facts([target, *parents], target, request=Operation.BUILD, wanted=wanted)):
        # A connector builds only the version of its card, so "latest" asks for that version.
        builder = declared(wanted.dataset_id, None if is_latest(wanted.version) else wanted.version)
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
    """Copy a release and its required payload sources to another registry.

    The copy skips identical releases and rejects conflicting files before writing to ``dest``.
    It replaces missing or damaged files at a local destination.

    Args:
        dataset_id: ``org/name`` for the latest version in ``source``, ``org/name@1.2.0`` for an exact release, or
            a ``DatasetRef``, which needs no lookup in ``source``.
        dest: The registry to copy to, or its URL or path.
        source: The source registry, URL, or path, or ``None`` for ``TIMENET_LOCAL_REGISTRY``.
        progress: A callback for copy progress events.

    Returns:
        The ID and version of the release in ``dest``.

    Raises:
        TimeNetRegistryError: If ``dest`` or ``source`` fails to open as a registry, or ``dest`` is read-only.
        TimeNetDatasetNotFoundError: If ``source`` lacks the release.
        TimeNetAccessError: If a registry withholds files or requires open access that a copied release lacks.
        TimeNetVersionConflictError: If ``dest`` holds different files for the release or a required payload source.
        TimeNetMissingParentError: If a required payload source is missing from ``source`` and incomplete in ``dest``.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by open_registry and copy_release
    target = open_registry(dest, label="dest") if isinstance(dest, (str, Path)) else dest
    if source is None:
        local_registry = default_local_registry()
        return _publish_from(
            dataset_id, dest=target, origin=local_registry, local_registry=local_registry, progress=progress
        )
    origin = open_registry(source, label="source") if isinstance(source, (str, Path)) else source
    # Only the default local registry gets the hints that copy or build a release into it.
    return _publish_from(dataset_id, dest=target, origin=origin, local_registry=None, progress=progress)


def _publish_from(
    dataset_id: str | DatasetRef,
    *,
    dest: WritableRegistry,
    origin: BaseRegistry,
    local_registry: LocalRegistry | None,
    progress: ProgressCallback | None,
) -> DatasetRef:
    """Copy a release and its required payload sources from one opened registry to another, as :func:`publish` does.

    Args:
        dataset_id: The release, as in :func:`publish`.
        dest: The registry to copy to.
        origin: The registry to copy from.
        local_registry: The local registry whose hints copy or build the release into it, or ``None`` for no such hint.
        progress: A callback for copy progress events.

    Returns:
        The ID and version of the release in ``dest``.

    Raises:
        TimeNetDatasetNotFoundError: If ``origin`` lacks the release.
        TimeNetAccessError: If a registry withholds files or requires open access that a copied release lacks.
        TimeNetVersionConflictError: If ``dest`` holds different files for the release or a required payload source.
        TimeNetMissingParentError: If a required payload source is missing from ``origin`` and incomplete in ``dest``.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by copy_release
    wanted = RequestedRelease.parse(dataset_id)
    facts = Facts([origin], local_registry, request=Operation.PUBLISH, dest=dest, wanted=wanted)
    with explain(facts):
        if isinstance(dataset_id, DatasetRef):
            # The caller names the exact release, so the copy needs no lookup.
            release = dataset_id
        else:
            answer = resolve_release([origin], wanted.dataset_id, wanted.version)
            if isinstance(answer, NotServed):
                raise answer.access_error(wanted.dataset_id)
            if isinstance(answer, NotFound):
                raise TimeNetDatasetNotFoundError(f"{wanted} is not in {origin}.")
            release = DatasetRef(dataset_id=wanted.dataset_id, version=answer.version)
        copy_release(release, origin, dest, progress=progress)
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
