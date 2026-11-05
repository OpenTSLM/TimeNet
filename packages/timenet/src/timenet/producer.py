"""Build dataset releases in a local store and publish them to other registries.

The default store and registry come from ``TIMENET_LOCAL_REGISTRY`` and ``TIMENET_REMOTE_REGISTRY``.
``TIMENET_AUTO_BUILD`` controls whether installed connectors supply missing parents by default.
"""

from collections.abc import Sequence
from pathlib import Path

from timenet._hints import Facts, explain
from timenet.builders import BuildOptions
from timenet.client import as_release_sources, default_sources, default_store
from timenet.config import check_store
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
from timenet.sources import NotFound, NotServed, ReleaseSource
from timenet.types.references import DatasetRef, split_ref


def build(  # ruff: ignore[too-many-arguments]
    dataset_id: str,
    *,
    store: str | Path | LocalRegistry | None = None,
    parents_from: Sequence[ReleaseSource | str | Path] | None = None,
    force: bool = False,
    open_access_only: bool = False,
    keep_cache: bool = False,
    progress: ProgressCallback | None = None,
) -> Plan:
    """Build a dataset release into a local store with its installed connector.

    Missing parents and payload sources come from the first matching source in ``parents_from``.
    The build uses a separate environment with the connector's dependencies unless ``TIMENET_ISOLATION`` is ``off``.

    Args:
        dataset_id: The dataset ID (``org/name``) or exact release (``org/name@1.2.0``).
        store: A ``LocalRegistry``, local path, or ``file://`` URI, or ``None`` for ``TIMENET_LOCAL_REGISTRY``.
        parents_from: Sources, URLs, or paths after the store, or ``None`` for :func:`~timenet.client.default_sources`.
        force: Build the release again, even if the store holds it.
        open_access_only: Require open access for every release before running the plan.
        keep_cache: Keep the raw downloads in ``TIMENET_CACHE`` after the build.
        progress: A callback for copy and build progress events.

    Returns:
        The completed plan with steps in execution order.

    Raises:
        TimeFValidationError: If the store is not local, overlaps the cache, or ``parents_from`` is invalid.
        TimeNetRegistryError: If the store is remote or a parent registry fails to open.
        TimeNetDatasetNotFoundError: If no installed connector builds the dataset.
        TimeNetVersionUnavailableError: If the requested version differs from the connector's version.
        TimeNetMissingParentError: If no source provides a required parent or payload source.
        TimeNetAccessError: If no source grants access to a parent or a release violates ``open_access_only``.
        TimeNetVersionConflictError: If a release to copy has different files from the same version in the store.
        TimeNetFileError: If a stored release is damaged and no registry in ``parents_from`` provides it.
        TimeNetBuildError: If a build fails.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the coordinator and local_registry_path
    target, parents = _build_inputs(store, parents_from)
    with explain(Facts([target, *parents], target, request="build", wanted=dataset_id), "sdk"):
        plan = plan_release_build(
            dataset_id, store=target, parents_from=parents, force=force, open_access_only=open_access_only
        )
        run_plan(plan, target, options=BuildOptions(keep_cache=keep_cache, progress=progress))
    return plan


def plan_build(
    dataset_id: str,
    *,
    store: str | Path | LocalRegistry | None = None,
    parents_from: Sequence[ReleaseSource | str | Path] | None = None,
    force: bool = False,
    open_access_only: bool = False,
) -> Plan:
    """Plan a dataset build without running it.

    Planning reads dataset cards and manifests.

    Args:
        dataset_id: The dataset ID (``org/name``) or exact release (``org/name@1.2.0``).
        store: The local store, with the same options and default as :func:`build`.
        parents_from: Sources for missing parents and payloads, with the same options and default as :func:`build`.
        force: Plan a new build even if the store holds the release.
        open_access_only: Refuse a plan that needs a release without open access.

    Returns:
        The build plan with steps in execution order.

    Raises:
        TimeFValidationError: If the store is not local, overlaps the cache, or ``parents_from`` is invalid.
        TimeNetRegistryError: If the store is remote or a parent registry fails to open.
        TimeNetDatasetNotFoundError: If no installed connector builds the dataset.
        TimeNetVersionUnavailableError: If the requested version differs from the connector's version.
        TimeNetMissingParentError: If no source provides a required parent or payload source.
        TimeNetAccessError: If ``open_access_only`` is set and the plan needs a release without open access.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the coordinator and local_registry_path
    target, parents = _build_inputs(store, parents_from)
    with explain(Facts([target, *parents], target, request="build", wanted=dataset_id), "sdk"):
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
    """Copy a release and its required payload sources to another registry.

    The copy skips identical releases and rejects conflicting files before writing to ``dest``.
    It replaces missing or damaged files at a local destination.

    Args:
        dataset_id: ``org/name`` for the latest version in ``source``, or ``org/name@1.2.0`` for an exact release.
        dest: The registry to copy to, or its URL or path.
        source: The source registry, URL, or path, or ``None`` for ``TIMENET_LOCAL_REGISTRY``.
        progress: A callback for copy progress events.

    Returns:
        The ID and version of the release in ``dest``.

    Raises:
        TimeNetRegistryError: If ``dest`` or ``source`` fails to open as a registry.
        TimeNetDatasetNotFoundError: If ``source`` lacks the release.
        TimeNetAccessError: If a registry withholds files or requires open access that a copied release lacks.
        TimeNetVersionConflictError: If ``dest`` holds different files for the release or a required payload source.
        TimeNetMissingParentError: If a required payload source is missing from ``source`` and incomplete in ``dest``.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by open_registry and copy_release
    target = open_registry(dest, label="dest") if isinstance(dest, (str, Path)) else dest
    store: LocalRegistry | None = None
    origin: BaseRegistry
    if source is None:
        store = origin = default_store()
    else:
        origin = open_registry(source, label="source") if isinstance(source, (str, Path)) else source
    bare_id, version = split_ref(dataset_id)
    wanted = bare_id if version is None else f"{bare_id}@{version}"
    # Only the default store gets the hints that copy or build a release into it.
    with explain(Facts([origin], store, request="publish", dest=target, wanted=wanted), "sdk"):
        answer = origin.resolve(bare_id, version)
        if isinstance(answer, NotServed):
            raise answer.access_error(bare_id)
        if isinstance(answer, NotFound):
            raise TimeNetDatasetNotFoundError(f"{wanted} is not in {origin}.")
        release = DatasetRef(dataset_id=bare_id, version=answer.version)
        copy_release(release, origin, target, progress=progress)
    return release


def _build_inputs(
    store: str | Path | LocalRegistry | None, parents_from: Sequence[ReleaseSource | str | Path] | None
) -> tuple[LocalRegistry, list[ReleaseSource]]:
    """Resolve the local store and parent sources from arguments or environment settings.

    Args:
        store: A local registry, path, or ``file://`` URI, or ``None`` for the configured store.
        parents_from: Sources, URLs, or paths for missing releases, or ``None`` for the configured defaults.

    Returns:
        The store and sources for missing releases in search order.

    Raises:
        TimeFValidationError: If the store is not local, overlaps the cache, or ``parents_from`` is invalid.
        TimeNetRegistryError: If the store is remote or a parent registry fails to open.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by local_registry_path and open_registry
    if store is None:
        target = default_store()
    elif isinstance(store, (str, Path)):
        target = LocalRegistry(local_registry_path(store, label="store"), store=True)
    elif isinstance(store, LocalRegistry):
        target = store
    else:
        raise TimeFValidationError(
            "store takes a local registry, because a build always lands locally. Publish the result with "
            "timenet.publish()."
        )
    if store is not None:
        check_store(target.root)
    if parents_from is None:
        # The first entry of the default list is the store, and the build already has its own.
        return target, default_sources(local_registry=target.root)[1:]
    return target, as_release_sources(parents_from, keyword="parents_from", store=target.root)
