"""Plan and read releases from ordered sources.

A read opens each release in the registry that provides it, so a release and its payload sources can come from
different registries. A release that only installed connectors provide is built into the local registry first.
"""

from collections.abc import Sequence
from typing import NoReturn, cast

from timenet.builders import declared, find_card
from timenet.coordinator.build import run_plan
from timenet.coordinator.plan import Plan, ReadStep, _Graph, describe_searched_sources
from timenet.errors import TimeNetDatasetNotFoundError
from timenet.registry.dependencies import attach_payload_sources
from timenet.registry.local import LocalRegistry
from timenet.registry.version import DatasetVersion
from timenet.sources import InstalledConnectors, NotFound, ReleaseSource
from timenet.types.references import DatasetRef, is_latest


def plan_read(
    dataset_id: str, version: str | None, *, sources: Sequence[ReleaseSource], store: LocalRegistry | None
) -> Plan:
    """Plan and validate the steps needed to read a release.

    A release that a registry provides gets a ``"read"`` step, and a release that only installed connectors
    provide gets the steps of its build into ``store``. This checks local files and payload source pins before
    any step runs.

    Args:
        dataset_id: The dataset ID.
        version: An exact version, ``"latest"``, or ``None``, as in :func:`~timenet.sources.resolve_release`.
        sources: Registries and ``InstalledConnectors()`` to search, in order.
        store: The local registry for copies and builds, or ``None`` to copy and build nothing.

    Returns:
        The ordered plan, ending with the requested release's step if it has one.

    Raises:
        TimeNetDatasetNotFoundError: If no source provides the release.
        TimeNetVersionUnavailableError: If the installed connector builds a different version.
        TimeNetAccessError: If a source lists a needed release but no source serves it.
        TimeNetMissingParentError: If a required payload source or build parent is unavailable.
        TimeNetFileError: If a required local copy has missing files or incorrect file sizes.
        TimeFFormatError: If a payload source conflicts with a pin or two pins conflict.
        TimeFValidationError: If a build has no local registry to build into, or the releases form a cycle.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the planner and the checks
    graph, _ = _walk(dataset_id, version, sources=sources, store=store)
    return Plan(tuple(graph.create_ordered_steps()))


def read_release(
    dataset_id: str,
    version: str | None,
    *,
    sources: Sequence[ReleaseSource],
    store: LocalRegistry | None,
) -> DatasetVersion:
    """Open a release with its payload sources after completing required copies and builds.

    This can raise any error listed by :func:`plan_read`.
    A local registry is required only for copies and builds.

    Args:
        dataset_id: The dataset ID.
        version: An exact version, ``"latest"``, or ``None``, as in :func:`~timenet.sources.resolve_release`.
        sources: Registries and ``InstalledConnectors()`` to search, in order.
        store: The local registry for copies and builds, or ``None`` to copy and build nothing.

    Returns:
        The release with attached payload sources that match its file pins.

    Raises:
        TimeNetDatasetNotFoundError: If no source provides the release.
        TimeNetVersionConflictError: If a copied release conflicts with the files in ``store``.
        TimeFFormatError: If a build changes files pinned by other releases.
        TimeNetBuildError: If a build fails.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the plan, the run and the open
    graph, release = _walk(dataset_id, version, sources=sources, store=store)
    plan = Plan(tuple(graph.create_ordered_steps()))
    if any(not isinstance(step, ReadStep) for step in plan.steps):
        # The plan copies and builds only into a local registry.
        run_plan(plan, cast("LocalRegistry", store))
    return attach_payload_sources(
        graph.open(release), graph.open, searched=lambda: describe_searched_sources(sources, release)
    )


def raise_not_found(dataset_id: str, version: str | None, sources: Sequence[ReleaseSource]) -> NoReturn:
    """Raise an error for an unavailable release.

    For an exact version, check the installed connector if the sources include it.

    Args:
        dataset_id: The dataset ID.
        version: An exact version, ``"latest"``, or ``None`` or ``""`` for no version.
        sources: The sources searched for the release.

    Raises:
        TimeNetVersionUnavailableError: If the installed connector declares a different version from the exact request.
        TimeNetInvalidCardError: If the connector's card is unreadable or invalid.
        TimeNetDatasetNotFoundError: Otherwise, with a message listing the sources searched.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by declared
    requested = None if is_latest(version) else DatasetRef.parse(f"{dataset_id}@{version}")
    connectors = any(isinstance(source, InstalledConnectors) for source in sources)
    if requested is not None and connectors and find_card(dataset_id) is not None:
        # The installed connector builds another version.
        declared(dataset_id, requested.version)
    names = ", ".join(str(source) for source in sources)
    wanted = dataset_id if requested is None else str(requested)
    raise TimeNetDatasetNotFoundError(f"{wanted} is in none of the sources: {names}.")


def _walk(
    dataset_id: str, version: str | None, *, sources: Sequence[ReleaseSource], store: LocalRegistry | None
) -> tuple[_Graph, DatasetRef]:
    """Plan a read in one walk of the release graph, and check the plan.

    Returns:
        The graph, and the requested release with its exact version.

    Raises:
        TimeNetDatasetNotFoundError: If no source provides the release.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by raise_not_found
    graph = _Graph(store, sources, open_only=False)
    answer = graph.lookups.resolve(sources, dataset_id, version)
    if isinstance(answer, NotFound):
        if not is_latest(version):
            # Nothing but a registry replaces a damaged copy in the local registry.
            damage = graph.in_output(DatasetRef.parse(f"{dataset_id}@{version}"))[1]
            if damage is not None:
                raise damage
        raise_not_found(dataset_id, version, sources)
    release = DatasetRef(dataset_id=dataset_id, version=answer.version)
    # The walk takes the answer, so that a read without an exact version does not ask again for the exact version.
    graph.visit(release, None, needed_in_local_registry=False, answer=answer)
    graph.check()
    return graph, release
