"""Plan and prepare the parent releases that a build needs, before any connector runs.

Each connector card lists its exact parent releases under ``parents``. The planner works in two
phases. First, it walks from the requested release through the cards and the manifests. The walk
finds a graph with one node for each release that the build needs. Each node records what the plan
does with the release, and the releases that it depends on. Then
:class:`graphlib.TopologicalSorter` puts each release after the releases that it depends on, and
finds cycles. A cycle, or a parent that the plan cannot get, fails here, before any download. If
the steps of the plan run in this order, each release that the build opens is in place before the
build starts. The build then only opens these releases.

A payload source is a release whose stored files another release reads. The release that reads
them records each file by path, size, and checksum. These records are the pins. If the plan builds
a missing payload source, the build must write the same files. The plan gives the pins to the
build. The build publishes the release only if the files match. Otherwise, it deletes the new files
and raises an error.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter
from typing import Literal

import pyarrow.fs as pafs

from timenet.builders import find_card
from timenet.errors import (
    TimeFFormatError,
    TimeFValidationError,
    TimeNetDatasetNotFoundError,
    TimeNetFileError,
    TimeNetMissingParentError,
)
from timenet.manifest import Manifest, PayloadSource
from timenet.registry.base import BaseRegistry
from timenet.registry.dependencies import check_file_sizes, prepare_local_copy
from timenet.registry.local import LocalRegistry
from timenet.registry.version import DatasetVersion
from timenet.registry.writable import WritableRegistry
from timenet.types import DatasetMetadata
from timenet.types.access import require_open_access
from timenet.types.references import DatasetRef


@dataclass(frozen=True)
class ParentStep:
    """One release that the plan fetches or builds before the build of the requested release.

    The output does not hold the release, or it does not hold all payload sources of the release. A
    fetch copies the release, and each payload source in its manifest that the output does not hold.
    """

    reference: DatasetRef
    action: Literal["fetch", "build"]
    pinned_by: Mapping[DatasetRef, PayloadSource] = field(default_factory=dict, hash=False)
    """For a build: each release that already reads files of this release, with its pins on them."""


@dataclass(frozen=True)
class _Release:
    """One node of the graph: a release that the build needs, and what the plan does with it."""

    opened: bool
    """True if the build opens the release with its payload sources. False if the release is only a
    payload source of another release, and the build reads only its pinned files."""
    action: Literal["fetch", "build"] | None
    """``"fetch"`` or ``"build"``, or ``None`` if the output holds the release."""
    dependencies: tuple[DatasetRef, ...]
    """The releases that must be in place before this release. A release to build depends on the
    parents on its card. An opened release that the plan does not build depends on its payload
    sources. Other releases depend on nothing."""


@dataclass
class _Graph:
    """The graph of the releases that a build needs, which the walk of :meth:`visit` finds."""

    output: WritableRegistry
    fetch_from: BaseRegistry | None
    build_parents: bool
    searched: str
    """The registries that the plan looks in, for the message of a missing release."""
    releases: dict[DatasetRef, _Release] = field(default_factory=dict)
    """The nodes, in the order in which the walk found them."""
    fetched: set[DatasetRef] = field(default_factory=set)
    """The payload sources that the plan fetches only as part of a fetch of the release that reads them."""
    pins: dict[DatasetRef, dict[DatasetRef, PayloadSource]] = field(default_factory=dict)
    """For each payload source: each release that reads its files, with its pins on them."""

    def visit(self, reference: DatasetRef, needed_by: DatasetRef, pinned: PayloadSource | None = None) -> None:
        """Add a release to the graph, then visit the releases that it depends on.

        The walk does not look for cycles. It stops at a release that it already found as an opened
        release, and at a payload source that it already found.

        Args:
            reference: The release.
            needed_by: The release that needs ``reference``.
            pinned: The pins of ``needed_by`` on the files of ``reference``, if ``reference`` is a
                payload source of ``needed_by``. ``None`` if the build opens ``reference`` as a
                parent.
        """
        opened = pinned is None
        if pinned is not None:
            self.pins.setdefault(reference, {})[needed_by] = pinned
        found = self.releases.get(reference)
        # If the walk found a payload source first, and later finds it as a parent, it looks at the
        # release again with its payload sources.
        if found is not None and (found.opened or not opened):
            return
        manifest, action = _locate(self.output, self.fetch_from, reference)
        if manifest is None:
            card = _card_to_build(
                reference,
                needed_by,
                self.output,
                build_parents=self.build_parents,
                pinned=not opened,
                searched=self.searched,
            )
            self.releases[reference] = _Release(opened, "build", tuple(card.parents.values()))
            # A build opens each of its parents with the payload sources of that parent.
            for parent in card.parents.values():
                self.visit(parent, reference)
        elif opened:
            # To open a release, the build needs its direct payload sources. These are the releases
            # whose files the release reads. The build does not need their own payload sources.
            sources = list(manifest.payload_sources.values())
            self.releases[reference] = _Release(opened, action, tuple(source.release for source in sources))
            for source in sources:
                self.visit(source.release, reference, source)
        else:
            self.releases[reference] = _Release(opened, action, ())
            # A fetch of the release that reads this payload source also copies the payload source.
            if action == "fetch":
                self.fetched.add(reference)

    def order(self) -> list[DatasetRef]:
        """Order the releases so that each release comes after the releases that it depends on.

        Returns:
            The releases, dependencies first.

        Raises:
            TimeFValidationError: If the releases depend on each other in a cycle. The message lists
                each release of the cycle before the release that it depends on. It starts with the
                release that the graph holds first, which is the requested release if the cycle
                holds it.
        """
        graph = {reference: release.dependencies for reference, release in self.releases.items()}
        try:
            return list(TopologicalSorter(graph).static_order())
        except CycleError as error:
            # The sorter lists each release before the release that depends on it, and repeats the
            # first release at the end.
            cycle = list(reversed(error.args[1][1:]))
            found = list(self.releases)
            start = cycle.index(min(cycle, key=found.index))
            cycle = cycle[start:] + cycle[:start]
            path = " -> ".join(str(reference) for reference in (*cycle, cycle[0]))
            raise TimeFValidationError(f"dataset parent cycle: {path}") from error

    def steps(self, requested: DatasetRef) -> list[ParentStep]:
        """Make one step for each release that the plan fetches or builds, in the order of :meth:`order`.

        Args:
            requested: The requested release. It gets no step, because the caller builds it.

        Returns:
            The steps, each after the steps of the releases that it depends on.

        Raises:
            TimeFValidationError: If the releases depend on each other in a cycle.
        """  # noqa: DOC502 - order() raises the cycle error
        steps: list[ParentStep] = []
        for reference in self.order():
            if reference == requested:
                continue
            release = self.releases[reference]
            if release.action == "build":
                steps.append(ParentStep(reference, "build", pinned_by=self.pins.get(reference, {})))
            # A fetch of an opened release also copies each payload source that the output does not
            # hold. So if the plan fetches a payload source only with the release that reads it, the
            # plan fetches that release again, even if the output holds it.
            elif release.opened and (
                release.action == "fetch" or any(source in self.fetched for source in release.dependencies)
            ):
                steps.append(ParentStep(reference, "fetch"))
        return steps


def plan_parents(
    dataset_id: str,
    output: WritableRegistry,
    *,
    fetch_from: BaseRegistry | None = None,
    build_parents: bool = False,
    force_build: bool = False,
) -> list[ParentStep]:
    """Plan how to get each parent release that a build of ``dataset_id`` needs.

    For each declared parent, a build needs the parent and the payload sources of that parent. It
    does not need the payload sources of those payload sources. The plan visits the declared parents
    of a release only if the plan builds that release. So if the output already holds the requested
    release, the plan visits its parents only if ``force_build`` is set.

    If the output does not hold a release, the plan fetches it from ``fetch_from``. If ``fetch_from``
    does not hold it either, the plan builds it with the installed connector that declares that
    exact version. If a local output holds a damaged copy, the plan fetches the release again and
    does not build it. A registry that is not local holds only releases whose cards declare
    ``access: open``. So for such an output, each release that the plan visits must declare
    ``access: open``.

    The plan works in two phases. First, a walk from the requested release finds the graph of the
    releases that the build needs. Then :class:`graphlib.TopologicalSorter` orders the graph and
    finds cycles, and the plan makes its steps in that order.

    Args:
        dataset_id: The dataset to build.
        output: The registry that the build writes to.
        fetch_from: The registry to fetch missing parents from. If it is ``None``, the plan fetches
            nothing.
        build_parents: Build missing parents with their installed connectors.
        force_build: Build the requested release again, even if the output holds it, and plan its
            parents.

    Returns:
        The steps to fetch or build parents. Each step comes after the steps of the releases that it
        depends on.

    Raises:
        LookupError: If no installed connector declares a card for ``dataset_id``.
        TimeNetInvalidCardError: If the card of an installed connector cannot be read.
        TimeFValidationError: If the parents form a cycle, if an installed connector declares another
            version than a parent pins, or if ``fetch_from`` is set and the output is not local.
        TimeNetMissingParentError: If a parent or a payload source is missing and ``build_parents``
            is not set.
        TimeNetDatasetNotFoundError: If a parent is missing and no installed connector builds it.
        TimeNetFileError: If a local output holds a damaged copy of a parent, and the plan cannot
            fetch the parent again.
        TimeNetAccessError: If the output is not local and the plan visits a release whose card, or
            a card in its lineage, does not declare ``access: open``.
    """  # noqa: DOC502 - raised by the helpers of the walk
    if fetch_from is not None and not isinstance(output, LocalRegistry):
        # A registry that is not local accepts a child only if it already holds the parents. Also, a
        # fetch copies a parent into the local cache, not into the output.
        raise TimeFValidationError("fetching parents needs a local output; publish the parents to the registry first")
    card = find_card(dataset_id)
    if card is None:
        raise LookupError(f"no installed connector declares a card for {dataset_id!r}")
    # For a dataset without parents, the plan does not read the output, so the build makes no extra request.
    if not card.parents or (not force_build and output.exists(card.dataset_id, str(card.dataset_version))):
        return []
    if not isinstance(output, LocalRegistry):
        require_open_access(card)
    searched = str(output) if fetch_from is None else f"{output} or {fetch_from}"
    graph = _Graph(output, fetch_from, build_parents=build_parents, searched=searched)
    root = DatasetRef(dataset_id=card.dataset_id, version=card.dataset_version)
    # The graph holds the requested release before the walk starts. So a parent that declares it
    # closes a cycle, and the walk does not look it up again.
    graph.releases[root] = _Release(opened=True, action="build", dependencies=tuple(card.parents.values()))
    for parent in card.parents.values():
        graph.visit(parent, root)
    return graph.steps(root)


def prepare_parents(  # noqa: PLR0913
    dataset_id: str,
    output: WritableRegistry,
    *,
    build: Callable[[ParentStep], object],
    fetch_from: BaseRegistry | None = None,
    build_parents: bool = False,
    force_build: bool = False,
) -> None:
    """Fetch or build each parent release that a build of ``dataset_id`` needs, before the build starts.

    Args:
        dataset_id: The dataset to build.
        output: The registry that the build writes to.
        build: Builds one parent into ``output``. For a step with ``pinned_by``, it must publish the
            release only if the new files match those pins.
        fetch_from: The registry to fetch missing parents from. If it is ``None``, the plan fetches
            nothing.
        build_parents: Build missing parents with their installed connectors.
        force_build: Build the requested release again, even if the output holds it, and plan its
            parents.

    Raises:
        TimeNetDatasetNotFoundError: If ``build`` does not publish the release of its step.
    """
    steps = plan_parents(
        dataset_id, output, fetch_from=fetch_from, build_parents=build_parents, force_build=force_build
    )
    for step in steps:
        if step.action == "fetch":
            prepare_local_copy(output.local_root, step.reference, fetch_from=fetch_from)
        else:
            build(step)
            # After a successful build, the output must hold the exact release with its payload sources.
            if isinstance(output, LocalRegistry):
                prepare_local_copy(output.root, step.reference)
            else:
                if _manifest(output, step.reference) is None:
                    raise TimeNetDatasetNotFoundError(f"builder did not publish parent release {step.reference}")
                output.open_version(step.reference.dataset_id, str(step.reference.version))


def _manifest(registry: BaseRegistry, reference: DatasetRef) -> Manifest | None:
    """Read the manifest of one release.

    This reads no other file and resolves no payload source.

    Returns:
        The manifest, or ``None`` if the registry does not hold the release.

    Raises:
        TimeFFormatError: If the manifest declares another dataset ID or version than ``reference``.
    """
    try:
        manifest = registry.get_manifest(reference.dataset_id, str(reference.version))
    except TimeNetDatasetNotFoundError:
        return None
    if manifest.dataset_id != reference.dataset_id or manifest.metadata.dataset_version != reference.version:
        raise TimeFFormatError(f"manifest for {reference} declares a different dataset ID or version")
    return manifest


def _locate(
    output: WritableRegistry, fetch_from: BaseRegistry | None, reference: DatasetRef
) -> tuple[Manifest | None, Literal["fetch"] | None]:
    """Find a release in the output, or else in ``fetch_from``.

    The output must hold an intact copy. In a local output, a copy with a missing file or a file of
    another size is damaged.

    Returns:
        The manifest and ``None`` if the output holds an intact copy. The manifest and ``"fetch"`` if
        the plan fetches the release. ``None`` and ``None`` if the plan builds it.

    Raises:
        TimeNetFileError: If a local output holds a copy with a missing file or a file of another
            size, and the plan cannot fetch the release again.
        TimeNetAccessError: If the output is not local and holds the release, and the card of the
            release, or a card in its lineage, does not declare ``access: open``.
    """  # noqa: DOC502 - the damage is raised from a variable, the access error by require_open_access
    manifest = _manifest(output, reference)
    damage = None
    if manifest is not None and isinstance(output, LocalRegistry):
        directory = output.root / reference.dataset_id / str(reference.version)
        try:
            check_file_sizes(
                DatasetVersion(manifest, pafs.LocalFileSystem(), str(directory)),
                manifest.files.all_files(),
                owner=f"committed release {reference} in {output}",
            )
        except TimeNetFileError as error:
            manifest, damage = None, error
    if manifest is not None:
        if not isinstance(output, LocalRegistry):
            require_open_access(manifest.metadata, manifest.lineage)
        return manifest, None
    fetched = None if fetch_from is None else _manifest(fetch_from, reference)
    if fetched is not None:
        return fetched, "fetch"
    if damage is not None:
        raise damage
    return None, None


def _card_to_build(  # noqa: PLR0913
    reference: DatasetRef,
    needed_by: DatasetRef,
    output: WritableRegistry,
    *,
    build_parents: bool,
    pinned: bool,
    searched: str,
) -> DatasetMetadata:
    """Return the card to build a release that the output does not hold and the plan cannot fetch.

    Returns:
        The card of the installed connector that builds exactly ``reference``.

    Raises:
        TimeNetMissingParentError: If ``build_parents`` is not set.
        TimeNetAccessError: If the output is not local and the card does not declare ``access: open``.
    """  # noqa: DOC502 - the access error is raised by require_open_access
    if not build_parents:
        raise TimeNetMissingParentError(reference, needed_by, pinned, searched)
    card = _exact_card(reference)
    if not isinstance(output, LocalRegistry):
        require_open_access(card)
    return card


def _exact_card(reference: DatasetRef) -> DatasetMetadata:
    """Return the card of the installed connector that builds exactly ``reference``.

    Returns:
        The card of the connector.

    Raises:
        TimeNetDatasetNotFoundError: If no installed connector builds the dataset.
        TimeFValidationError: If the installed connector declares another version.
        TimeNetInvalidCardError: If the card of the installed connector cannot be read.
    """  # noqa: DOC502 - find_card raises the card error
    card = find_card(reference.dataset_id)
    if card is None:
        raise TimeNetDatasetNotFoundError(f"parent release {reference} is missing and no installed connector builds it")
    if card.dataset_version != reference.version:
        raise TimeFValidationError(
            f"parent release {reference} is missing and the installed connector builds "
            f"version {card.dataset_version}, not that exact version"
        )
    return card
