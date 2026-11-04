"""Plan and prepare the parent releases needed to build a dataset.

Each :class:`Step` copies or builds one release after the releases it needs.
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
class Step:
    """A plan step that copies or builds one release.

    A copy includes the release and its missing payload sources.
    """

    release: DatasetRef
    action: Literal["copy", "build"]
    depends_on: tuple[DatasetRef, ...] = ()
    """The releases with steps that must finish before this step runs."""
    pinned_by: Mapping[DatasetRef, PayloadSource] = field(default_factory=dict, hash=False)
    """The releases that read this release's files, mapped to their expected file records.

    A build must reproduce every recorded file before it publishes the release.
    """


@dataclass(frozen=True)
class Plan:
    """The copy and build steps for a dataset, in execution order."""

    steps: tuple[Step, ...]


@dataclass(frozen=True)
class _Release:
    """A required release, its dependencies, and its planned action."""

    opened: bool
    """True if the build opens the release with its payload sources. False if the release is only a
    payload source of another release, and the build reads only its pinned files."""
    action: Literal["copy", "build"] | None
    """The planned action, or ``None`` if the output already holds the release."""
    dependencies: tuple[DatasetRef, ...]
    """The releases that must be in place before this release. A release to build depends on the
    parents on its card. An opened release that the plan does not build depends on its payload
    sources. Other releases depend on nothing."""


@dataclass
class _Graph:
    """The releases a build needs and the dependencies between them."""

    output: WritableRegistry
    fetch_from: BaseRegistry | None
    build_parents: bool
    open_only: bool
    """Whether each visited release and its lineage must declare ``access: open``."""
    searched: str
    """The registries that the plan looks in, for the message of a missing release."""
    releases: dict[DatasetRef, _Release] = field(default_factory=dict)
    """The nodes, in the order in which the walk found them."""
    fetched: set[DatasetRef] = field(default_factory=set)
    """The payload sources that the plan copies as part of another release."""
    pins: dict[DatasetRef, dict[DatasetRef, PayloadSource]] = field(default_factory=dict)
    """For each payload source: each release that reads its files, with its pins on them."""

    def visit(self, reference: DatasetRef, needed_by: DatasetRef, pinned: PayloadSource | None = None) -> None:
        """Add a release and its required dependencies to the graph.

        Use :meth:`order` to check the completed graph for cycles.

        Args:
            reference: The release to add.
            needed_by: The release that needs ``reference``.
            pinned: File pins from ``needed_by``, or ``None`` to open ``reference`` as a parent.
        """
        opened = pinned is None
        if pinned is not None:
            self.pins.setdefault(reference, {})[needed_by] = pinned
        found = self.releases.get(reference)
        # If the walk found a payload source first, and later finds it as a parent, it looks at the
        # release again with its payload sources.
        if found is not None and (found.opened or not opened):
            return
        manifest, action = _locate(self.output, self.fetch_from, reference, open_only=self.open_only)
        if manifest is None:
            card = _card_to_build(
                reference,
                needed_by,
                build_parents=self.build_parents,
                open_only=self.open_only,
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
            # A copy of the release that reads this payload source also copies the payload source.
            if action == "copy":
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

    def steps(self, requested: DatasetRef) -> list[Step]:
        """Create copy and build steps with dependencies first.

        Each step names only dependencies that also have steps.

        Args:
            requested: The requested release, which gets a copy step if needed but no build step.

        Returns:
            The steps in dependency order.

        Raises:
            TimeFValidationError: If the releases form a dependency cycle.
        """  # ruff: ignore[docstring-extraneous-exception] - order() raises the cycle error
        chosen: dict[DatasetRef, Literal["copy", "build"]] = {}
        for reference in self.order():
            release = self.releases[reference]
            if reference == requested and release.action == "build":
                continue
            if release.action == "build":
                chosen[reference] = "build"
            # A copy of an opened release also copies each payload source that the output does not
            # hold. So if the plan copies a payload source only with the release that reads it, the
            # plan copies that release again, even if the output holds it.
            elif release.opened and (
                release.action == "copy" or any(source in self.fetched for source in release.dependencies)
            ):
                chosen[reference] = "copy"
        return [
            Step(
                reference,
                action,
                depends_on=tuple(need for need in self.releases[reference].dependencies if need in chosen),
                pinned_by=self.pins.get(reference, {}) if action == "build" else {},
            )
            for reference, action in chosen.items()
        ]


def plan_parents(  # ruff: ignore[too-many-arguments]
    dataset_id: str,
    output: WritableRegistry,
    *,
    fetch_from: BaseRegistry | None = None,
    build_parents: bool = False,
    force_build: bool = False,
    open_access_only: bool = False,
) -> list[Step]:
    """Plan the copy and build steps for a dataset's parents.

    Copy missing parents and their payload sources from ``fetch_from``, or build them if enabled.
    Remote registries and ``open_access_only`` require open access for releases and their lineage.

    Args:
        dataset_id: The dataset to build.
        output: The registry that the build writes to.
        fetch_from: The registry to copy missing parents from for a local output, or ``None``.
        build_parents: Build missing parents with their installed connectors.
        force_build: Plan parents for a rebuild even if the requested release already exists.
        open_access_only: Require ``access: open`` for each release, including the requested release.

    Returns:
        The copy and build steps, with dependencies first.

    Raises:
        LookupError: If no installed connector declares a card for ``dataset_id``.
        TimeNetInvalidCardError: If an installed connector's card cannot be read.
        TimeFValidationError: If parents form a cycle, versions conflict, or a copy targets a remote output.
        TimeNetMissingParentError: If a parent or payload source is missing and ``build_parents`` is false.
        TimeNetDatasetNotFoundError: If a parent is missing and no installed connector builds it.
        TimeNetFileError: If a local parent is damaged and no replacement is available.
        TimeNetAccessError: If a release or its lineage lacks required open access.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the helpers of the walk
    if fetch_from is not None and not isinstance(output, LocalRegistry):
        # A registry that is not local accepts a child only if it already holds the parents. A copy
        # step writes a parent only to a local directory, never to such a registry.
        raise TimeFValidationError(
            "copying parents from a registry needs a local output; publish the parents to the registry first"
        )
    card = find_card(dataset_id)
    if card is None:
        raise LookupError(f"no installed connector declares a card for {dataset_id!r}")
    open_only = open_access_only or output.requires_open_access
    if open_only:
        require_open_access(card)
    # For a dataset without parents, the plan does not read the output, so the build makes no extra request.
    if not card.parents:
        return []
    root = DatasetRef(dataset_id=card.dataset_id, version=card.dataset_version)
    # If the output already holds the requested release, a reader needs only its payload sources to
    # open it. Its declared parents matter only to a rebuild.
    held = None if force_build else _manifest(output, root)
    if held is not None and open_only:
        # The installed card has no lineage, so the check above does not see an ancestor that is not open.
        require_open_access(held.metadata, held.lineage)
    searched = str(output) if fetch_from is None else f"{output} or {fetch_from}"
    graph = _Graph(output, fetch_from, build_parents=build_parents, open_only=open_only, searched=searched)
    # The graph holds the requested release before the walk starts. So a release that depends on it
    # closes a cycle, and the walk does not look it up again.
    if held is None:
        graph.releases[root] = _Release(opened=True, action="build", dependencies=tuple(card.parents.values()))
        for parent in card.parents.values():
            graph.visit(parent, root)
    else:
        sources = list(held.payload_sources.values())
        graph.releases[root] = _Release(
            opened=True, action=None, dependencies=tuple(source.release for source in sources)
        )
        for source in sources:
            graph.visit(source.release, root, source)
    return graph.steps(root)


def prepare_parents(  # ruff: ignore[too-many-arguments]
    dataset_id: str,
    output: WritableRegistry,
    *,
    build: Callable[[Step], object],
    fetch_from: BaseRegistry | None = None,
    build_parents: bool = False,
    force_build: bool = False,
) -> None:
    """Copy or build a dataset's required parent releases.

    The ``build`` callback must publish each release only if its files match the step's
    ``pinned_by`` records.

    Args:
        dataset_id: The dataset to build.
        output: The destination registry.
        build: The function that builds one parent into ``output``.
        fetch_from: The registry to copy missing parents from, or ``None`` to copy nothing.
        build_parents: Build missing parents with their installed connectors.
        force_build: Prepare a rebuild's parents even if ``output`` holds the requested release.

    Raises:
        TimeNetDatasetNotFoundError: If ``build`` does not publish the release of its step.
    """
    steps = plan_parents(
        dataset_id, output, fetch_from=fetch_from, build_parents=build_parents, force_build=force_build
    )
    for step in steps:
        if step.action == "copy":
            prepare_local_copy(output.local_root, step.release, fetch_from=fetch_from)
        else:
            build(step)
            # After a successful build, the output must hold the exact release with its payload sources.
            if isinstance(output, LocalRegistry):
                prepare_local_copy(output.root, step.release)
            else:
                if _manifest(output, step.release) is None:
                    raise TimeNetDatasetNotFoundError(f"builder did not publish parent release {step.release}")
                output.open_version(step.release.dataset_id, str(step.release.version))


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
    output: WritableRegistry, fetch_from: BaseRegistry | None, reference: DatasetRef, *, open_only: bool
) -> tuple[Manifest | None, Literal["copy"] | None]:
    # The damage is raised from a variable, the access error by require_open_access.
    # ruff: ignore[docstring-extraneous-exception]
    """Find a release in the output or the source registry.

    A local copy is intact when all files exist at their recorded sizes.
    Enforce open access for the release and its lineage when ``open_only`` or a copy requires it.

    Returns:
        The manifest and action (``None`` to reuse, ``"copy"`` to fetch), or ``(None, None)`` if absent.

    Raises:
        TimeNetFileError: If the local copy is damaged and the source registry lacks the release.
        TimeNetAccessError: If the release or its lineage lacks required open access.
    """
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
        if open_only:
            require_open_access(manifest.metadata, manifest.lineage)
        return manifest, None
    fetched = None if fetch_from is None else _manifest(fetch_from, reference)
    if fetched is not None:
        # A registry that requires open access serves only releases whose cards declare access: open.
        # So the plan refuses another release before any download.
        if open_only or (fetch_from is not None and fetch_from.requires_open_access):
            require_open_access(fetched.metadata, fetched.lineage)
        return fetched, "copy"
    if damage is not None:
        raise damage
    return None, None


def _card_to_build(  # ruff: ignore[too-many-arguments]
    reference: DatasetRef,
    needed_by: DatasetRef,
    *,
    build_parents: bool,
    open_only: bool,
    pinned: bool,
    searched: str,
) -> DatasetMetadata:
    """Find a connector card for building a missing release.

    Returns:
        The connector card for the exact dataset ID and version in ``reference``.

    Raises:
        TimeNetMissingParentError: If ``build_parents`` is false.
        TimeNetAccessError: If ``open_only`` is set and the card does not declare ``access: open``.
    """  # ruff: ignore[docstring-extraneous-exception] - the access error is raised by require_open_access
    if not build_parents:
        raise TimeNetMissingParentError(reference, needed_by, searched, payload_source=pinned)
    card = _exact_card(reference)
    if open_only:
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
    """  # ruff: ignore[docstring-extraneous-exception] - find_card raises the card error
    card = find_card(reference.dataset_id)
    if card is None:
        raise TimeNetDatasetNotFoundError(f"parent release {reference} is missing and no installed connector builds it")
    if card.dataset_version != reference.version:
        raise TimeFValidationError(
            f"parent release {reference} is missing and the installed connector builds "
            f"version {card.dataset_version}, not that exact version"
        )
    return card
