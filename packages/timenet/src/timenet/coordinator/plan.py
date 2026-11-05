"""Plan how to obtain the parent releases that a build needs, before any connector runs.

A plan is a list of :class:`Step` objects. Each step copies a release from a registry or builds it
with :class:`~timenet.sources.InstalledConnectors`, after the steps of the releases that it depends
on.
"""

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from graphlib import CycleError, TopologicalSorter
from typing import Literal, cast

from timenet.builders import declared, find_card
from timenet.errors import (
    TimeFValidationError,
    TimeNetDatasetNotFoundError,
    TimeNetFileError,
    TimeNetMissingParentError,
)
from timenet.manifest import Manifest, PayloadSource
from timenet.registry.base import BaseRegistry
from timenet.registry.dependencies import check_pins, validate_manifest_reference
from timenet.registry.local import LocalRegistry
from timenet.sources import (
    Answer,
    Found,
    InstalledConnectors,
    NotServed,
    ReleaseSource,
    resolve_release,
)
from timenet.types.access import require_open_access
from timenet.types.references import DatasetRef, is_latest


@dataclass(frozen=True)
class Step:
    """A plan step that copies or builds one release.

    A copy takes the release from the registry ``entry``, with every payload source that it needs. A
    build builds the release with ``entry``, an :class:`~timenet.sources.InstalledConnectors`.
    """

    release: DatasetRef
    action: Literal["copy", "build"]
    entry: ReleaseSource
    """For a copy: the registry to copy from. For a build: the installed connectors."""
    depends_on: tuple[DatasetRef, ...] = ()
    """The releases with steps that must finish before this step runs."""
    pinned_by: Mapping[DatasetRef, PayloadSource] = field(default_factory=dict, hash=False)
    """The releases that read this release's files, mapped to their expected file records.

    A build must reproduce every recorded file before it publishes the release.
    """
    force: bool = False
    """Build or copy the release again, also if the store holds it."""


@dataclass(frozen=True)
class Plan:
    """The copy and build steps for a dataset, in execution order."""

    steps: tuple[Step, ...]


Role = Literal["dataset", "payload-source"]
"""How the build uses a release. It opens a "dataset" and needs its payload sources too. It only
reads files of a "payload-source"."""


@dataclass(frozen=True)
class _Release:
    """A required release, its dependencies, and its planned action."""

    role: Role
    """How the build uses the release."""
    step: Step | None
    """The copy or the build, without its dependencies, or ``None`` if the output holds the release."""
    dependencies: tuple[DatasetRef, ...]
    """The releases that must be in place before this release. A release to build depends on the
    parents on its card. A release to copy depends on each payload source of its copy that gets a
    step of its own. A dataset that the output holds depends on its payload sources."""


@dataclass
class _Graph:
    """The releases a build needs and the dependencies between them."""

    output: LocalRegistry
    sources: Sequence[ReleaseSource]
    """The sources of the releases that the output lacks, in order, without the output."""
    open_only: bool
    """True if each release that the walk visits must declare ``access: open``."""
    releases: dict[DatasetRef, _Release] = field(default_factory=dict)
    """The nodes, in the order in which the walk found them."""
    pins: dict[DatasetRef, dict[DatasetRef, PayloadSource]] = field(default_factory=dict)
    """For each payload source: each release that reads its files, with its pins on them."""
    answers: dict[tuple[int, DatasetRef], Answer] = field(default_factory=dict)
    """Each answer that the plan got, by source and release. The keys use ``id()``, because a source
    need not be hashable. Each such source stays alive while the plan runs, because it is in the
    list."""
    manifests: dict[tuple[int, DatasetRef], Manifest | None] = field(default_factory=dict)
    """Each manifest that the plan used, by registry and release, or ``None`` for a release that the
    registry does not hold. A manifest comes from the answer of the registry. The plan reads it only
    from the output, and from a local registry, which answers from its files without a manifest."""
    damage: dict[DatasetRef, TimeNetFileError | None] = field(default_factory=dict)
    """For each release that a local output holds: the error of its first missing or resized file."""

    def visit(self, reference: DatasetRef, needed_by: DatasetRef, pinned: PayloadSource | None = None) -> None:
        """Add a release and its required dependencies to the graph.

        The walk does not look for cycles. It stops at a release that it already found as a dataset,
        and at a payload source that it already found.

        Args:
            reference: The release to add.
            needed_by: The release that needs ``reference``.
            pinned: The pins of ``needed_by`` on the files of ``reference``, if ``reference`` is a
                payload source of ``needed_by``. ``None`` if the build opens ``reference`` as a
                parent.

        Raises:
            TimeNetFileError: If a local output holds a damaged copy, and no registry other than the
                output provides the release.
            TimeNetAccessError: If the first source that lists the release does not serve its files,
                and no later source provides it.
            TimeNetMissingParentError: If no source provides the release.
            TimeFValidationError: If a source answers with an entry that is neither a registry nor
                the installed connectors.
        """  # ruff: ignore[docstring-extraneous-exception] - the file and access errors are raised as values
        role: Role = "dataset" if pinned is None else "payload-source"
        if pinned is not None:
            self.pins.setdefault(reference, {})[needed_by] = pinned
        found = self.releases.get(reference)
        # If the walk found a payload source first, and later finds it as a parent, it looks at the
        # release again with its payload sources.
        if found is not None and (found.role == "dataset" or role == "payload-source"):
            return
        held, damage = self.in_output(reference)
        if held is not None:
            if self.open_only:
                require_open_access(held.metadata, held.lineage)
            # To open a release, the build needs its direct payload sources. These are the releases
            # whose files the release reads. The build does not need their own payload sources.
            sources = list(held.payload_sources.values()) if role == "dataset" else []
            self.releases[reference] = _Release(role, None, tuple(source.release for source in sources))
            for source in sources:
                self.visit(source.release, reference, source)
            return
        answer = self.resolve(reference)
        entry = answer.entry if isinstance(answer, Found) else None
        if isinstance(entry, BaseRegistry) and not _is_same_registry(entry, self.output):
            self.copy(reference, role, entry)
        elif damage is not None:
            # A copy from another registry repairs a damaged copy. Nothing else replaces it.
            raise damage
        elif isinstance(entry, InstalledConnectors):
            self.build(reference, role, entry)
        elif isinstance(answer, NotServed):
            raise answer.access_error(reference.dataset_id)
        elif entry is None or _is_same_registry(entry, self.output):
            raise TimeNetMissingParentError(
                reference, needed_by, self.describe_searched_sources(reference), payload_source=role == "payload-source"
            )
        else:
            raise TimeFValidationError(
                f"{entry} provides {reference}, but a build copies only from a registry and builds only "
                "with InstalledConnectors()"
            )

    def copy(self, reference: DatasetRef, role: Role, registry: BaseRegistry) -> None:
        """Add a release that the plan copies from a registry, and walk the payload sources of the copy.

        Args:
            reference: The release.
            role: How the build uses the release.
            registry: The registry to copy from.

        Raises:
            TimeNetDatasetNotFoundError: If the registry no longer holds the release.
        """
        manifest = self.manifest(registry, reference)
        if manifest is None:
            raise TimeNetDatasetNotFoundError(f"{registry} no longer holds {reference}")
        if self.open_only:
            require_open_access(manifest.metadata, manifest.lineage)
        # The node is in the graph before the walk, so that a release of the walk that depends on it
        # closes a cycle and does not look it up again.
        self.releases[reference] = _Release(role, Step(reference, "copy", registry), ())
        steps = self.resolve_dependencies_of_copy(reference, registry, manifest)
        self.releases[reference] = _Release(role, Step(reference, "copy", registry), steps)

    def resolve_dependencies_of_copy(
        self, release: DatasetRef, registry: BaseRegistry, manifest: Manifest
    ) -> tuple[DatasetRef, ...]:
        """Plan the payload sources that a copy of a release cannot bring, and return them.

        A copy from ``registry`` brings each payload source that ``registry`` serves, at every depth,
        and uses each payload source that the output holds complete. Each other payload source gets
        a step of its own, and the copy step runs after it. Each payload source must match the files
        that the release that reads it pins.

        Args:
            release: The release to copy.
            registry: The registry to copy from.
            manifest: The manifest of the release in ``registry``.

        Returns:
            The payload sources that get a step of their own.

        Raises:
            TimeFFormatError: If a payload source differs from the files that a release that reads it
                pins.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by check_pins
        steps: list[DatasetRef] = []
        readers = deque([(release, manifest)])
        walked = {release}
        while readers:
            reader, read = readers.popleft()
            for pinned in read.payload_sources.values():
                source = pinned.release
                held = self.served(registry, source)
                if held is None:
                    held = self.in_output(source)[0]
                if held is None:
                    self.visit(source, reader, pinned)
                    if source not in steps:
                        steps.append(source)
                    continue
                check_pins(reader, pinned, held)
                if self.open_only:
                    require_open_access(held.metadata, held.lineage)
                if source not in walked:
                    walked.add(source)
                    readers.append((source, held))
        return tuple(steps)

    def build(self, reference: DatasetRef, role: Role, connectors: InstalledConnectors) -> None:
        """Add a release that the plan builds, then visit the parents on its card.

        Args:
            reference: The release.
            role: How the build uses the release.
            connectors: The source that builds it.

        Raises:
            TimeNetDatasetNotFoundError: If no installed connector builds the release.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by declared
        card = declared(reference.dataset_id, reference.version).card
        if self.open_only:
            require_open_access(card)
        parents = tuple(card.parents.values())
        self.releases[reference] = _Release(role, Step(reference, "build", connectors), parents)
        # A build opens each of its parents with the payload sources of that parent.
        for parent in parents:
            self.visit(parent, reference)

    def resolve(self, reference: DatasetRef) -> Answer:
        """Ask the sources for the exact version of a release, by the rules of :mod:`timenet.sources`.

        Args:
            reference: The release.

        Returns:
            The answer of the first source that provides the release, or else the first
            ``NotServed``, or else ``NotFound()``.
        """
        sources = [_Remembered(source, self) for source in self.sources]
        return resolve_release(sources, reference.dataset_id, str(reference.version))

    def ask(self, source: ReleaseSource, reference: DatasetRef) -> Answer:
        """Ask one source for the exact version of a release, once in this plan.

        Args:
            source: The source.
            reference: The release.

        Returns:
            The answer of the source.
        """
        key = (id(source), reference)
        if key not in self.answers:
            self.answers[key] = source.resolve(reference.dataset_id, str(reference.version))
        return self.answers[key]

    def served(self, registry: BaseRegistry, reference: DatasetRef) -> Manifest | None:
        """Return the manifest of a release that a registry serves.

        Returns:
            The manifest, or ``None`` if the registry does not hold the release, or lists it without
            serving its files.
        """
        answer = self.ask(registry, reference)
        return self.manifest(registry, reference) if isinstance(answer, Found) else None

    def manifest(self, registry: BaseRegistry, reference: DatasetRef) -> Manifest | None:
        """Return the manifest of a release in a registry, once in this plan.

        The manifest comes from the answer of the registry, if the plan asked it and the answer
        carries one. Else the plan reads it: from the output, and from a local registry, which
        answers from its files.

        Returns:
            The manifest, or ``None`` if the registry does not hold the release.
        """
        key = (id(registry), reference)
        if key not in self.manifests:
            answer = self.answers.get(key)
            if isinstance(answer, Found) and answer.manifest is not None:
                validate_manifest_reference(answer.manifest, reference)
                self.manifests[key] = answer.manifest
            else:
                self.manifests[key] = registry.find_manifest(reference)
        return self.manifests[key]

    def in_output(self, reference: DatasetRef) -> tuple[Manifest | None, TimeNetFileError | None]:
        """Look for an intact copy of a release in the output.

        In a local output, a copy with a missing file or a file of another size is damaged.

        Returns:
            The manifest of an intact copy and ``None``. For a damaged copy, ``None`` and the error
            that names its first missing or resized file. Else ``None`` and ``None``.
        """
        manifest = self.manifest(self.output, reference)
        if manifest is None:
            return manifest, None
        if reference not in self.damage:
            self.damage[reference] = self.output.damage(reference, manifest)
        damage = self.damage[reference]
        return (manifest, None) if damage is None else (None, damage)

    def describe_searched_sources(self, reference: DatasetRef) -> str:
        """Name the output and each source, for the message of a missing release.

        Args:
            reference: The missing release.

        Returns:
            The names, joined with " or ". For the installed connectors, the name also gives the
            version that the connector of the dataset builds, if it builds another version.
        """
        names = [str(self.output)]
        for source in self.sources:
            card = find_card(reference.dataset_id) if isinstance(source, InstalledConnectors) else None
            if card is not None and card.dataset_version != reference.version:
                names.append(f"{source} ({reference.dataset_id} builds {card.dataset_version})")
            else:
                names.append(str(source))
        return " or ".join(names)

    def check_copied_pins(self) -> None:
        """Make sure that each payload source that the plan copies matches the pins of its readers.

        Raises:
            TimeFFormatError: If the copy in the registry differs from the files that a release that
                reads it pins.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by check_pins
        for reference, readers in self.pins.items():
            step = self.releases[reference].step
            # The plan gives each copy step a registry as its entry.
            manifest = (
                None
                if step is None or step.action != "copy"
                else self.manifest(cast("BaseRegistry", step.entry), reference)
            )
            if manifest is None:
                continue
            for reader, pinned in readers.items():
                check_pins(reader, pinned, manifest)

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

    def create_ordered_steps(self) -> list[Step]:
        """Make one step for each release that the plan copies or builds, in the order of :meth:`order`.

        Returns:
            The steps, the build of the requested release included, each after the steps of the
            releases that it depends on. ``depends_on`` of a step names the nearest releases with a
            step that it depends on, at any depth.

        Raises:
            TimeFValidationError: If the releases form a dependency cycle.
        """  # ruff: ignore[docstring-extraneous-exception] - order() raises the cycle error
        nearest: dict[DatasetRef, tuple[DatasetRef, ...]] = {}
        steps = []
        for reference in self.order():
            release = self.releases[reference]
            # The order puts each dependency first, so its nearest releases with a step are known.
            needs: dict[DatasetRef, None] = {}
            for need in release.dependencies:
                needs.update(dict.fromkeys((need,) if self.releases[need].step is not None else nearest[need]))
            nearest[reference] = tuple(needs)
            if release.step is not None:
                pinned_by = self.pins.get(reference, {}) if release.step.action == "build" else {}
                steps.append(replace(release.step, depends_on=nearest[reference], pinned_by=pinned_by))
        return steps


class _Remembered:
    """A source whose answers come through :meth:`_Graph.ask`, so that the plan asks it once for a release."""

    def __init__(self, source: ReleaseSource, graph: _Graph) -> None:
        self._source = source
        self._graph = graph

    def resolve(self, dataset_id: str, version: str | None) -> Answer:
        """Answer for the source.

        Args:
            dataset_id: The dataset ID.
            version: An exact version, or ``None``, ``""`` or ``"latest"`` for the latest version.

        Returns:
            The answer of the source.
        """
        # The plan asks only for exact versions, so it keeps only those answers.
        if is_latest(version):
            return self._source.resolve(dataset_id, version)
        return self._graph.ask(self._source, DatasetRef.parse(f"{dataset_id}@{version}"))


def plan_steps(  # ruff: ignore[too-many-arguments]
    dataset_id: str,
    output: LocalRegistry,
    *,
    parents_from: Sequence[ReleaseSource],
    force_build: bool,
    open_access_only: bool,
    pinned_by: Mapping[DatasetRef, PayloadSource] | None = None,
) -> tuple[DatasetRef, list[Step]]:
    """Plan each step of a build of ``dataset_id``, its own build included.

    A damaged copy of the requested release in the output gets a forced build.

    Args:
        dataset_id: The dataset to build.
        output: The local registry that the build writes to.
        parents_from: Sources for missing dependencies, in preference order.
        force_build: Plan a rebuild even if the requested release exists.
        open_access_only: Require ``access: open`` for every needed release.
        pinned_by: File pins grouped by dependent release that the requested build must preserve.

    Returns:
        The requested release, and the steps. The build of the requested release comes last,
        because it depends on each other release of the plan. If the output holds the requested
        release and ``force_build`` is not set, the plan has no build of it.

    Raises:
        TimeNetDatasetNotFoundError: If no installed connector builds the dataset.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by declared
    # The output is never a source for itself.
    parents_from = [source for source in parents_from if not _is_same_registry(source, output)]
    card = declared(dataset_id).card
    open_only = open_access_only
    if open_only:
        require_open_access(card)
    root = DatasetRef(dataset_id=card.dataset_id, version=card.dataset_version)
    graph = _Graph(output, parents_from, open_only=open_only)
    if pinned_by:
        graph.pins[root] = dict(pinned_by)
    # If the output already holds the requested release, a reader needs only its payload sources to
    # open it. Its declared parents matter only to a rebuild.
    held, damage = (None, None) if force_build else graph.in_output(root)
    if held is not None and open_only:
        # The installed card has no lineage, so the check above does not see an ancestor that is not open.
        require_open_access(held.metadata, held.lineage)
    # The graph holds the requested release before the walk starts. So a release that depends on it
    # closes a cycle, and the walk does not look it up again.
    if held is None:
        step = Step(root, "build", InstalledConnectors(), force=force_build or damage is not None)
        graph.releases[root] = _Release(role="dataset", step=step, dependencies=tuple(card.parents.values()))
        for parent in card.parents.values():
            graph.visit(parent, root)
    else:
        sources = list(held.payload_sources.values())
        graph.releases[root] = _Release(
            role="dataset", step=None, dependencies=tuple(source.release for source in sources)
        )
        for source in sources:
            graph.visit(source.release, root, source)
    graph.check_copied_pins()
    return root, graph.create_ordered_steps()


def _is_same_registry(source: ReleaseSource, output: LocalRegistry) -> bool:
    """Return whether a source is the output, or a local registry over the directory of a local output.

    Args:
        source: The source.
        output: The registry that the build writes to.

    Returns:
        ``True`` if a copy from the source would copy the output onto itself.
    """
    if source is output:
        return True
    return isinstance(source, LocalRegistry) and source.root.resolve() == output.root.resolve()
