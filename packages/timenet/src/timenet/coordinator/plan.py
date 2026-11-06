"""Plan release copies and builds with dependencies first, before any connector runs."""

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from graphlib import CycleError, TopologicalSorter
from typing import Literal, cast

from timenet.builders import declared, find_card
from timenet.errors import (
    TimeFValidationError,
    TimeNetDatasetNotFoundError,
    TimeNetFileError,
    TimeNetMissingParentError,
    TimeNetNotBuiltError,
)
from timenet.manifest import Manifest, PayloadSource
from timenet.registry.base import BaseRegistry
from timenet.registry.dependencies import check_pins, validate_manifest_reference
from timenet.registry.local import LocalRegistry, is_same_registry
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
    """A planned read, copy, or build of one release.

    Reads open releases in ``entry``, and copies fetch them with their payload sources.
    Builds use ``entry`` as the installed connector source.
    """

    release: DatasetRef
    action: Literal["read", "copy", "build"]
    entry: ReleaseSource
    """The source registry for a read or copy, or the installed connectors for a build."""
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
    """The steps of a build or read in execution order."""

    steps: tuple[Step, ...]


Role = Literal["dataset", "payload-source"]
"""The release's role as a dataset or a source of payload files.

A ``"dataset"`` requires its payload sources to open. A ``"payload-source"`` supplies only its own files.
"""


@dataclass(frozen=True)
class _Release:
    """A required release with its role, planned step and dependencies."""

    role: Role
    """The release's role as a dataset or a source of payload files."""
    step: Step | None
    """The copy or build step before dependencies are set, or ``None`` for an existing release."""
    dependencies: tuple[DatasetRef, ...]
    """The parent or payload source releases that must be ready first."""


@dataclass
class Lookups:
    """A cache of source answers and release manifests for one plan.

    The cache stays valid only while no step runs, so use one instance for one plan only.
    """

    answers: dict[tuple[int, DatasetRef], Answer] = field(default_factory=dict)
    """Source answers keyed by source identity and release.

    Keep each source alive while its answers are cached.
    """
    manifests: dict[tuple[int, DatasetRef], Manifest | None] = field(default_factory=dict)
    """Cached manifests keyed by registry identity and release, with ``None`` for missing releases."""

    def resolve(self, sources: Sequence[ReleaseSource], dataset_id: str, version: str | None) -> Answer:
        """Resolve a release from ordered sources and cache answers for exact versions.

        Exact versions use the first matching source, and latest requests use the highest available version.

        Args:
            sources: The sources to search, in order.
            dataset_id: The dataset ID.
            version: An exact version, or ``None`` for the latest version.

        Returns:
            The selected ``Found`` answer, otherwise the first ``NotServed`` answer or ``NotFound()``.
        """
        answer = resolve_release([_Remembered(source, self) for source in sources], dataset_id, version)
        if isinstance(answer, Found):
            self.answers.setdefault(
                (id(answer.entry), DatasetRef(dataset_id=dataset_id, version=answer.version)), answer
            )
        return answer

    def ask(self, source: ReleaseSource, reference: DatasetRef) -> Answer:
        """Return a cached answer for an exact release, or ask the source and cache it.

        Args:
            source: The source to query.
            reference: The exact release.

        Returns:
            The source's answer for the release.
        """
        key = (id(source), reference)
        if key not in self.answers:
            self.answers[key] = source.resolve(reference.dataset_id, str(reference.version))
        return self.answers[key]

    def manifest(self, registry: BaseRegistry, reference: DatasetRef) -> Manifest | None:
        """Return a cached release manifest, or read and cache it.

        A manifest in a cached source answer takes precedence over a registry read.

        Returns:
            The manifest, or ``None`` if the release is absent from the registry.
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


@dataclass
class _Graph:
    """A graph of releases needed for a build."""

    output: LocalRegistry
    sources: Sequence[ReleaseSource]
    """Sources for missing releases in preference order, excluding the output."""
    open_only: bool
    """True if each release that the walk visits must declare ``access: open``."""
    builds: bool = True
    """Whether the plan allows builds from installed connectors."""
    releases: dict[DatasetRef, _Release] = field(default_factory=dict)
    """The nodes, in the order in which the walk found them."""
    pins: dict[DatasetRef, dict[DatasetRef, PayloadSource]] = field(default_factory=dict)
    """For each payload source: each release that reads its files, with its pins on them."""
    lookups: Lookups = field(default_factory=Lookups)
    """The cache of source answers and release manifests."""
    damage: dict[DatasetRef, TimeNetFileError | None] = field(default_factory=dict)
    """The first file error per checked local release, or ``None`` if all file sizes match."""

    def visit(self, reference: DatasetRef, needed_by: DatasetRef, pinned: PayloadSource | None = None) -> None:
        """Add a required release and its dependencies to the graph.

        Args:
            reference: The release to add.
            needed_by: The release that requires ``reference``.
            pinned: Required file records, or ``None`` for a build parent.

        Raises:
            TimeNetFileError: If the output holds a damaged copy that no other registry provides.
            TimeNetAccessError: If sources list the release but do not provide it.
            TimeNetMissingParentError: If no source provides the release.
            TimeNetNotBuiltError: If only connectors provide the release and ``builds`` is false.
            TimeFValidationError: If the source is neither a registry nor ``InstalledConnectors``.
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
        answer = self.lookups.resolve(self.sources, reference.dataset_id, str(reference.version))
        entry = answer.entry if isinstance(answer, Found) else None
        if isinstance(entry, BaseRegistry) and not is_same_registry(entry, self.output):
            self.copy(reference, role, entry)
        elif damage is not None:
            # A copy from another registry repairs a damaged copy. Nothing else replaces it.
            raise damage
        elif isinstance(entry, InstalledConnectors):
            if not self.builds:
                # A download builds nothing.
                raise TimeNetNotBuiltError(reference, _not_built(reference))
            self.build(reference, role, entry)
        elif isinstance(answer, NotServed):
            raise answer.access_error(reference.dataset_id)
        elif entry is None or is_same_registry(entry, self.output):
            searched = describe_searched_sources((self.output, *self.sources), reference)
            raise TimeNetMissingParentError(reference, needed_by, searched, payload_source=role == "payload-source")
        else:
            raise TimeFValidationError(
                f"{entry} provides {reference}, but a build copies only from a registry and builds only "
                "with InstalledConnectors()"
            )

    def copy(self, reference: DatasetRef, role: Role, registry: BaseRegistry, *, force: bool = False) -> None:
        """Add a copy step and plan its required payload sources.

        Args:
            reference: The exact release to copy.
            role: Whether the release is a dataset or a payload source.
            registry: The registry to copy from.
            force: Copy the release again, even if the output holds a complete copy.

        Raises:
            TimeNetDatasetNotFoundError: If the registry lacks the release.
        """
        manifest = self.lookups.manifest(registry, reference)
        if manifest is None:
            raise TimeNetDatasetNotFoundError(f"{registry} no longer holds {reference}")
        if self.open_only:
            require_open_access(manifest.metadata, manifest.lineage)
        # The node is in the graph before the walk, so that a release of the walk that depends on it
        # closes a cycle and does not look it up again.
        step = Step(reference, "copy", registry, force=force)
        self.releases[reference] = _Release(role, step, ())
        steps = self.resolve_dependencies_of_copy(reference, registry, manifest)
        self.releases[reference] = _Release(role, step, steps)

    def resolve_dependencies_of_copy(
        self, release: DatasetRef, registry: BaseRegistry, manifest: Manifest
    ) -> tuple[DatasetRef, ...]:
        """Plan separate steps for payload sources unavailable to a copy.

        The copy includes the payload sources that ``registry`` serves, at every depth, and reuses complete output
        copies. Other payload sources need separate steps before the copy.

        Args:
            release: The release to copy.
            registry: The registry to copy from.
            manifest: The manifest of the release in ``registry``.

        Returns:
            The payload source releases that need separate steps.

        Raises:
            TimeFFormatError: If a payload source fails a required file pin.
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
        """Add a build step and plan the parents declared in its card.

        Args:
            reference: The release to build.
            role: Whether the release is a dataset or a payload source.
            connectors: The installed connector source for the build.

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

    def served(self, registry: BaseRegistry, reference: DatasetRef) -> Manifest | None:
        """Return a release manifest if the registry serves the release.

        Returns:
            The manifest, or ``None`` if the registry cannot serve the release.
        """
        answer = self.lookups.ask(registry, reference)
        return self.lookups.manifest(registry, reference) if isinstance(answer, Found) else None

    def in_output(self, reference: DatasetRef) -> tuple[Manifest | None, TimeNetFileError | None]:
        """Find a release in the output and check local files for damage.

        Missing files and incorrect file sizes count as damage in a local registry.

        Returns:
            ``(manifest, None)`` if found, ``(None, error)`` if damaged, or ``(None, None)`` if absent.
        """
        manifest = self.lookups.manifest(self.output, reference)
        if manifest is None:
            return manifest, None
        if reference not in self.damage:
            self.damage[reference] = self.output.damage(reference, manifest)
        damage = self.damage[reference]
        return (manifest, None) if damage is None else (None, damage)

    def check_copied_pins(self) -> None:
        """Check that copied payload sources match every pinned file requirement.

        Raises:
            TimeFFormatError: If a copied payload source conflicts with a file pin.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by check_pins
        for reference, readers in self.pins.items():
            step = self.releases[reference].step
            # The plan gives each copy step a registry as its entry.
            manifest = (
                None
                if step is None or step.action != "copy"
                else self.lookups.manifest(cast("BaseRegistry", step.entry), reference)
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
        """Create copy and build steps in dependency order.

        Returns:
            The ordered steps, including the requested build if needed.

        Raises:
            TimeFValidationError: If the dependencies form a cycle.
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
    """A source wrapper that caches answers for exact versions."""

    def __init__(self, source: ReleaseSource, lookups: Lookups) -> None:
        self._source = source
        self._lookups = lookups

    def resolve(self, dataset_id: str, version: str | None) -> Answer:
        """Find a release, using the cache for exact versions.

        Args:
            dataset_id: The dataset ID.
            version: An exact version, or ``None``, ``""`` or ``"latest"`` for the latest version.

        Returns:
            The source's answer for the release.
        """
        # The plan asks only for exact versions, so it keeps only those answers.
        if is_latest(version):
            return self._source.resolve(dataset_id, version)
        return self._lookups.ask(self._source, DatasetRef.parse(f"{dataset_id}@{version}"))


def plan_steps(  # ruff: ignore[too-many-arguments]
    dataset_id: str,
    output: LocalRegistry,
    *,
    parents_from: Sequence[ReleaseSource],
    force_build: bool,
    open_access_only: bool,
    pinned_by: Mapping[DatasetRef, PayloadSource] | None = None,
    lookups: Lookups | None = None,
) -> tuple[DatasetRef, list[Step]]:
    """Plan the steps to build a dataset and its dependencies.

    A damaged copy of the requested release in the output gets a forced build.

    Args:
        dataset_id: The dataset to build.
        output: The destination registry.
        parents_from: The sources to search for missing releases, in order.
        force_build: Rebuild the requested release, even if the output holds it.
        open_access_only: Require open access for every release.
        pinned_by: The required file records for this release, keyed by the releases that need them.
        lookups: Cached source answers and manifests to reuse.

    Returns:
        The requested release and its steps, with its own build last if needed.

    Raises:
        TimeNetDatasetNotFoundError: If no installed connector builds the dataset.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by declared
    # The output is never a source for itself.
    parents_from = [source for source in parents_from if not is_same_registry(source, output)]
    card = declared(dataset_id).card
    open_only = open_access_only
    if open_only:
        require_open_access(card)
    root = DatasetRef(dataset_id=card.dataset_id, version=card.dataset_version)
    graph = _Graph(output, parents_from, open_only=open_only, lookups=lookups or Lookups())
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


def plan_download(
    release: DatasetRef,
    store: LocalRegistry,
    *,
    sources: Sequence[ReleaseSource],
    force: bool = False,
    lookups: Lookups | None = None,
) -> Plan:
    """Plan the copies that put a release and its payload sources into the store.

    Each release that the store lacks comes from the first registry other than the store that
    provides it. The plan builds nothing.

    Args:
        release: The exact release.
        store: The destination store.
        sources: The sources to search in order, excluding registries at the store's directory.
        force: Copy the release again, even if the store holds a complete copy.
        lookups: Cached source answers and manifests, or ``None`` to query sources.

    Returns:
        The copy steps, in the order in which they run.

    Raises:
        TimeNetNotBuiltError: If only installed connectors provide a release that the store lacks.
        TimeNetDatasetNotFoundError: If ``force`` is set and only the store holds the release.
        TimeNetFileError: If the store holds a damaged copy that no other registry provides.
        TimeNetAccessError: If a source lists a release that the store lacks but no source serves it.
        TimeNetMissingParentError: If no registry provides a payload source that the store lacks.
        TimeFFormatError: If a payload source differs from the files that a release pins.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the walk and as values
    others = [source for source in sources if not is_same_registry(source, store)]
    graph = _Graph(store, others, open_only=False, builds=False, lookups=lookups or Lookups())
    held, damage = graph.in_output(release)
    if held is not None and not force:
        # The store holds a complete copy, so only the payload sources that it lacks need a copy.
        pins = list(held.payload_sources.values())
        graph.releases[release] = _Release("dataset", None, tuple(pinned.release for pinned in pins))
        for pinned in pins:
            graph.visit(pinned.release, release, pinned)
    else:
        _add_release_copy(graph, release, held=held is not None, damage=damage, force=force)
    graph.check_copied_pins()
    return Plan(tuple(graph.create_ordered_steps()))


def _add_release_copy(
    graph: _Graph, release: DatasetRef, *, held: bool, damage: TimeNetFileError | None, force: bool
) -> None:
    """Add a release copy to a download plan.

    Args:
        graph: The graph of the download, whose sources do not hold the store.
        release: The exact release.
        held: Whether the store holds a copy of the release.
        damage: The first damaged file of the copy in the store, or ``None``.
        force: Copy the release again, even if the store holds a complete copy.

    Raises:
        TimeNetFileError: If the store holds a damaged copy that no other registry provides.
        TimeNetNotBuiltError: If only installed connectors provide the release.
        TimeNetDatasetNotFoundError: If no registry other than the store provides the release.
        TimeNetAccessError: If a registry lists the release but serves none of its files.
    """  # ruff: ignore[docstring-extraneous-exception] - raised as values
    answer = graph.lookups.resolve(graph.sources, release.dataset_id, str(release.version))
    entry = answer.entry if isinstance(answer, Found) else None
    if isinstance(entry, BaseRegistry):
        # A registry other than the store provides the release, so the plan copies it from there.
        graph.copy(release, "dataset", entry, force=force)
    elif damage is not None:
        # Only a registry replaces a damaged copy in the store.
        raise damage
    elif held:
        # With force, the copy needs a registry other than the store, and none provides the release.
        raise TimeNetDatasetNotFoundError(f"no registry other than the store provides {release} to copy again")
    elif isinstance(answer, NotServed):
        # A registry lists the release but serves none of its files.
        raise answer.access_error(release.dataset_id)
    elif isinstance(entry, InstalledConnectors):
        # Only installed connectors provide the release, and the plan builds nothing.
        raise TimeNetNotBuiltError(release, _not_built(release))
    else:
        # No source other than the store provides the release.
        names = ", ".join(str(source) for source in graph.sources)
        raise TimeNetDatasetNotFoundError(f"{release} is in none of the sources: {names}.")


def _not_built(release: DatasetRef) -> str:
    """Describe a release that only installed connectors provide, in a plan that builds nothing.

    Args:
        release: The release.

    Returns:
        The error message.
    """
    return f"{release} is not built yet: only an installed connector provides it."


def describe_searched_sources(sources: Iterable[ReleaseSource], reference: DatasetRef) -> str:
    """Describe the sources searched for a missing release.

    Args:
        sources: The searched sources, in order.
        reference: The missing release.

    Returns:
        Source names joined with " or ", including connector versions that differ from the requested version.
    """
    names = []
    for source in sources:
        card = find_card(reference.dataset_id) if isinstance(source, InstalledConnectors) else None
        if card is not None and card.dataset_version != reference.version:
            names.append(f"{source} ({reference.dataset_id} builds {card.dataset_version})")
        else:
            names.append(str(source))
    return " or ".join(names)
