"""Plan release copies and builds with dependencies first, before any connector runs."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from graphlib import CycleError, TopologicalSorter
from typing import ClassVar, Literal, TypeAlias

from timenet.builders import InstalledBuilder, find_card
from timenet.errors import (
    TimeFFormatError,
    TimeFValidationError,
    TimeNetDatasetNotFoundError,
    TimeNetFileError,
    TimeNetMissingParentError,
    TimeNetNotBuiltError,
)
from timenet.manifest import FilePart, Manifest, PayloadSource
from timenet.registry.base import BaseRegistry
from timenet.registry.dependencies import (
    CopySources,
    check_pins,
    find_payload_sources,
    validate_manifest_reference,
)
from timenet.registry.local import LocalRegistry, is_same_registry
from timenet.registry.version import DatasetVersion
from timenet.sources import (
    Found,
    FoundInConnector,
    FoundInRegistry,
    InstalledConnectors,
    NotFound,
    NotServed,
    ReleaseLookup,
    ReleaseSource,
    resolve_release,
)
from timenet.types.access import require_open_access
from timenet.types.references import DatasetRef
from timenet.types.version import Version


@dataclass(frozen=True)
class ReadStep:
    """A planned read of one release in the registry that holds it."""

    release: DatasetRef
    registry: BaseRegistry
    """The registry to read from."""
    depends_on: tuple[DatasetRef, ...] = ()
    """The releases with steps that must finish before this step runs."""
    action: ClassVar[Literal["read"]] = "read"
    """The kind of step, for display."""


@dataclass(frozen=True)
class CopyStep:
    """A planned copy of one release from a registry into the local registry."""

    release: DatasetRef
    registry: BaseRegistry
    """The registry to copy from."""
    depends_on: tuple[DatasetRef, ...] = ()
    """The releases with steps that must finish before this step runs."""
    with_payload_sources: bool = True
    """Also copy the payload sources that the manifest of the release lists."""
    force: bool = False
    """Copy the release again, also if the local registry holds it."""
    action: ClassVar[Literal["copy"]] = "copy"
    """The kind of step, for display."""


@dataclass(frozen=True)
class BuildStep:
    """A planned build of one release by an installed connector into the local registry."""

    release: DatasetRef
    builder: InstalledBuilder = field(hash=False)
    """The installed connector that builds the release."""
    depends_on: tuple[DatasetRef, ...] = ()
    """The releases with steps that must finish before this step runs."""
    pinned_by: Mapping[DatasetRef, PayloadSource] = field(default_factory=dict, hash=False)
    """The releases that read this release's files, mapped to their expected file records.

    A build must reproduce every recorded file before it publishes the release.
    """
    force: bool = False
    """Build the release again, also if the local registry holds it."""
    action: ClassVar[Literal["build"]] = "build"
    """The kind of step, for display."""


Step: TypeAlias = ReadStep | CopyStep | BuildStep
"""A planned read, copy, or build of one release."""


@dataclass(frozen=True)
class Plan:
    """The steps of a build or read in execution order."""

    steps: tuple[Step, ...]


Role = Literal["dataset", "payload-source"]
"""The release's role as a dataset or a source of payload files.

A release needs exactly the releases that its manifest lists as payload sources, never the payload sources of a
payload source. So a ``"dataset"`` needs its payload sources to open, and a ``"payload-source"`` supplies only its
own files.
"""


@dataclass(frozen=True)
class _Release:
    """A required release with its role, its need, its planned step and its dependencies."""

    role: Role
    """The release's role as a dataset or a source of payload files."""
    needed_in_local_registry: bool
    """True if the plan needs a complete copy in the output, False if a source that serves the release is enough."""
    step: Step | None
    """The step before dependencies are set, or ``None`` for a release that the output holds."""
    dependencies: tuple[DatasetRef, ...]
    """The parent or payload source releases that must be ready first."""


@dataclass
class Lookups:
    """A cache of source answers and release manifests for one plan.

    The cache stays valid only while no step runs, so use one instance for one plan only.
    """

    answers: dict[tuple[int, DatasetRef], ReleaseLookup] = field(default_factory=dict)
    """Source answers keyed by source identity and release.

    Keep each source alive while its answers are cached.
    """
    manifests: dict[tuple[int, DatasetRef], Manifest | None] = field(default_factory=dict)
    """Cached manifests keyed by registry identity and release, with ``None`` for missing releases."""

    def resolve(self, sources: Sequence[ReleaseSource], dataset_id: str, version: str | None) -> ReleaseLookup:
        """Resolve a release from ordered sources and cache answers for exact versions.

        Args:
            sources: The sources to search, in order.
            dataset_id: The dataset ID.
            version: An exact version, ``"latest"``, or ``None``, as in :func:`~timenet.sources.resolve_release`.

        Returns:
            The selected ``Found`` answer, otherwise the first ``NotServed`` answer or ``NotFound()``.
        """
        return resolve_release(sources, dataset_id, version, ask=self._ask)

    def ask(self, source: ReleaseSource, reference: DatasetRef) -> ReleaseLookup:
        """Return a cached answer for an exact release, or ask the source and cache it.

        Args:
            source: The source to query.
            reference: The exact release.

        Returns:
            The source's answer for the release.
        """
        key = (id(source), reference)
        if key not in self.answers:
            self.answers[key] = source.resolve(reference.dataset_id, reference.version)
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
            if isinstance(answer, FoundInRegistry | NotServed) and answer.manifest is not None:
                validate_manifest_reference(answer.manifest, reference)
                self.manifests[key] = answer.manifest
            else:
                self.manifests[key] = registry.find_manifest(reference)
        return self.manifests[key]

    def _ask(self, source: ReleaseSource, dataset_id: str, version: Version | None) -> ReleaseLookup:
        """Return the answer of a source for a release, from the cache for an exact version."""
        if version is not None:
            return self.ask(source, DatasetRef(dataset_id=dataset_id, version=version))
        answer = source.resolve(dataset_id, None)
        if isinstance(answer, Found):
            # A later request for the exact version gets the same answer.
            self.answers.setdefault((id(source), DatasetRef(dataset_id=dataset_id, version=answer.version)), answer)
        return answer


@dataclass
class _Graph:
    """A graph of required releases and their read, copy or build steps."""

    output: LocalRegistry | None
    """The local registry for copies and builds, or ``None`` for a read that copies and builds nothing."""
    sources: Sequence[ReleaseSource]
    """The sources in preference order. A read reads the output like any other source."""
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

    @cached_property
    def build_sources(self) -> list[ReleaseSource]:
        """The sources without the output, which is never a source of its own copies and builds."""
        output = self.output
        return [source for source in self.sources if output is None or not is_same_registry(source, output)]

    def visit(  # ruff: ignore[too-many-arguments]
        self,
        reference: DatasetRef,
        needed_by: DatasetRef | None,
        pinned: PayloadSource | None = None,
        *,
        needed_in_local_registry: bool,
        answer: ReleaseLookup | None = None,
        force: bool = False,
    ) -> None:
        """Add a release and its dependencies to the graph.

        A node is planned again when a later visit needs more of it: a dataset instead of a payload source, or a
        copy in the output instead of a read.

        Args:
            reference: The release to add.
            needed_by: The release that requires ``reference``, or ``None`` for the requested release.
            pinned: The file records that ``needed_by`` requires, or ``None`` for a parent dataset.
            needed_in_local_registry: True if the plan needs a complete copy in the output.
            answer: The answer of the sources for ``reference``, or ``None`` to ask them.
            force: Copy the release again, even if the output holds a complete copy.
        """
        role: Role = "dataset" if pinned is None else "payload-source"
        if pinned is not None and needed_by is not None:
            self.pins.setdefault(reference, {})[needed_by] = pinned
        found = self.releases.get(reference)
        if found is not None:
            role = "dataset" if "dataset" in {found.role, role} else "payload-source"
            needed_in_local_registry = found.needed_in_local_registry or needed_in_local_registry
            if (role, needed_in_local_registry) == (found.role, found.needed_in_local_registry):
                return
        if needed_in_local_registry:
            self.local(reference, needed_by, role, force=force)
        else:
            self.read(reference, needed_by, role, answer)

    def read(
        self, reference: DatasetRef, needed_by: DatasetRef | None, role: Role, answer: ReleaseLookup | None
    ) -> None:
        """Plan a release that a source can serve, and plan its payload sources if it is a dataset.

        Args:
            reference: The exact release.
            needed_by: The release that needs ``reference``, or ``None`` for the requested release.
            role: Whether the release is a dataset or a payload source.
            answer: The answer of the sources for ``reference``, or ``None`` to ask them.

        Raises:
            TimeNetFileError: If a source other than a registry provides the release and the output holds a
                damaged copy.
            TimeFValidationError: If only installed connectors provide the release and the plan has no output.
            TimeNetDatasetNotFoundError: If the registry of the answer no longer holds the release.
            TimeNetAccessError: If a source lists the release but no source serves it.
            TimeNetMissingParentError: If no source provides the payload source.
        """  # ruff: ignore[docstring-extraneous-exception] - the file, access and missing errors are raised as values
        if answer is None:
            answer = self.lookups.resolve(self.sources, reference.dataset_id, str(reference.version))
        if isinstance(answer, FoundInRegistry):
            manifest = self.lookups.manifest(answer.registry, reference)
            if manifest is None:
                raise TimeNetDatasetNotFoundError(f"{answer.registry} no longer holds {reference}")
            # The check of the local files waits for the end of the walk, because a later copy can repair them.
            sources = list(manifest.payload_sources.values()) if role == "dataset" else []
            step = ReadStep(reference, answer.registry)
            self.releases[reference] = _Release(role, False, step, tuple(source.release for source in sources))
            for source in sources:
                self.visit(source.release, reference, source, needed_in_local_registry=False)
            return
        # Nothing but a registry replaces a damaged copy in the output.
        damage = self.in_output(reference)[1]
        if damage is not None:
            raise damage
        if isinstance(answer, FoundInConnector):
            if self.output is None:
                raise TimeFValidationError(
                    f"{reference} needs a build by {InstalledConnectors()}, but the read has no local registry to "
                    "build into"
                )
            self.local(reference, needed_by, role)
        elif isinstance(answer, NotServed):
            raise answer.access_error(reference.dataset_id)
        else:
            raise self.missing(reference, needed_by, role, self.sources)

    def local(self, reference: DatasetRef, needed_by: DatasetRef | None, role: Role, *, force: bool = False) -> None:
        """Plan a complete copy of a release in the output, from the output itself, a copy or a build.

        Args:
            reference: The exact release.
            needed_by: The release that needs ``reference``, or ``None`` for the requested release.
            role: Whether the release is a dataset or a payload source.
            force: Copy the release again, even if the output holds a complete copy.

        Raises:
            TimeFValidationError: If the plan has no output.
            TimeNetNotBuiltError: If the plan builds nothing and only installed connectors provide the release.
            TimeNetDatasetNotFoundError: If no source provides the requested release, or ``force`` is set and only
                the output holds it.
            TimeNetFileError: If the output holds a damaged copy that no other registry provides.
            TimeNetAccessError: If a source lists the release but no source serves it.
            TimeNetMissingParentError: If no source provides the release.
        """  # ruff: ignore[docstring-extraneous-exception] - the file, access and missing errors are raised as values
        output = self.output
        if output is None:
            raise TimeFValidationError(f"{reference} must be in a local registry, but the read has none")
        held, damage = self.in_output(reference)
        if held is not None and not force:
            if self.open_only:
                require_open_access(held.metadata, held.lineage)
            sources = list(held.payload_sources.values()) if role == "dataset" else []
            self.releases[reference] = _Release(role, True, None, tuple(source.release for source in sources))
            for source in sources:
                self.visit(source.release, reference, source, needed_in_local_registry=True)
            return
        answer = self.lookups.resolve(self.build_sources, reference.dataset_id, str(reference.version))
        if isinstance(answer, FoundInRegistry):
            self.copy(reference, role, answer.registry, force=force)
        elif damage is not None:
            # A copy from another registry repairs a damaged copy. Nothing else replaces it.
            raise damage
        elif held is not None:
            # With force, the copy needs a registry other than the output, and none provides the release.
            raise TimeNetDatasetNotFoundError(
                f"No registry other than the local registry provides {reference}, so download(force=True) cannot copy "
                "it again."
            )
        elif isinstance(answer, FoundInConnector):
            if not self.builds:
                # A download builds nothing.
                raise TimeNetNotBuiltError(reference, answer.builder.card)
            self.build(reference, role, answer.builder)
        elif isinstance(answer, NotServed):
            raise answer.access_error(reference.dataset_id)
        else:
            # No source provides the release, or only the output does.
            raise self.missing(reference, needed_by, role, (output, *self.build_sources))

    def copy(self, reference: DatasetRef, role: Role, registry: BaseRegistry, *, force: bool = False) -> None:
        """Add a copy step and plan the payload sources that the copy does not bring.

        A dataset copy brings each payload source that ``registry`` serves. Each other payload source that the output
        lacks gets its own step first.

        Args:
            reference: The exact release to copy.
            role: Whether the release is a dataset or a payload source.
            registry: The registry to copy from.
            force: Copy the release again, even if the output holds a complete copy.

        Raises:
            TimeNetDatasetNotFoundError: If the registry lacks the release.
            TimeFFormatError: If a payload source that the copy brings or reuses fails a file pin.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by find_payload_sources
        manifest = self.lookups.manifest(registry, reference)
        if manifest is None:
            raise TimeNetDatasetNotFoundError(f"{registry} no longer holds {reference}")
        if self.open_only:
            require_open_access(manifest.metadata, manifest.lineage)
        sources = CopySources()
        if role == "dataset":
            sources = find_payload_sources(
                reference,
                manifest,
                registry,
                find=lambda source: self.listed(registry, source),
                held=lambda source: self.in_output(source)[0],
            )
        if self.open_only:
            for found in (*sources.fetched.values(), *sources.held.values()):
                require_open_access(found.metadata, found.lineage)
        step = CopyStep(reference, registry, with_payload_sources=role == "dataset", force=force)
        # The node holds its final dependencies before the walk visits them. So a release of the walk
        # that depends on it closes a cycle and does not look it up again.
        self.releases[reference] = _Release(role, True, step, tuple(sources.missing))
        for source, pinned in sources.missing.items():
            self.visit(source, reference, pinned, needed_in_local_registry=True)

    def build(self, reference: DatasetRef, role: Role, builder: InstalledBuilder) -> None:
        """Add a build step and plan the parents declared in its card.

        Args:
            reference: The release to build.
            role: Whether the release is a dataset or a payload source.
            builder: The installed connector that builds the release.
        """
        card = builder.card
        if self.open_only:
            require_open_access(card)
        parents = tuple(card.parents.values())
        self.releases[reference] = _Release(role, True, BuildStep(reference, builder), parents)
        # A build opens each of its parents with the payload sources of that parent.
        for parent in parents:
            self.visit(parent, reference, needed_in_local_registry=True)

    @staticmethod
    def missing(
        reference: DatasetRef, needed_by: DatasetRef | None, role: Role, searched: Iterable[ReleaseSource]
    ) -> TimeNetDatasetNotFoundError:
        """Create the error for a release that no source provides.

        Args:
            reference: The missing release.
            needed_by: The release that needs ``reference``, or ``None`` for the requested release.
            role: Whether the release is a dataset or a payload source.
            searched: The searched sources, in order.

        Returns:
            A ``TimeNetDatasetNotFoundError`` for the requested release, else a ``TimeNetMissingParentError``.
        """
        if needed_by is None:
            names = ", ".join(str(source) for source in searched)
            return TimeNetDatasetNotFoundError(f"{reference} is in none of the sources: {names}.")
        described = describe_searched_sources(searched, reference)
        return TimeNetMissingParentError(reference, needed_by, described, payload_source=role == "payload-source")

    def listed(self, registry: BaseRegistry, reference: DatasetRef) -> Manifest | None:
        """Get a release's manifest if the registry holds the release, also if it does not serve it.

        Returns:
            The manifest, or ``None`` if the registry does not hold the release.
        """
        if isinstance(self.lookups.ask(registry, reference), NotFound):
            return None
        return self.lookups.manifest(registry, reference)

    def in_output(self, reference: DatasetRef) -> tuple[Manifest | None, TimeNetFileError | None]:
        """Find a release in the output and check local files for damage.

        Missing files and incorrect file sizes count as damage in a local registry.

        Returns:
            ``(manifest, None)`` if found, ``(None, error)`` if damaged, or ``(None, None)`` if absent or if the
            plan has no output.
        """
        if self.output is None:
            return None, None
        manifest = self.lookups.manifest(self.output, reference)
        if manifest is None:
            return manifest, None
        if reference not in self.damage:
            self.damage[reference] = self.output.damage(reference, manifest)
        damage = self.damage[reference]
        return (manifest, None) if damage is None else (None, damage)

    def check(self) -> None:
        """Check the pins of each pinned release, then the files of each release that a read opens in place.

        Raises:
            TimeFFormatError: If a read, copied or held release fails a file pin, or two readers of a build pin one
                file differently.
            TimeNetDatasetNotFoundError: If a registry no longer holds a release of the plan.
            TimeNetFileError: If a local file of a release that a read opens is missing or has another size.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by check_pins and open_release
        for reference, readers in self.pins.items():
            step = self.releases[reference].step
            if isinstance(step, BuildStep):
                # One build must match every reader, so the readers must pin each file alike.
                _check_agreement(reference, readers)
                continue
            # A read or a copy uses the release in its registry. A release without a step is intact in the output.
            registry = self.output if step is None else step.registry
            manifest = None if registry is None else self.lookups.manifest(registry, reference)
            if manifest is None:
                raise TimeNetDatasetNotFoundError(f"{registry} no longer holds {reference}")
            for reader, pinned in readers.items():
                check_pins(reader, pinned, manifest)
        for reference, release in self.releases.items():
            step = release.step
            if isinstance(step, ReadStep) and isinstance(step.registry, LocalRegistry):
                # Opening a local release checks each of its files.
                step.registry.open_release(reference, manifest=self.lookups.manifest(step.registry, reference))

    def open(self, reference: DatasetRef) -> DatasetVersion:
        """Open a release of the plan without its payload sources, after the steps of the plan ran.

        Returns:
            The release in the registry of its read step, else in the output.

        Raises:
            TimeFValidationError: If the release has no read step and the plan has no output.
        """
        step = self.releases[reference].step
        if isinstance(step, ReadStep):
            return step.registry.open_release(reference, manifest=self.lookups.manifest(step.registry, reference))
        if self.output is None:
            # Without an output, the planner plans only read steps. So this is a bug of the planner.
            raise TimeFValidationError(f"{reference} has no read step and the plan of the read has no output")
        # The steps changed the output, so the release opens with a manifest that is read again.
        return self.output.open_release(reference)

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
        steps: list[Step] = []
        for reference in self.order():
            release = self.releases[reference]
            # The order puts each dependency first, so its nearest releases with a step are known.
            needs: dict[DatasetRef, None] = {}
            for need in release.dependencies:
                needs.update(dict.fromkeys((need,) if self.releases[need].step is not None else nearest[need]))
            nearest[reference] = tuple(needs)
            step = release.step
            if isinstance(step, BuildStep):
                step = replace(step, pinned_by=self.pins.get(reference, {}))
            if step is not None:
                steps.append(replace(step, depends_on=nearest[reference]))
        return steps


def _check_agreement(release: DatasetRef, readers: Mapping[DatasetRef, PayloadSource]) -> None:
    """Check that the readers of a release pin each of its files alike.

    Args:
        release: The release whose files are pinned.
        readers: The releases that pin files of ``release``, with their file requirements.

    Raises:
        TimeFFormatError: If two readers pin one file as different files.
    """
    pinned: dict[str, tuple[DatasetRef, FilePart]] = {}
    for reader, source in readers.items():
        for part in source.files:
            first_reader, first_part = pinned.setdefault(part.path, (reader, part))
            if first_part != part:
                raise TimeFFormatError(
                    f"{first_reader} and {reader} pin {part.path!r} of {release} as different files, "
                    f"so no build of {release} can match both"
                )


def plan_steps(
    builder: InstalledBuilder,
    output: LocalRegistry,
    *,
    parents_from: Sequence[ReleaseSource],
    force_build: bool,
    open_access_only: bool,
) -> tuple[DatasetRef, list[Step]]:
    """Plan the steps to build a dataset and its dependencies.

    A damaged copy of the requested release in the output gets a forced build.

    Args:
        builder: The installed connector that builds the dataset.
        output: The destination registry.
        parents_from: The sources to search for missing releases, in order.
        force_build: Rebuild the requested release, even if the output holds it.
        open_access_only: Require open access for every release.

    Returns:
        The requested release and its steps, with its own build last if needed.
    """
    card = builder.card
    if open_access_only:
        require_open_access(card)
    root = builder.release
    # The graph leaves the output out of the sources of its copies and builds.
    graph = _Graph(output, parents_from, open_only=open_access_only)
    # If the output already holds the requested release, a reader needs only its payload sources to
    # open it. Its declared parents matter only to a rebuild.
    held, damage = (None, None) if force_build else graph.in_output(root)
    if held is not None and open_access_only:
        # The installed card has no lineage, so the check above does not see an ancestor that is not open.
        require_open_access(held.metadata, held.lineage)
    # The graph holds the requested release before the walk starts. So a release that depends on it
    # closes a cycle, and the walk does not look it up again.
    if held is None:
        step = BuildStep(root, builder, force=force_build or damage is not None)
        graph.releases[root] = _Release("dataset", True, step, tuple(card.parents.values()))
        for parent in card.parents.values():
            graph.visit(parent, root, needed_in_local_registry=True)
    else:
        sources = list(held.payload_sources.values())
        graph.releases[root] = _Release("dataset", True, None, tuple(source.release for source in sources))
        for source in sources:
            graph.visit(source.release, root, source, needed_in_local_registry=True)
    graph.check()
    return root, graph.create_ordered_steps()


def plan_download(
    release: DatasetRef,
    store: LocalRegistry,
    *,
    sources: Sequence[ReleaseSource],
    force: bool = False,
    lookups: Lookups | None = None,
) -> Plan:
    """Plan the copies that put a release and its payload sources into the local registry.

    Each release that the local registry lacks comes from the first registry other than the local registry
    that provides it. The plan builds nothing.

    Args:
        release: The exact release.
        store: The local registry to copy into.
        sources: The sources to search in order, excluding registries at the directory of ``store``.
        force: Copy the release again, even if the local registry holds a complete copy.
        lookups: The answers and manifests that resolved ``release``, or ``None`` to ask the sources.

    Returns:
        The copy steps, in the order in which they run.

    Raises:
        TimeNetNotBuiltError: If only installed connectors provide a release that the local registry lacks.
        TimeNetDatasetNotFoundError: If ``force`` is set and only the local registry holds the release.
        TimeNetFileError: If the local registry holds a damaged copy that no other registry provides.
        TimeNetAccessError: If a source lists a release that the local registry lacks but no source serves it.
        TimeNetMissingParentError: If no registry provides a payload source that the local registry lacks.
        TimeFFormatError: If a payload source differs from the files that a release pins.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the walk and as values
    others = [source for source in sources if not is_same_registry(source, store)]
    graph = _Graph(store, others, open_only=False, builds=False, lookups=lookups or Lookups())
    graph.visit(release, None, needed_in_local_registry=True, force=force)
    graph.check()
    return Plan(tuple(graph.create_ordered_steps()))


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
