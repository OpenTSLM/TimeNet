"""Plan release copies and builds with dependencies first, before any connector runs."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from graphlib import CycleError, TopologicalSorter
from typing import ClassVar, Literal, TypeAlias

from timenet.builders import InstalledBuilder, find_card
from timenet.errors import (
    TimeFFormatError,
    TimeFValidationError,
    TimeNetDatasetNotFoundError,
    TimeNetFileError,
    TimeNetMissingParentError,
)
from timenet.manifest import FilePart, Manifest, PayloadSource
from timenet.registry.base import BaseRegistry
from timenet.registry.dependencies import (
    CopySources,
    check_pins,
    find_payload_sources,
    validate_manifest_reference,
)
from timenet.registry.local import LocalRegistry
from timenet.sources import (
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
class CopyStep:
    """A planned copy of one release from a registry into the local registry."""

    release: DatasetRef
    registry: BaseRegistry
    """The registry to copy from."""
    depends_on: tuple[DatasetRef, ...] = ()
    """The releases with steps that must finish before this step runs."""
    with_payload_sources: bool = True
    """Also copy the payload sources that the manifest of the release lists."""
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


Step: TypeAlias = CopyStep | BuildStep
"""A planned copy or build of one release."""


@dataclass(frozen=True)
class Plan:
    """The copy and build steps for a dataset, in execution order."""

    steps: tuple[Step, ...]


Role = Literal["dataset", "payload-source"]
"""The release's role as a dataset or a source of payload files.

A release needs exactly the releases that its manifest lists as payload sources, never the payload sources of a
payload source. So a ``"dataset"`` needs its payload sources to open, and a ``"payload-source"`` supplies only its
own files.
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
class _Graph:
    """A graph of required releases and their copy or build steps."""

    output: LocalRegistry
    sources: Sequence[ReleaseSource]
    """Sources for missing releases in preference order, excluding the output."""
    open_only: bool
    """True if each release that the walk visits must declare ``access: open``."""
    releases: dict[DatasetRef, _Release] = field(default_factory=dict)
    """The nodes, in the order in which the walk found them."""
    pins: dict[DatasetRef, dict[DatasetRef, PayloadSource]] = field(default_factory=dict)
    """For each payload source: each release that reads its files, with its pins on them."""
    answers: dict[tuple[int, DatasetRef], ReleaseLookup] = field(default_factory=dict)
    """Cached source answers by source identity and release."""
    manifests: dict[tuple[int, DatasetRef], Manifest | None] = field(default_factory=dict)
    """Cached manifests keyed by registry identity and release, with ``None`` for missing releases."""
    damage: dict[DatasetRef, TimeNetFileError | None] = field(default_factory=dict)
    """The first file error per checked local release, or ``None`` if all file sizes match."""

    def visit(self, reference: DatasetRef, needed_by: DatasetRef, pinned: PayloadSource | None = None) -> None:
        """Add a release and its dependencies to the graph.

        The graph revisits an existing entry only when its role changes from payload source to dataset.

        Args:
            reference: The release to add.
            needed_by: The release that needs ``reference``.
            pinned: File pins from ``needed_by``, or ``None`` for a parent dataset.

        Raises:
            TimeNetFileError: If the output holds a damaged copy that no other registry provides.
            TimeNetAccessError: If a source lists the release but no source serves it.
            TimeNetMissingParentError: If no source provides the release.
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
            sources = list(held.payload_sources.values()) if role == "dataset" else []
            self.releases[reference] = _Release(role, None, tuple(source.release for source in sources))
            for source in sources:
                self.visit(source.release, reference, source)
            return
        answer = self.resolve(reference)
        if isinstance(answer, FoundInRegistry) and not _is_same_registry(answer.registry, self.output):
            self.copy(reference, role, answer.registry)
        elif damage is not None:
            # A copy from another registry repairs a damaged copy. Nothing else replaces it.
            raise damage
        elif isinstance(answer, FoundInConnector):
            self.build(reference, role, answer.builder)
        elif isinstance(answer, NotServed):
            raise answer.access_error(reference.dataset_id)
        else:
            # No source provides the release, or only the output does.
            raise TimeNetMissingParentError(
                reference, needed_by, self.describe_searched_sources(reference), payload_source=role == "payload-source"
            )

    def copy(self, reference: DatasetRef, role: Role, registry: BaseRegistry) -> None:
        """Add a copy step and plan the payload sources that the copy does not bring.

        A dataset copy brings each payload source that ``registry`` serves. Each other payload source that the output
        lacks gets its own step first.

        Args:
            reference: The release to copy.
            role: Whether the release is a dataset or a payload source.
            registry: The registry to copy from.

        Raises:
            TimeNetDatasetNotFoundError: If the registry no longer holds the release.
            TimeFFormatError: If a payload source that the copy brings or reuses fails a file pin.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by find_payload_sources
        manifest = self.manifest(registry, reference)
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
        step = CopyStep(reference, registry, with_payload_sources=role == "dataset")
        # The node holds its final dependencies before the walk visits them. So a release of the walk
        # that depends on it closes a cycle and does not look it up again.
        self.releases[reference] = _Release(role, step, tuple(sources.missing))
        for source, pinned in sources.missing.items():
            self.visit(source, reference, pinned)

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
        self.releases[reference] = _Release(role, BuildStep(reference, builder), parents)
        # A build opens each of its parents with the payload sources of that parent.
        for parent in parents:
            self.visit(parent, reference)

    def resolve(self, reference: DatasetRef) -> ReleaseLookup:
        """Find an exact release across the sources in preference order.

        Args:
            reference: The exact release to find.

        Returns:
            The first ``Found`` answer, otherwise the first ``NotServed`` answer or ``NotFound()``.
        """
        return resolve_release(self.sources, reference.dataset_id, str(reference.version), ask=self._ask)

    def ask(self, source: ReleaseSource, reference: DatasetRef) -> ReleaseLookup:
        """Get and cache a source's answer for an exact release.

        Args:
            source: The source to query.
            reference: The exact release to find.

        Returns:
            The cached or newly fetched answer.
        """
        key = (id(source), reference)
        if key not in self.answers:
            self.answers[key] = source.resolve(reference.dataset_id, reference.version)
        return self.answers[key]

    def _ask(self, source: ReleaseSource, dataset_id: str, version: Version | None) -> ReleaseLookup:
        """Return the answer of a source for a release, from :meth:`ask` for an exact version."""
        # The plan asks only for exact versions, so it keeps only those answers.
        if version is None:
            return source.resolve(dataset_id, None)
        return self.ask(source, DatasetRef(dataset_id=dataset_id, version=version))

    def listed(self, registry: BaseRegistry, reference: DatasetRef) -> Manifest | None:
        """Get a release's manifest if the registry holds the release, also if it does not serve it.

        Returns:
            The manifest, or ``None`` if the registry does not hold the release.
        """
        if isinstance(self.ask(registry, reference), NotFound):
            return None
        return self.manifest(registry, reference)

    def manifest(self, registry: BaseRegistry, reference: DatasetRef) -> Manifest | None:
        """Read and cache a release's manifest.

        A manifest in a cached source answer takes precedence over a registry read.

        Returns:
            The manifest, or ``None`` if the registry does not hold the release.
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

    def in_output(self, reference: DatasetRef) -> tuple[Manifest | None, TimeNetFileError | None]:
        """Check a release in the output for missing files or incorrect file sizes.

        Returns:
            ``(manifest, None)`` if intact, ``(None, error)`` if damaged, else ``(None, None)``.
        """
        manifest = self.manifest(self.output, reference)
        if manifest is None:
            return manifest, None
        if reference not in self.damage:
            self.damage[reference] = self.output.damage(reference, manifest)
        damage = self.damage[reference]
        return (manifest, None) if damage is None else (None, damage)

    def describe_searched_sources(self, reference: DatasetRef) -> str:
        """Describe the output and sources searched for a missing release.

        Connector descriptions include their declared version when it differs from the requested version.

        Args:
            reference: The missing release.

        Returns:
            The source descriptions joined with ``" or "``.
        """
        names = [str(self.output)]
        for source in self.sources:
            card = find_card(reference.dataset_id) if isinstance(source, InstalledConnectors) else None
            if card is not None and card.dataset_version != reference.version:
                names.append(f"{source} ({reference.dataset_id} builds {card.dataset_version})")
            else:
                names.append(str(source))
        return " or ".join(names)

    def check(self) -> None:
        """Check each pinned release against the pins of every release that reads it.

        Raises:
            TimeFFormatError: If a copied or held release fails a file pin, or two readers of a build pin one file
                differently.
            TimeNetDatasetNotFoundError: If a registry no longer holds a copied or held release.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by check_pins
        for reference, readers in self.pins.items():
            step = self.releases[reference].step
            if isinstance(step, BuildStep):
                # One build must match every reader, so the readers must pin each file alike.
                _check_agreement(reference, readers)
                continue
            # A copy reads the release from its registry. A release without a step is intact in the output.
            registry = self.output if step is None else step.registry
            manifest = self.manifest(registry, reference)
            if manifest is None:
                raise TimeNetDatasetNotFoundError(f"{registry} no longer holds {reference}")
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


def plan_steps(  # ruff: ignore[too-many-arguments]
    builder: InstalledBuilder,
    output: LocalRegistry,
    *,
    parents_from: Sequence[ReleaseSource],
    force_build: bool,
    open_access_only: bool,
    pinned_by: Mapping[DatasetRef, PayloadSource] | None = None,
) -> tuple[DatasetRef, list[Step]]:
    """Plan a dataset build and its dependencies.

    A damaged copy of the requested release in the output gets a forced build.

    Args:
        builder: The installed connector that builds the dataset.
        output: The local registry that the build writes to.
        parents_from: Sources for missing dependencies, in preference order.
        force_build: Plan a rebuild even if the requested release exists.
        open_access_only: Require ``access: open`` for every needed release.
        pinned_by: File pins grouped by dependent release that the requested build must preserve.

    Returns:
        The requested release and ordered steps, with its own build last if needed.
    """
    # The output is never a source for itself.
    parents_from = [source for source in parents_from if not _is_same_registry(source, output)]
    card = builder.card
    if open_access_only:
        require_open_access(card)
    root = builder.release
    graph = _Graph(output, parents_from, open_only=open_access_only)
    if pinned_by:
        graph.pins[root] = dict(pinned_by)
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
    graph.check()
    return root, graph.create_ordered_steps()


def _is_same_registry(source: ReleaseSource, output: LocalRegistry) -> bool:
    """Check whether a source is the output registry.

    Args:
        source: The source to compare.
        output: The destination registry.

    Returns:
        ``True`` for the same object or local registries with the same resolved directory.
    """
    if source is output:
        return True
    return isinstance(source, LocalRegistry) and source.root.resolve() == output.root.resolve()
