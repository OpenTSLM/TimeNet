"""Plan and read releases from ordered sources.

A read opens each release in the registry that provides it, so a release and its payload sources can come from
different registries. A release that only installed connectors provide is built into the local store first.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import NoReturn, cast

from timenet.builders import declared, find_card
from timenet.coordinator.build import run_plan
from timenet.coordinator.plan import Lookups, Plan, Step, describe_searched_sources, plan_steps
from timenet.errors import (
    TimeFFormatError,
    TimeFValidationError,
    TimeNetDatasetNotFoundError,
    TimeNetMissingParentError,
    TimeNetVersionUnavailableError,
)
from timenet.manifest import FilePart, Manifest, PayloadSource
from timenet.registry.base import BaseRegistry
from timenet.registry.dependencies import attach_payload_sources, check_pins
from timenet.registry.local import LocalRegistry
from timenet.registry.version import DatasetVersion
from timenet.sources import (
    Answer,
    Found,
    InstalledConnectors,
    NotFound,
    NotServed,
    ReleaseSource,
)
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
        version: An exact version, or ``None``, ``""`` or ``"latest"`` for the latest version.
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
        TimeFValidationError: If a build lacks a store, steps conflict, or a source type is unsupported.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by the planner and the checks
    planner = _ReadPlanner(sources, store)
    planner.plan(dataset_id, version)
    return Plan(tuple(planner.steps.values()))


def read_release(
    dataset_id: str,
    version: str | None,
    *,
    sources: Sequence[ReleaseSource],
    store: LocalRegistry | None,
) -> DatasetVersion:
    """Open a release with its payload sources after completing required copies and builds.

    This can raise any error listed by :func:`plan_read`.
    A local store is required only for copies and builds.

    Args:
        dataset_id: The dataset ID.
        version: An exact version, or ``None``, ``""`` or ``"latest"`` for the latest version.
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
    planner = _ReadPlanner(sources, store)
    release = planner.plan(dataset_id, version)
    plan = Plan(tuple(planner.steps.values()))
    if any(step.action != "read" for step in plan.steps):
        # The plan copies and builds only into a store.
        run_plan(plan, cast("LocalRegistry", store))
    return attach_payload_sources(
        planner.open(release), planner.open, searched=describe_searched_sources(planner.sources, release)
    )


def raise_not_found(dataset_id: str, version: str | None, sources: Sequence[ReleaseSource]) -> NoReturn:
    """Raise the error for a release that no source provides.

    For an exact version, an installed connector that builds another version gives a more exact error.

    Args:
        dataset_id: The dataset ID.
        version: An exact version, or ``None``, ``""`` or ``"latest"`` for the latest.
        sources: The sources searched for the release.

    Raises:
        TimeNetVersionUnavailableError: If the installed connector declares another version than the exact request.
        TimeNetInvalidCardError: If the card of the connector cannot be read.
        TimeNetDatasetNotFoundError: Otherwise, with a message that names the searched sources.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by declared
    requested = None if is_latest(version) else DatasetRef.parse(f"{dataset_id}@{version}")
    connectors = any(isinstance(source, InstalledConnectors) for source in sources)
    if requested is not None and connectors and find_card(dataset_id) is not None:
        # The installed connector builds another version.
        declared(dataset_id, requested.version)
    names = ", ".join(str(source) for source in sources)
    wanted = dataset_id if requested is None else str(requested)
    raise TimeNetDatasetNotFoundError(f"{wanted} is in none of the sources: {names}.")


@dataclass
class _ReadPlanner:
    """A planner that finds releases and collects the steps to read them."""

    sources: Sequence[ReleaseSource]
    store: LocalRegistry | None
    lookups: Lookups = field(default_factory=Lookups)
    """Cached source answers and release manifests."""
    steps: dict[DatasetRef, Step] = field(default_factory=dict)
    """Steps keyed by release, in execution order."""

    def plan(self, dataset_id: str, version: str | None) -> DatasetRef:
        """Plan the requested release and check its dependencies and local files.

        Returns:
            The requested release with an exact version.

        Raises:
            TimeNetDatasetNotFoundError: If no source provides the release.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by raise_not_found
        version = None if is_latest(version) else version
        answer = self.lookups.resolve(self.sources, dataset_id, version)
        if isinstance(answer, NotFound):
            if version is not None:
                self.check_store_copy(DatasetRef.parse(f"{dataset_id}@{version}"))
            raise_not_found(dataset_id, version, self.sources)
        release = DatasetRef(dataset_id=dataset_id, version=answer.version)
        registry = self.provide(release, answer)
        if registry is not None:
            payload_sources = list(self.manifest(registry, release).payload_sources.values())
            for pinned in payload_sources:
                self.add_payload_source(pinned, release)
            # A payload source without a step is in the store already.
            depends_on = tuple(pinned.release for pinned in payload_sources if pinned.release in self.steps)
            self.add(Step(release, "read", registry, depends_on=depends_on))
        self.check()
        return release

    def add_payload_source(self, pinned: PayloadSource, reader: DatasetRef) -> None:
        """Resolve a payload source and add the steps needed to read it.

        Raises:
            TimeNetMissingParentError: If no source provides the payload source.
        """
        source = pinned.release
        answer = self.lookups.resolve(self.sources, source.dataset_id, str(source.version))
        if isinstance(answer, NotFound):
            self.check_store_copy(source)
            raise TimeNetMissingParentError(
                source, reader, describe_searched_sources(self.sources, source), payload_source=True
            )
        registry = self.provide(source, answer, pinned_by={reader: pinned})
        if registry is not None:
            check_pins(reader, pinned, self.manifest(registry, source))
            self.add(Step(source, "read", registry))

    def provide(
        self, release: DatasetRef, answer: Answer, pinned_by: Mapping[DatasetRef, PayloadSource] | None = None
    ) -> BaseRegistry | None:
        """Select a registry or plan a local build for a release.

        Args:
            release: The exact release.
            answer: The ``Found`` or ``NotServed`` result of a source lookup.
            pinned_by: The releases that pin files of ``release``, with their file requirements.

        Returns:
            The registry that serves the release, or ``None`` when the plan uses the local store.

        Raises:
            TimeFValidationError: If a build lacks a store or the source type is unsupported.
            TimeNetVersionUnavailableError: If the source's version differs from the installed connector's card.
            TimeFFormatError: If the stored release conflicts with a file pin.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by check_pins
        entry = answer.entry if isinstance(answer, Found) else None
        if isinstance(entry, BaseRegistry):
            return entry
        # Nothing but a registry replaces a damaged copy in the store.
        self.check_store_copy(release)
        if isinstance(answer, NotServed):
            raise answer.access_error(release.dataset_id)
        if not isinstance(entry, InstalledConnectors):
            raise TimeFValidationError(
                f"{entry} provides {release}, but a read reads only from a registry and builds only with "
                "InstalledConnectors()"
            )
        if self.store is None:
            raise TimeFValidationError(f"{release} needs a build by {entry}, but the read has no store to build into")
        built, steps = plan_steps(
            release.dataset_id,
            self.store,
            parents_from=self.sources,
            force_build=False,
            open_access_only=False,
            pinned_by=pinned_by,
            lookups=self.lookups,
        )
        if built != release:
            # A source answered for the installed connectors with a version that the card does not declare.
            raise TimeNetVersionUnavailableError(release, built.version)
        for step in steps:
            self.add(step)
        held = self.lookups.manifest(self.store, release)
        if held is not None:
            # The store holds the release, so no build checks the pins on it. Check them now.
            for reader, pinned in (pinned_by or {}).items():
                check_pins(reader, pinned, held)
        return None

    def add(self, step: Step) -> None:
        """Add a step or merge it with an existing step for the same release.

        A copy or build replaces a read and moves to the end of the plan.
        Matching builds keep their position and combine their file pins.

        Raises:
            TimeFFormatError: If two builds have conflicting pins for the same file.
            TimeFValidationError: If the steps cannot be combined.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by _merge_pins
        held = self.steps.get(step.release)
        if held is None:
            # The release has no step yet.
            self.steps[step.release] = step
        elif held == step or (step.action == "read" and held.action != "read"):
            # The same step again, or a read of a release that the plan already copies or builds.
            return
        elif held.action == "read" and step.action != "read":
            # A copy or build replaces the read and moves to the end, after the steps it depends on.
            del self.steps[step.release]
            self.steps[step.release] = step
        elif held.action == step.action == "build" and (held.entry, held.depends_on) == (step.entry, step.depends_on):
            # Two builds of the same release merge their file pins.
            self.steps[step.release] = replace(
                held, pinned_by=_merge_pins(step.release, held.pinned_by, step.pinned_by)
            )
        else:
            # Any other pair of steps conflicts.
            raise TimeFValidationError(
                f"the read needs two different steps for {step.release}; build or copy it into the store "
                "first, then read again"
            )

    def check(self) -> None:
        """Check that local read steps have all files with the recorded sizes.

        Raises:
            TimeNetFileError: If a local file is missing or has an incorrect size.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by open_release
        for step in self.steps.values():
            if step.action == "read" and isinstance(step.entry, LocalRegistry):
                # Opening a local release checks each of its files.
                step.entry.open_release(step.release, manifest=self.manifest(step.entry, step.release))

    def check_store_copy(self, release: DatasetRef) -> None:
        """Check the files of a stored release if a local copy exists.

        Raises:
            TimeNetFileError: If a stored file is missing or has an incorrect size.
        """  # ruff: ignore[docstring-extraneous-exception] - raised from the damage of the copy
        if self.store is None:
            return
        manifest = self.lookups.manifest(self.store, release)
        damage = None if manifest is None else self.store.damage(release, manifest)
        if damage is not None:
            raise damage

    def manifest(self, registry: BaseRegistry, release: DatasetRef) -> Manifest:
        """Read a release manifest and cache it for later lookups.

        Returns:
            The manifest of the requested release.

        Raises:
            TimeNetDatasetNotFoundError: If the release is absent from the registry.
        """
        manifest = self.lookups.manifest(registry, release)
        if manifest is None:
            raise TimeNetDatasetNotFoundError(f"{registry} no longer holds {release}")
        return manifest

    def open(self, release: DatasetRef) -> DatasetVersion:
        """Open a planned release without attaching its payload sources.

        Run all copy and build steps before opening the release.

        Returns:
            A handle from the registry of the release's read step, else from the local store.

        Raises:
            TimeNetDatasetNotFoundError: If the release has no read step and no local store is set.
        """
        step = self.steps.get(release)
        if step is not None and step.action == "read":
            # The plan reads only from registries.
            registry = cast("BaseRegistry", step.entry)
            return registry.open_release(release, manifest=self.manifest(registry, release))
        if self.store is None:
            raise TimeNetDatasetNotFoundError(f"{release} has no source in the plan of the read")
        return self.store.open_release(release)


def _merge_pins(
    release: DatasetRef, first: Mapping[DatasetRef, PayloadSource], second: Mapping[DatasetRef, PayloadSource]
) -> dict[DatasetRef, PayloadSource]:
    """Merge file pins for two builds of the same release and check for conflicts.

    Args:
        release: The release whose files are pinned.
        first: The releases that pin files of the first build, with their file requirements.
        second: The releases that pin files of the second build, with their file requirements.

    Returns:
        The combined pins, with ``second`` taking precedence for releases present in both mappings.

    Raises:
        TimeFFormatError: If two pins disagree on the same file.
    """
    pinned: dict[str, tuple[DatasetRef, FilePart]] = {}
    for pins in (first, second):
        for reader, source in pins.items():
            for part in source.files:
                first_reader, first_part = pinned.setdefault(part.path, (reader, part))
                if first_part != part:
                    raise TimeFFormatError(
                        f"{first_reader} and {reader} pin {part.path!r} of {release} as different files, "
                        f"so no build of {release} can match both"
                    )
    return {**first, **second}
