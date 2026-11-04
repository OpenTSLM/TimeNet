"""Read a release through an ordered list of sources.

A read opens a release where it is. The release comes from the sources by the rules of
:mod:`timenet.sources`: an exact version comes from the first source that provides it, and "latest"
is the highest version that any source provides. Each payload source of the release resolves on its
own, at its exact version, through the same list. So a release and its payload sources can come
from different registries. If only :class:`~timenet.sources.InstalledConnectors` provides a
release, the read builds it into the local store first, with each release that its build needs.

:func:`plan_read` plans a read and checks the plan before anything runs. :func:`read_release` runs
the copies and the builds of the plan with :func:`~timenet.coordinator.build.run_plan`, and then
opens the release with its payload sources.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import cast

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
    """Plan a read of a release, and check the plan, without running it.

    The plan has one step for each release that the read needs, in run order. A registry that
    provides the release gives a ``"read"`` step: the read opens the release there, in place. If only
    the installed connectors provide it, the plan builds it into ``store``, as
    :func:`~timenet.coordinator.build.plan_release_build` does with ``sources`` as the sources of its
    parents. Each payload source of a release that the read opens in a registry resolves on its own,
    in the same way, and it must match the pins of the release. A release that a step copies or
    builds appears once, and the read opens it from ``store``. If two builds in the plan need one
    release, the plan builds it once, and that build must match the pins of each release that reads
    it. A payload source without a step, which ``store`` already holds, opens from ``store`` too.

    Before any step runs, the plan checks the size of each file of each release that the read opens
    in a local registry, and the pins on each payload source that ``store`` holds. A damaged copy in
    ``store`` fails the read too, unless a registry provides the release, so nothing builds over it.
    The plan and the builds that it plans share each answer and each manifest that they get, so the
    plan asks a source for a release once.

    Args:
        dataset_id: The dataset ID.
        version: An exact version, such as ``"1.2.0"``, or ``None``, ``""`` or ``"latest"`` for the
            latest version.
        sources: The sources, in order: registries and ``InstalledConnectors()``.
        store: The local registry that copies and builds go to, or ``None`` if the read must not
            copy or build anything.

    Returns:
        The plan. If the requested release has a step, that step comes last.

    Raises:
        TimeNetDatasetNotFoundError: If no source provides the release. The message names each
            source.
        TimeNetVersionUnavailableError: If no source provides the exact version, and the installed
            connector of the dataset builds another version. Also if a source answers that the
            installed connectors build a version that the card of the dataset does not declare.
        TimeNetAccessError: If the first source that lists the release, or a payload source, does
            not serve its files, and no later source provides it.
        TimeNetMissingParentError: If no source provides a payload source, or a release that a build
            needs. For the installed connectors, the message names the version that the connector
            builds, if it builds another one.
        TimeNetFileError: If a release that the read opens in a local registry, or a copy in
            ``store`` that no registry replaces, has a missing file or a file of another size.
        TimeFFormatError: If a payload source differs from the files that its reader pins, or if two
            releases that read one release that the plan builds pin one of its files as different
            files.
        TimeFValidationError: If the read needs a build and ``store`` is ``None``, if the plan needs
            two different steps for one release that are not two builds with the same dependencies,
            or if a source answers with an entry that is neither a registry nor the installed
            connectors.
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
    """Open a release with its payload sources, after the copies and the builds that the read needs.

    This function plans with :func:`plan_read`, and runs a plan that copies or builds with
    :func:`~timenet.coordinator.build.run_plan`. A plan that only reads runs nothing, so a read
    without a store works. Then it opens each release from the registry of its ``"read"`` step, with
    the manifest that the plan got, else from ``store``.

    Args:
        dataset_id: The dataset ID.
        version: An exact version, or ``None``, ``""`` or ``"latest"`` for the latest version.
        sources: Registries and ``InstalledConnectors()`` to search, in order.
        store: The local registry for copies and builds, or ``None`` to copy and build nothing.

    Returns:
        The release, with each of its payload sources attached and checked against its pins.

    Raises:
        TimeNetDatasetNotFoundError: If no source provides the release. :func:`plan_read` names this
            error and the other errors of the plan.
        TimeNetVersionConflictError: If a copy finds that ``store`` holds a release with other files.
        TimeFFormatError: If a build does not reproduce the files that other releases pin.
        TimeNetBuildError: If a build runs but fails.
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


@dataclass
class _ReadPlanner:
    """Plan one read: find the source of each release that the read needs, and collect the steps."""

    sources: Sequence[ReleaseSource]
    store: LocalRegistry | None
    lookups: Lookups = field(default_factory=Lookups)
    """The source answers and manifests that the plan got."""
    steps: dict[DatasetRef, Step] = field(default_factory=dict)
    """One step for each release, in run order."""

    def plan(self, dataset_id: str, version: str | None) -> DatasetRef:
        """Add the steps of the requested release, then check the plan.

        Returns:
            The exact release.

        Raises:
            TimeNetDatasetNotFoundError: If no source provides the release.
        """
        version = None if is_latest(version) else version
        answer = self.lookups.resolve(self.sources, dataset_id, version)
        if isinstance(answer, NotFound):
            requested = None if version is None else DatasetRef.parse(f"{dataset_id}@{version}")
            if requested is not None:
                self.check_store_copy(requested)
                connectors = any(isinstance(source, InstalledConnectors) for source in self.sources)
                if connectors and find_card(dataset_id) is not None:
                    # The installed connector builds another version.
                    declared(dataset_id, requested.version)
            names = ", ".join(str(source) for source in self.sources)
            wanted = dataset_id if requested is None else str(requested)
            raise TimeNetDatasetNotFoundError(f"{wanted} is in none of the sources: {names}.")
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
        """Add the steps of one payload source, which resolves through the sources on its own.

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
        """Pick how the read gets a release that a source provides or lists.

        Args:
            release: The exact release.
            answer: The answer of the sources, ``Found`` or ``NotServed``.
            pinned_by: The releases that read files of ``release``, with their pins.

        Returns:
            The registry that serves the release. ``None`` if the read opens the release from the
            store: the plan builds it there, and its steps are in the plan, or the store holds it.

        Raises:
            TimeFValidationError: If the release needs a build and the read has no store, or if a
                source answers with an entry that is neither a registry nor the installed connectors.
            TimeNetVersionUnavailableError: If the installed connectors answer with a version that
                the card of the dataset does not declare.
            TimeFFormatError: If the store holds the release with other files than a reader pins.
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
        """Add a step to the end of the plan, unless the plan has one for its release.

        A copy or a build of a release replaces a read of it, and the read then opens the release
        from the store. The copy or the build moves to the end, after the steps that it depends on.
        Two builds of a release with the same dependencies become one build, which stays where the
        first one is, before each step that needs it. That build must match the pins of each
        reader of both builds.

        Raises:
            TimeFFormatError: If the two builds have readers that pin one file of the release as
                different files.
            TimeFValidationError: If the plan has another step for the release, which is not a read
                that ``step`` replaces, and not a build with the same dependencies.
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
        """Check each file of each release that the read opens in a local registry, by size.

        Raises:
            TimeNetFileError: If a file of such a release is missing or has another size.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by open_release
        for step in self.steps.values():
            if step.action == "read" and isinstance(step.entry, LocalRegistry):
                # Opening a local release checks each of its files.
                step.entry.open_release(step.release, manifest=self.manifest(step.entry, step.release))

    def check_store_copy(self, release: DatasetRef) -> None:
        """Check the copy of a release in the store, if the store holds one.

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
        """Return the manifest of a release that a registry provides, once in this plan.

        Raises:
            TimeNetDatasetNotFoundError: If the registry no longer holds the release.
        """
        manifest = self.lookups.manifest(registry, release)
        if manifest is None:
            raise TimeNetDatasetNotFoundError(f"{registry} no longer holds {release}")
        return manifest

    def open(self, release: DatasetRef) -> DatasetVersion:
        """Open one release of the plan without its payload sources, after the plan ran.

        Returns:
            The release from the registry of its ``"read"`` step, else from the store.

        Raises:
            TimeNetDatasetNotFoundError: If the release has no ``"read"`` step and the read has no store.
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
    """Merge the pins of the readers of two builds of one release, which one build must match.

    Args:
        release: The release that the readers pin.
        first: The readers of the first build, with their pins.
        second: The readers of the second build, with their pins.

    Returns:
        Each reader of both builds, with its pins. A reader of both builds keeps its pins from
        ``second``, which agree with its pins from ``first`` on each file that both pin.

    Raises:
        TimeFFormatError: If two readers, or one reader in both builds, pin one file of the release
            as different files. No build can then match both.
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
