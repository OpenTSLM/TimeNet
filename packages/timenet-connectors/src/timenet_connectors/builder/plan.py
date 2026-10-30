"""Plan and prepare the parent releases a build needs before any connector runs.

Every connector card lists its exact parent releases under ``parents``. The planner walks those cards
from the requested dataset and decides, for each parent the output registry lacks, whether to fetch
or build it. Cycles and parents that cannot be obtained fail here, before anything downloads.
Parents come before their children in the plan, so running it in order leaves every release a build
opens already in place, and the build itself only opens them.

A payload source is pinned: the release that reads its files records each of them by path, size, and
checksum. A build of a missing payload source must reproduce those files. The plan hands the pins to
the build, which publishes the release only if the files match, and otherwise discards it and raises.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
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
    """One parent release that the output registry lacks, or lacks payload sources of, and how to obtain it.

    A fetch brings the release with each direct payload source the output lacks.
    """

    reference: DatasetRef
    action: Literal["fetch", "build"]
    pinned_by: Mapping[DatasetRef, PayloadSource] = field(default_factory=dict, hash=False)
    """For a build: each release that already reads files of this one, with its pins on them."""


def plan_parents(
    dataset_id: str,
    output: WritableRegistry,
    *,
    fetch_from: BaseRegistry | None = None,
    build_parents: bool = False,
    force_build: bool = False,
) -> list[ParentStep]:
    """Plan how to get every parent release that a build of ``dataset_id`` needs.

    For each declared parent, a build needs the parent and the payload sources of that parent. It
    does not need the payload sources of those sources. The plan visits the declared parents of a
    release only if that release gets built. So if the output already holds the requested release,
    the plan covers its parents only when ``force_build`` is set. Without ``force_build``, the plan
    covers only the payload sources of that release.

    A missing release is fetched from ``fetch_from``, or built from an installed connector that
    declares exactly that version. A damaged copy in a local output is fetched again, and never
    built over. An output that is not local accepts only releases with ``open`` access, which need
    no credentials. So every release that the plan visits for such an output must have ``open``
    access. A registry that is not local serves only releases with ``open`` access, so a release to
    fetch from one must have ``open`` access too.

    Args:
        dataset_id: The dataset to build.
        output: The registry the build writes to.
        fetch_from: Registry to fetch missing parents from. ``None`` disables fetching.
        build_parents: Build missing parents from their installed connectors.
        force_build: Build the requested release again even if the output holds it, and plan its parents.

    Returns:
        The parents to fetch or build, each after the parents it needs itself.

    Raises:
        LookupError: If no installed connector declares a card for ``dataset_id``.
        TimeFValidationError: If parents form a cycle, an installed connector declares a different
            version than a parent pins, or fetching targets a remote output registry.
        TimeNetMissingParentError: If a parent or payload source is missing and building is off.
        TimeNetDatasetNotFoundError: If a parent is missing and no installed connector builds it.
        TimeNetFileError: If the output holds a parent with a damaged copy that cannot be fetched again.
        TimeNetAccessError: If the plan visits a release without ``open`` access in an output that is
            not local, or would fetch one from a registry that is not local.
    """  # noqa: DOC502 - raised by the helpers of the walk
    if fetch_from is not None and not isinstance(output, LocalRegistry):
        # A remote registry only accepts a child whose parents it already serves, and a fetched
        # parent lands in the local cache, not in the output registry.
        raise TimeFValidationError("fetching parents needs a local output; publish the parents to the registry first")
    card = find_card(dataset_id)
    if card is None:
        raise LookupError(f"no installed connector declares a card for {dataset_id!r}")
    # A root dataset never touches the output registry here, so its build costs no extra round trip.
    if not card.parents:
        return []
    root = DatasetRef(dataset_id=card.dataset_id, version=card.dataset_version)
    # A release that is already built needs only its payload sources, to open. Its declared parents
    # matter only to a rebuild.
    held = None if force_build else _manifest(output, root)
    if held is None and not isinstance(output, LocalRegistry):
        require_open_access(card)
    searched = str(output) if fetch_from is None else f"{output} or {fetch_from}"
    steps: list[tuple[DatasetRef, Literal["fetch", "build"]]] = []
    # Whether each visited release was planned with its payload sources.
    planned: dict[str, bool] = {}
    # The payload sources that a fetch brings with the release that reads them.
    fetched: set[str] = set()
    # The releases that read files of each visited payload source, with their pins.
    pins: dict[str, dict[DatasetRef, PayloadSource]] = {}

    def visit(ref: DatasetRef, path: tuple[DatasetRef, ...], pinned: PayloadSource | None = None) -> None:
        key = str(ref)
        with_sources = pinned is None
        if pinned is not None:
            pins.setdefault(key, {})[path[-1]] = pinned
        if ref in path:
            raise TimeFValidationError(f"dataset parent cycle: {' -> '.join(str(item) for item in (*path, ref))}")
        if key in planned and (planned[key] or not with_sources):
            return
        planned[key] = with_sources
        manifest, action = _locate(output, fetch_from, ref)
        if manifest is None:
            built = _card_to_build(
                ref, path[-1], output, build_parents=build_parents, pinned=not with_sources, searched=searched
            )
            # A build opens each of its parents with that parent's payload sources.
            for dependency in built.parents.values():
                visit(dependency, (*path, ref))
            action = "build"
        elif with_sources:
            # Opening a release needs its direct payload sources, which list every release whose
            # files it reads, but not their own sources.
            sources = list(manifest.payload_sources.values())
            for source in sources:
                visit(source.release, (*path, ref), source)
            # Fetching a release fetches the payload sources it lacks, so a release whose source
            # must be fetched is fetched again, even when the output holds it.
            if action is None and any(str(source.release) in fetched for source in sources):
                action = "fetch"
        elif action == "fetch":
            fetched.add(key)
            return
        # A payload source that a later visit declares as a parent is built once.
        if action is not None and (ref, action) not in steps:
            steps.append((ref, action))

    if held is not None:
        sources = list(held.payload_sources.values())
        for source in sources:
            visit(source.release, (root,), source)
        # As for a parent: a fetch of the built release brings each payload source it lacks.
        if any(str(source.release) in fetched for source in sources):
            steps.append((root, "fetch"))
    else:
        for ref in card.parents.values():
            visit(ref, (root,))
    return [
        ParentStep(ref, action, pinned_by=pins.get(str(ref), {}) if action == "build" else {}) for ref, action in steps
    ]


def prepare_parents(  # noqa: PLR0913
    dataset_id: str,
    output: WritableRegistry,
    *,
    build: Callable[[ParentStep], object],
    fetch_from: BaseRegistry | None = None,
    build_parents: bool = False,
    force_build: bool = False,
) -> None:
    """Fetch or build every parent release a build of ``dataset_id`` needs, before it starts.

    Args:
        dataset_id: The dataset to build.
        output: The registry the build writes to.
        build: Builds one parent into ``output``. A build step with ``pinned_by`` must publish the
            release only if it reproduces those pins.
        fetch_from: Registry to fetch missing parents from. ``None`` disables fetching.
        build_parents: Build missing parents from their installed connectors.
        force_build: Build the requested release again even if the output holds it, and plan its parents.

    Raises:
        TimeNetDatasetNotFoundError: If a builder does not produce the required release.
    """
    steps = plan_parents(
        dataset_id, output, fetch_from=fetch_from, build_parents=build_parents, force_build=force_build
    )
    for step in steps:
        if step.action == "fetch":
            prepare_local_copy(output.local_root, step.reference, fetch_from=fetch_from)
        else:
            build(step)
            # A successful callback must leave the exact release and its payload sources usable.
            if isinstance(output, LocalRegistry):
                prepare_local_copy(output.root, step.reference)
            else:
                if _manifest(output, step.reference) is None:
                    raise TimeNetDatasetNotFoundError(f"builder did not publish parent release {step.reference}")
                output.open_version(step.reference.dataset_id, str(step.reference.version))


def _manifest(registry: BaseRegistry, reference: DatasetRef) -> Manifest | None:
    """Read one release's manifest without fetching files or resolving its payload sources.

    Returns:
        The manifest, or None if the registry does not hold the release.

    Raises:
        TimeFFormatError: If the manifest identifies a different release.
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
    """Find a release in the output, or else in the registry to fetch from.

    Returns:
        The manifest and ``None`` for a release the output holds usable, the manifest and
        ``"fetch"`` for one to fetch, or ``None`` twice for one to build.

    Raises:
        TimeNetFileError: If the output holds a local copy that lacks a file or holds one with another
            size, and the release cannot be fetched again.
        TimeNetAccessError: If the release is not open, and the output that holds it, or the
            registry to fetch it from, is not local.
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
        # A registry other than a local one serves open releases only, so check before any download.
        if not isinstance(fetch_from, LocalRegistry):
            require_open_access(fetched.metadata, fetched.lineage)
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
    """Return the card that builds a release the plan can neither find nor fetch.

    Returns:
        The card of the installed connector that builds exactly ``reference``.

    Raises:
        TimeNetMissingParentError: If building parents is off.
        TimeNetAccessError: If the output is not local and the card is not open.
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
        The connector's card.

    Raises:
        TimeNetDatasetNotFoundError: If no installed connector builds the dataset.
        TimeFValidationError: If the installed connector declares a different version.
    """
    card = find_card(reference.dataset_id)
    if card is None:
        raise TimeNetDatasetNotFoundError(f"parent release {reference} is missing and no installed connector builds it")
    if card.dataset_version != reference.version:
        raise TimeFValidationError(
            f"parent release {reference} is missing and the installed connector builds "
            f"version {card.dataset_version}, not that exact version"
        )
    return card
