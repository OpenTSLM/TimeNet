"""Plan and prepare the parent releases a build needs before any connector runs.

Every connector card lists its exact parent releases under ``inputs``. The planner walks those cards
from the requested dataset and decides, for each parent the output registry lacks, whether to fetch
or build it. Cycles and parents that cannot be obtained fail here, before anything downloads.
Parents come before their children in the plan, so running it in order leaves every release a build
opens already in place, and the build itself only opens them.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from timenet.errors import TimeFFormatError, TimeFValidationError, TimeNetDatasetNotFoundError
from timenet.manifest import Manifest
from timenet.registry.base import BaseRegistry
from timenet.registry.dataset_graph_dependency_tools import prepare_dependency_graph
from timenet.registry.local import LocalRegistry
from timenet.registry.shared import complete_copy
from timenet.registry.writable import WritableRegistry
from timenet.types import DatasetMetadata
from timenet.types.references import DatasetRef
from timenet_connectors.discovery import read_card


@dataclass(frozen=True)
class ParentStep:
    """One parent release the output registry lacks, and how to obtain it."""

    reference: DatasetRef
    action: Literal["fetch", "build"]


def plan_parents(
    dataset_id: str,
    output: WritableRegistry,
    *,
    fetch_from: BaseRegistry | None = None,
    build_parents: bool = False,
    force: bool = False,
) -> list[ParentStep]:
    """Plan how to obtain every parent release a build of ``dataset_id`` needs.

    Stored releases need their physical payload sources, not their original build inputs. Missing
    releases are fetched from ``fetch_from`` or built from an installed connector that declares
    exactly that version. Only builds visit card inputs. A release that is already built needs no
    parents unless ``force``.

    Args:
        dataset_id: The dataset to build.
        output: The registry the build writes to.
        fetch_from: Registry to fetch missing parents from. ``None`` disables fetching.
        build_parents: Build missing parents from their installed connectors.
        force: Plan parents even when the requested release is already built.

    Returns:
        The missing parents, each after the parents it needs itself.

    Raises:
        TimeFFormatError: If a committed local parent is incomplete and cannot be fetched.
        TimeFValidationError: If inputs form a cycle, an installed connector declares a different
            version than a parent pins, or fetching targets a remote output registry.
        TimeNetDatasetNotFoundError: If a parent is missing and can be neither fetched nor built.
    """  # noqa: DOC502 (LookupError for an unknown id is raised by read_card)
    if fetch_from is not None and not isinstance(output, LocalRegistry):
        # A remote registry only accepts a child whose parents it already serves, and a fetched
        # parent lands in the local cache, not in the output registry.
        raise TimeFValidationError("fetching parents needs a local output; publish the parents to the registry first")
    card = read_card(dataset_id)
    # A root dataset never touches the output registry here, so its build costs no extra round trip.
    if not card.inputs or (not force and output.exists(card.dataset_id, str(card.dataset_version))):
        return []
    steps: list[ParentStep] = []
    visited: set[str] = set()

    def visit(ref: DatasetRef, path: tuple[str, ...]) -> None:
        key = str(ref)
        if key in path:
            raise TimeFValidationError(f"dataset input cycle: {' -> '.join([*path, key])}")
        if key in visited:
            return
        manifest = _available_manifest(output, ref)
        action: Literal["fetch", "build"] | None = None
        if manifest is None and fetch_from is not None:
            manifest = _available_manifest(fetch_from, ref)
            if manifest is not None:
                action = "fetch"
        if manifest is None:
            if isinstance(output, LocalRegistry) and output.exists(ref.dataset_id, str(ref.version)):
                raise TimeFFormatError(
                    f"committed parent release {ref} is incomplete at {output.root}; "
                    "restore it with --fetch-parents <registry> or rebuild that parent explicitly with --force"
                )
            if not build_parents:
                raise TimeNetDatasetNotFoundError(
                    f"parent release {ref} of {path[-1]} is missing; "
                    "fetch it with --fetch-parents <registry> or build it with --build-parents"
                )
            dependencies = _exact_card(ref).inputs.values()
            action = "build"
        else:
            dependencies = manifest.payload_sources.values()
        for dependency in dependencies:
            visit(dependency, (*path, key))
        if action is not None:
            steps.append(ParentStep(ref, action))
        visited.add(key)

    for ref in card.inputs.values():
        visit(ref, (f"{card.dataset_id}@{card.dataset_version}",))
    return steps


def prepare_parents(  # noqa: PLR0913
    dataset_id: str,
    output: WritableRegistry,
    *,
    build: Callable[[str], object],
    fetch_from: BaseRegistry | None = None,
    build_parents: bool = False,
    force: bool = False,
) -> None:
    """Fetch or build every parent release a build of ``dataset_id`` needs, before it starts.

    Args:
        dataset_id: The dataset to build.
        output: The registry the build writes to.
        build: Builds one parent, by dataset ID, into ``output``.
        fetch_from: Registry to fetch missing parents from. ``None`` disables fetching.
        build_parents: Build missing parents from their installed connectors.
        force: Prepare parents even when the requested release is already built.

    Raises:
        TimeNetDatasetNotFoundError: If a builder does not produce the required release.
    """
    steps = plan_parents(dataset_id, output, fetch_from=fetch_from, build_parents=build_parents, force=force)
    for step in steps:
        if step.action == "fetch":
            prepare_dependency_graph(output.local_root, step.reference, fetch_from=fetch_from)
        else:
            build(step.reference.dataset_id)
            # A successful callback must leave the exact release and its payload sources usable.
            if isinstance(output, LocalRegistry):
                prepare_dependency_graph(output.root, step.reference)
            else:
                if _available_manifest(output, step.reference) is None:
                    raise TimeNetDatasetNotFoundError(f"builder did not publish parent release {step.reference}")
                output.open_version(step.reference.dataset_id, str(step.reference.version))


def _available_manifest(registry: BaseRegistry, reference: DatasetRef) -> Manifest | None:
    """Inspect one release without fetching files or resolving its physical sources.

    Returns:
        The manifest, or None if the release is absent or its local files are incomplete.

    Raises:
        TimeFFormatError: If the manifest identifies a different release.
    """
    try:
        manifest = registry.get_manifest(reference.dataset_id, str(reference.version))
    except TimeNetDatasetNotFoundError:
        return None
    if manifest.dataset_id != reference.dataset_id or manifest.metadata.dataset_version != reference.version:
        raise TimeFFormatError(f"manifest for {reference} declares a different dataset ID or version")
    if isinstance(registry, LocalRegistry):
        directory = registry.root / reference.dataset_id / str(reference.version)
        if not complete_copy(directory, manifest):
            return None
    return manifest


def _exact_card(reference: DatasetRef) -> DatasetMetadata:
    """Return the card of the installed connector that builds exactly ``reference``.

    Returns:
        The connector's card.

    Raises:
        TimeNetDatasetNotFoundError: If no installed connector builds the dataset.
        TimeFValidationError: If the installed connector declares a different version.
    """
    try:
        card = read_card(reference.dataset_id)
    except LookupError as exc:
        raise TimeNetDatasetNotFoundError(f"parent release {reference} is missing and no connector builds it") from exc
    if card.dataset_version != reference.version:
        raise TimeFValidationError(
            f"parent release {reference} is missing and the installed connector builds "
            f"version {card.dataset_version}, not that exact version"
        )
    return card
