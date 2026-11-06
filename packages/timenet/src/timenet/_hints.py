"""Suggest SDK calls or CLI commands to resolve TimeNet errors."""

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import shlex
from typing import Literal

import httpx

from timenet import builders
from timenet._fs import same_directory
from timenet.config import lies_in, settings
from timenet.errors import (
    TimeNetAccessError,
    TimeNetDatasetNotFoundError,
    TimeNetError,
    TimeNetFileError,
    TimeNetIsolationError,
    TimeNetManifestError,
    TimeNetMissingParentError,
    TimeNetNotBuiltError,
    TimeNetRegistryError,
    TimeNetVersionConflictError,
    TimeNetVersionUnavailableError,
)
from timenet.registry import BaseRegistry, LocalRegistry, WritableRegistry, open_registry
from timenet.registry._paths import release_dir, split_release_dir
from timenet.registry.local import is_same_registry
from timenet.sources import Found, InstalledConnectors, ReleaseSource, resolve_release
from timenet.types.references import DatasetRef, split_ref


Wording = Literal["sdk", "cli"]
"""The hint style: SDK calls or CLI commands."""

_PACKAGE = "a connector package that builds it, such as timenet-connectors"


@dataclass(frozen=True)
class Facts:
    """Details of an operation for choosing an error hint."""

    sources: Sequence[ReleaseSource]
    """The sources of the call in order, with the store."""
    store: LocalRegistry | None
    """The operation's store, or ``None`` if absent or reading from another registry."""
    request: Literal["read", "download", "build", "publish", "clear"] = "read"
    """What the call does."""
    local_registry: str | None = None
    """The ``--local-registry`` value that the command got (CLI only)."""
    remote_registry: str | None = None
    """The ``--remote-registry`` value that the command got (CLI only)."""
    dest: WritableRegistry | None = None
    """The registry that the call publishes to, or ``None``."""
    wanted: str | None = None
    """The requested dataset ID, with an optional ``@version``, or ``None``."""


@contextmanager
def explain(facts: Facts, wording: Wording) -> Generator[None]:
    """Add a hint to a TimeNet error from the block and raise it again.

    An outer block replaces any hint set by an inner block.

    Args:
        facts: Details of the operation.
        wording: The hint style, ``"sdk"`` or ``"cli"``.

    Yields:
        Nothing.

    Raises:
        TimeNetError: The error of the block, with its hint.
    """
    try:
        yield
    except TimeNetError as error:
        error.hint = hint(error, facts, wording)
        raise


def hint(error: TimeNetError, facts: Facts, wording: Wording) -> str | None:
    """Suggest a way to resolve a TimeNet error.

    Return ``None`` if no remedy is known or a lookup fails.

    Args:
        error: The error.
        facts: Details of the operation.
        wording: The hint style, ``"sdk"`` or ``"cli"``.

    Returns:
        One or two sentences, or ``None``.
    """
    try:
        return _remedy(error, facts, wording)
    except Exception:
        return None


def build_command(dataset_id: str, facts: Facts, *, force: bool = False) -> str:
    """Create a ``timenet-build build`` command with the supplied registry flags.

    Args:
        dataset_id: The dataset ID, optionally with ``@version``.
        facts: Details containing the registry flags to include.
        force: Add ``--force``.

    Returns:
        The command line.
    """
    return f"timenet-build build {dataset_id}{_flags(facts)}{' --force' if force else ''}"


def download_command(release: DatasetRef | str, facts: Facts) -> str:
    """Create a ``timenet download`` command with the supplied registry flags.

    Args:
        release: The release, or a dataset ID with an optional ``@version``.
        facts: Details containing the registry flags to include.

    Returns:
        The command line.
    """
    return f"timenet download {release}{_flags(facts)}"


def _remedy(  # ruff: ignore[too-many-return-statements] - one branch per situation
    error: TimeNetError, facts: Facts, wording: Wording
) -> str | None:
    """Return the hint of an error for its situation, or None."""
    if isinstance(error, TimeNetVersionUnavailableError):
        return _version(error, facts, wording)
    if isinstance(error, TimeNetNotBuiltError):
        return _not_built(error.release, facts, wording)
    if isinstance(error, TimeNetMissingParentError):
        return _missing(error.missing.dataset_id, str(error.missing.version), facts, wording)
    if type(error) is TimeNetDatasetNotFoundError and facts.wanted is not None:
        if facts.request == "build":
            # Only an installed connector builds the requested release.
            return None if _connector_packages() else f"Install {_PACKAGE}."
        return _missing(*split_ref(facts.wanted), facts, wording)
    if isinstance(error, TimeNetAccessError):
        return _access(error.dataset, facts, wording)
    if isinstance(error, TimeNetFileError):
        return _damaged(error.release, facts, wording)
    if isinstance(error, TimeNetVersionConflictError):
        return _conflict(error, facts)
    if isinstance(error, TimeNetManifestError):
        return _bad_manifest(error, facts)
    if isinstance(error, TimeNetIsolationError):
        return f"Build in this interpreter with {'--no-isolation' if wording == 'cli' else 'TIMENET_ISOLATION=off'}."
    if isinstance(error, TimeNetRegistryError) and isinstance(error.__cause__, httpx.TransportError):
        return _unreachable(facts, wording)
    return None


def _missing(dataset_id: str, version: str | None, facts: Facts, wording: Wording) -> str | None:
    """Return a hint that names a registry or a build for a release that no source of the call provides."""
    if facts.request == "publish":
        # The release must be in the store first.
        return _into_store(dataset_id, version, facts, wording)
    add = (
        f"Add a registry that holds it to {_sources(facts)}"
        if wording == "sdk"
        else "Pass a registry that holds it with --remote-registry <url>"
    )
    return f"{add}{_or_build(dataset_id, version, facts, wording)}."


def _or_build(dataset_id: str, version: str | None, facts: Facts, wording: Wording) -> str:
    """Return the second half of a sentence that names the build of a release, or an empty string."""
    if facts.request == "download":
        # A download never builds.
        return ""
    if not _connector_packages():
        return f", or install {_PACKAGE}"
    if _lists_connectors(facts) or not _builds(dataset_id, version):
        return ""
    if wording == "cli":
        return ", or build it from its installed connector with --auto-build"
    return f", or add {_connectors(facts)} to build it"


def _not_built(release: DatasetRef, facts: Facts, wording: Wording) -> str:
    """Return a hint that names the build of a release that only a connector provides, or a registry version."""
    build = _build(release.dataset_id, facts, wording)
    if facts.request == "read":
        return f"Read its card with get_card('{release.dataset_id}'), or build it with {build}."
    held = None
    if facts.request == "download" and facts.wanted is not None and split_ref(facts.wanted)[0] == release.dataset_id:
        # A registry can hold another version of the requested release.
        registries = [source for source in facts.sources if isinstance(source, BaseRegistry)]
        answer = resolve_release(registries, release.dataset_id, None)
        held = DatasetRef(dataset_id=release.dataset_id, version=answer.version) if isinstance(answer, Found) else None
    if held is None:
        return f"Build it with {build}."
    download = f"`{download_command(held, facts)}`" if wording == "cli" else f"download('{held}')"
    return f"Build it with {build}, or download the version that a registry holds with {download}."


def _version(error: TimeNetVersionUnavailableError, facts: Facts, wording: Wording) -> str:
    """Return a hint that names the version that the installed connector builds."""
    if facts.request == "download":
        # A download never builds, so the version of the connector needs a build first.
        return _not_built(DatasetRef(dataset_id=error.release.dataset_id, version=error.declared), facts, wording)
    return f"Ask for version {error.declared}, or leave out the version."


def _access(release: DatasetRef, facts: Facts, wording: Wording) -> str:
    """Return a hint that names the local build of a release that no source serves, or a registry for it."""
    dest = facts.dest
    if dest is not None and dest.requires_open_access and facts.request == "publish":
        return f"{dest} serves open releases only. Publish it to a registry of your own, such as a local directory."
    if dest is not None and dest.requires_open_access and wording == "cli":
        return (
            f"{dest} serves open releases only, so --publish-to takes open releases only. Build it into the store "
            "without --publish-to instead."
        )
    build = _build(release.dataset_id, facts, wording)
    card = builders.find_card(release.dataset_id)
    if card is None or card.dataset_version != release.version:
        other = "" if card is None else f"The installed connector builds version {card.dataset_version}. "
        return f"{other}Install the connector package that builds {release}, then build it locally with {build}."
    if facts.request == "download" or _lists_connectors(facts):
        return f"Build it locally with {build}."
    if wording == "cli":
        return "Build it locally from its installed connector with --auto-build."
    return f"Add {_connectors(facts)} to {_sources(facts)} to build it locally."


def _damaged(release: DatasetRef, facts: Facts, wording: Wording) -> str | None:
    """Return a hint that names the copy or the build that replaces a damaged copy in the store, or None."""
    store = facts.store
    held = None if store is None else store.find_manifest(release)
    if store is None or held is None or store.damage(release, held) is None:
        return None
    remedy = _into_store(release.dataset_id, str(release.version), facts, wording, again=True)
    if remedy is not None:
        return remedy
    if wording == "cli":
        local = "" if facts.local_registry is None else f" --local-registry {shlex.quote(facts.local_registry)}"
        command = f"timenet download {release}{local} --remote-registry <url>"
        return f"Copy an intact release into the store from a registry that holds it with `{command}`."
    return f"Add a registry that holds an intact copy to {_sources(facts)}, then call {_download(release, facts)}."


def _into_store(
    dataset_id: str, version: str | None, facts: Facts, wording: Wording, *, again: bool = False
) -> str | None:
    """Return a hint that names the copy from a registry, else the build, that puts a release in the store."""
    store = facts.store
    if store is None:
        return None
    wanted = dataset_id if version is None else f"{dataset_id}@{version}"
    if isinstance(resolve_release(_registries(store, facts), dataset_id, version), Found):
        call = f"`{download_command(wanted, facts)}`" if wording == "cli" else _download(wanted, facts)
        return f"Copy {'an intact release' if again else 'it'} into the store with {call}."
    if not _builds(dataset_id, version):
        return None
    if again:
        return f"Build it again with {_build(dataset_id, facts, wording, force=True)}."
    return f"Build it into the store with {_build(wanted, facts, wording)}."


def _conflict(error: TimeNetVersionConflictError, facts: Facts) -> str:
    """Return a hint that names the delete of a store copy for a consumer, or a new version for a producer."""
    store = facts.store
    if store is not None and error.registry == str(store):
        # The store holds another copy, so a consumer takes the copy of the registry instead.
        return f"Delete {release_dir(store.root, error.release)} to take the copy of {error.source}."
    return "A published version cannot change, so a changed release needs a new version."


def _bad_manifest(error: TimeNetManifestError, facts: Facts) -> str | None:
    """Return a hint that names the delete that lets a download or a build replace a release, or None."""
    store = facts.store
    directory = Path(error.where).parent
    if store is None or not lies_in(directory, store.root):
        return None
    release = split_release_dir(directory)[1]
    # A download or a build reads the manifest first, so it fails until the directory is gone.
    again = f"Delete {directory} and download or build {release} again"
    return f"{again}, then clear again." if facts.request == "clear" else f"{again}."


def _unreachable(facts: Facts, wording: Wording) -> str | None:
    """Return a hint that names the store copy or a registry instead of one that the call cannot reach."""
    if facts.dest is not None:
        return None
    leave = (
        f"leave the registry out of {_sources(facts)}"
        if wording == "sdk"
        else "pass a registry that you can reach with --remote-registry <url>"
    )
    held = None
    if facts.request in {"read", "download"} and facts.store is not None and facts.wanted is not None:
        dataset_id, version = split_ref(facts.wanted)
        # Only the latest version asks every source. The store answers an exact version first.
        answer = facts.store.resolve(dataset_id, None) if version is None else None
        held = DatasetRef(dataset_id=dataset_id, version=answer.version) if isinstance(answer, Found) else None
    if held is None:
        return f"{leave[0].upper()}{leave[1:]}."
    return f"Ask for {held} to read the copy in the store, or {leave}."


def _builds(dataset_id: str, version: str | None) -> bool:
    """Return whether an installed connector builds a release."""
    card = builders.find_card(dataset_id)
    return card is not None and (version is None or str(card.dataset_version) == version)


def _lists_connectors(facts: Facts) -> bool:
    """Return whether the sources of a call list the installed connectors."""
    return any(isinstance(source, InstalledConnectors) for source in facts.sources)


def _connectors(facts: Facts) -> str:
    """Return what an SDK call adds to its sources to build a release."""
    return (
        "InstalledConnectors()"
        if facts.store is not None
        else "LocalRegistry(<path>, store=True) and InstalledConnectors()"
    )


def _connector_packages() -> bool:
    """Return whether any package registers a builder backend."""
    return bool(tuple(builders.entry_points(group=builders.ENTRY_POINT_GROUP)))


def _registries(store: LocalRegistry, facts: Facts) -> list[BaseRegistry]:
    """Return the registries other than the store that the suggested download reads."""
    if facts.request == "publish":
        # A publish has no list of sources, and the download reads the default registry after the store.
        return [open_registry(settings().remote_registry, store=store.root)]
    return [
        source for source in facts.sources if isinstance(source, BaseRegistry) and not is_same_registry(source, store)
    ]


def _build(dataset_id: str, facts: Facts, wording: Wording, *, force: bool = False) -> str:
    """Return the call or the command that builds a dataset into the store of the call."""
    if wording == "cli":
        return f"`{build_command(dataset_id, facts, force=force)}`"
    store = facts.store
    keywords = ""
    if store is not None and not same_directory(store.root, settings().local_registry_dir):
        keywords += f", store={str(store.root)!r}"
    if force:
        keywords += ", force=True"
    return f"timenet.build('{dataset_id}'{keywords})"


def _download(release: DatasetRef | str, facts: Facts) -> str:
    """Return the SDK call that copies a release into the store."""
    client = "" if facts.request in {"read", "download"} else "timenet.TimeNet()."
    return f"{client}download('{release}')"


def _sources(facts: Facts) -> str:
    """Return the name of the sources of an SDK call."""
    return "parents_from" if facts.request == "build" else "the sources"


def _flags(facts: Facts) -> str:
    """Return the registry flags that a CLI command got, for a command that it suggests."""
    flags = (("--local-registry", facts.local_registry), ("--remote-registry", facts.remote_registry))
    return "".join(f" {flag} {shlex.quote(value)}" for flag, value in flags if value is not None)
