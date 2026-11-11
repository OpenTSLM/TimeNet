"""Suggest SDK calls or CLI commands to resolve TimeNet errors."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum, auto
import shlex
from typing import Any, TypeAlias

from typing_extensions import override

from timenet import builders
from timenet._fs import same_directory
from timenet.config import settings
from timenet.errors import (
    TimeFFormatError,
    TimeFFormatVersionError,
    TimeNetAccessError,
    TimeNetDatasetNotFoundError,
    TimeNetError,
    TimeNetFileError,
    TimeNetInvalidCardError,
    TimeNetIsolationError,
    TimeNetManifestError,
    TimeNetMissingParentError,
    TimeNetNoCatalogError,
    TimeNetNotBuiltError,
    TimeNetOriginError,
    TimeNetRegistryError,
    TimeNetUnreachableError,
    TimeNetVersionConflictError,
    TimeNetVersionUnavailableError,
)
from timenet.registry import BaseRegistry, LocalRegistry, WritableRegistry
from timenet.registry._paths import release_dir
from timenet.registry.local import is_same_registry
from timenet.sources import Found, InstalledConnectors, ReleaseSource, resolve_release
from timenet.types.references import LATEST_VERSION, DatasetRef, split_ref


_PACKAGE = "a connector package that builds it, such as timenet-connectors"


class Operation(Enum):
    """What a call does."""

    READ = auto()
    DOWNLOAD = auto()
    BUILD = auto()
    PUBLISH = auto()
    CLEAR = auto()


@dataclass(frozen=True)
class RequestedRelease:
    """A requested dataset ID, with an optional version."""

    dataset_id: str
    """The dataset ID."""
    version: str | None = None
    """The requested version or ``"latest"``, or ``None`` without a version."""

    @classmethod
    def parse(cls, value: "str | DatasetRef") -> "RequestedRelease":
        """Read a dataset ID with an optional ``@version``, or an exact release.

        Args:
            value: The dataset ID with an optional ``@version``, or the release.

        Returns:
            The requested release.

        Raises:
            TimeFValidationError: If ``value`` is not a valid reference.
        """  # ruff: ignore[docstring-extraneous-exception] (raised by split_ref)
        if isinstance(value, DatasetRef):
            return cls(value.dataset_id, str(value.version))
        return cls(*split_ref(value))

    def __str__(self) -> str:
        return self.dataset_id if self.version is None else f"{self.dataset_id}@{self.version}"


@dataclass(frozen=True)
class Facts:
    """Details of an operation for choosing an error hint."""

    sources: Sequence[ReleaseSource]
    """The sources of the call in order, with the local registry."""
    local_registry: LocalRegistry | None
    """The local registry that the call uses, or ``None`` if it uses none."""
    request: Operation = Operation.READ
    """What the call does."""
    dest: WritableRegistry | None = None
    """The registry that the call publishes to, or ``None``."""
    wanted: RequestedRelease | None = None
    """The requested release, or ``None``."""


class HintWording(ABC):
    """The parts of a hint that name SDK calls or CLI commands."""

    @abstractmethod
    def build(self, dataset_id: str, facts: Facts, *, force: bool = False) -> str:
        """Return the call or the command that builds a dataset into the local registry of the call.

        Args:
            dataset_id: The dataset ID, optionally with ``@version``.
            facts: Details of the operation.
            force: Build the release again.

        Returns:
            The call or the command.
        """

    @abstractmethod
    def download(self, release: DatasetRef | str, facts: Facts, *, force: bool = False) -> str:
        """Return the call or the command that copies a release into the local registry.

        Args:
            release: The release, or a dataset ID with an optional ``@version``.
            facts: Details of the operation.
            force: Copy the release again.

        Returns:
            The call or the command.
        """

    @abstractmethod
    def add_registry(self) -> str:
        """Return the start of a sentence that adds a registry that holds a release."""

    @abstractmethod
    def or_build_from_connectors(self, facts: Facts) -> str:
        """Return the end of a sentence that builds a release from its installed connector.

        Args:
            facts: Details of the operation.

        Returns:
            The end of the sentence, from its comma.
        """

    @abstractmethod
    def build_locally_from_connectors(self, facts: Facts) -> str:
        """Return the sentence that builds a release locally from its installed connector.

        Args:
            facts: Details of the operation.

        Returns:
            The sentence.
        """

    @abstractmethod
    def copy_intact(self, release: DatasetRef, facts: Facts) -> str:
        """Return the sentence that copies an intact release from a registry that the call does not have.

        Args:
            release: The release.
            facts: Details of the operation.

        Returns:
            The sentence.
        """

    @abstractmethod
    def read_exact_release(self) -> str:
        """Return the call or the command that reads a release by its exact ID and version."""

    @abstractmethod
    def leave_unreachable(self) -> str:
        """Return the part of a sentence that replaces a registry that the call cannot reach."""

    @abstractmethod
    def run_in_this_interpreter(self) -> str:
        """Return the flag or the variable that builds in this interpreter."""

    @abstractmethod
    def publish_to_open_only(self, dest: WritableRegistry) -> str | None:
        """Return the hint for a publish target that serves open releases only, or ``None``.

        Args:
            dest: The registry that the call publishes to.

        Returns:
            The hint, or ``None`` if the front end has no such target.
        """


class SdkWording(HintWording):
    """The parts of a hint for the SDK: calls and keywords."""

    @override
    def build(self, dataset_id: str, facts: Facts, *, force: bool = False) -> str:
        local_registry = facts.local_registry
        keywords = ""
        if local_registry is not None and not same_directory(local_registry.root, settings().local_registry_dir):
            keywords += f", local_registry={str(local_registry.root)!r}"
        if force:
            keywords += ", force=True"
        return f"timenet.build('{dataset_id}'{keywords})"

    @override
    def download(self, release: DatasetRef | str, facts: Facts, *, force: bool = False) -> str:
        client = "" if facts.request in {Operation.READ, Operation.DOWNLOAD} else "timenet.TimeNet()."
        return f"{client}download('{release}'{', force=True' if force else ''})"

    @override
    def add_registry(self) -> str:
        return "Add a registry that holds it to the sources"

    @override
    def or_build_from_connectors(self, facts: Facts) -> str:
        return f", or add {_connectors(facts)} to build it"

    @override
    def build_locally_from_connectors(self, facts: Facts) -> str:
        return f"Add {_connectors(facts)} to the sources to build it locally."

    @override
    def copy_intact(self, release: DatasetRef, facts: Facts) -> str:
        return f"Add a registry that holds an intact copy to the sources, then call {self.download(release, facts)}."

    @override
    def read_exact_release(self) -> str:
        return "open('<id>@<version>')"

    @override
    def leave_unreachable(self) -> str:
        return "leave the unreachable registry out of the sources"

    @override
    def run_in_this_interpreter(self) -> str:
        return "TIMENET_ISOLATION=off"

    @override
    def publish_to_open_only(self, dest: WritableRegistry) -> str | None:
        # The SDK names the target of a publish only in publish(), which has a hint of its own.
        return None


@dataclass(frozen=True)
class CliWording(HintWording):
    """The parts of a hint for a CLI: commands and flags, with the registry flags that the command got."""

    local_registry: str | None = None
    """The ``--local-registry`` value that the command got, or ``None``."""
    remote_registry: str | None = None
    """The ``--remote-registry`` value that the command got, or ``None``."""

    def build_command(self, dataset_id: str, *, force: bool = False) -> str:
        """Create a ``timenet-build build`` command with the registry flags of the command.

        Args:
            dataset_id: The dataset ID, optionally with ``@version``.
            force: Add ``--force``.

        Returns:
            The command line.
        """
        return f"timenet-build build {dataset_id}{self._flags()}{' --force' if force else ''}"

    def download_command(
        self, release: DatasetRef | str, *, force: bool = False, remote_registry: str | None = None
    ) -> str:
        """Create a ``timenet download`` command with the registry flags of the command.

        Args:
            release: The release, or a dataset ID with an optional ``@version``.
            force: Add ``--force``.
            remote_registry: A placeholder such as ``<url>`` for ``--remote-registry``, written as it is, or
                ``None`` for the flag of the command.

        Returns:
            The command line.
        """
        return f"timenet download {release}{self._flags(remote_registry)}{' --force' if force else ''}"

    @override
    def build(self, dataset_id: str, facts: Facts, *, force: bool = False) -> str:
        return f"`{self.build_command(dataset_id, force=force)}`"

    @override
    def download(self, release: DatasetRef | str, facts: Facts, *, force: bool = False) -> str:
        return f"`{self.download_command(release, force=force)}`"

    @override
    def add_registry(self) -> str:
        return "Pass a registry that holds it with --remote-registry <url>"

    @override
    def or_build_from_connectors(self, facts: Facts) -> str:
        return ", or build it from its installed connector with --auto-build"

    @override
    def build_locally_from_connectors(self, facts: Facts) -> str:
        return "Build it locally from its installed connector with --auto-build."

    @override
    def copy_intact(self, release: DatasetRef, facts: Facts) -> str:
        command = self.download_command(release, remote_registry="<url>")
        return f"Copy an intact release into the local registry from a registry that holds it with `{command}`."

    @override
    def read_exact_release(self) -> str:
        return f"`timenet info <id>@<version>{self._flags()}`"

    @override
    def leave_unreachable(self) -> str:
        return "pass a registry that you can reach with --remote-registry <url>"

    @override
    def run_in_this_interpreter(self) -> str:
        return "--no-isolation"

    @override
    def publish_to_open_only(self, dest: WritableRegistry) -> str | None:
        return (
            f"{dest} serves open releases only, so --publish-to takes open releases only. Build it into the local "
            "registry without --publish-to instead."
        )

    def _flags(self, remote_registry: str | None = None) -> str:
        """Return the registry flags of the command, for a command that it suggests."""
        remote = remote_registry or (None if self.remote_registry is None else shlex.quote(self.remote_registry))
        local = None if self.local_registry is None else shlex.quote(self.local_registry)
        flags = (("--local-registry", local), ("--remote-registry", remote))
        return "".join(f" {flag} {value}" for flag, value in flags if value is not None)


_CURRENT_HINT_WORDING: ContextVar[HintWording] = ContextVar(
    "current_hint_wording",
    default=SdkWording(),  # ruff: ignore[mutable-contextvar-default] (SdkWording holds no state)
)
"""The wording of the hints that :func:`explain` adds: SDK calls, unless a CLI command set its own."""


@contextmanager
def use_wording(wording: HintWording) -> Generator[None]:
    """Give the hints of the block the given wording.

    Args:
        wording: The calls or the commands that the hints name.

    Yields:
        Nothing.
    """
    token = _CURRENT_HINT_WORDING.set(wording)
    try:
        yield
    finally:
        _CURRENT_HINT_WORDING.reset(token)


@contextmanager
def explain(facts: Facts) -> Generator[None]:
    """Add a hint in the current wording to a TimeNet error from the block and raise it again.

    An outer block replaces any hint set by an inner block.

    Args:
        facts: Details of the operation.

    Yields:
        Nothing.

    Raises:
        TimeNetError: The error of the block, with its hint.
    """
    try:
        yield
    except TimeNetError as error:
        error.hint = hint(error, facts, _CURRENT_HINT_WORDING.get())
        raise


def hint(error: TimeNetError, facts: Facts, wording: HintWording) -> str | None:
    """Suggest a way to resolve a TimeNet error.

    Return ``None`` if no remedy is known or a lookup fails with a registry, format or card error.

    Args:
        error: The error.
        facts: Details of the operation.
        wording: The calls or the commands that the hint names.

    Returns:
        One or two sentences, or ``None``.
    """
    try:
        return _remedy(error, facts, wording)
    except (TimeNetRegistryError, TimeFFormatError, TimeNetInvalidCardError):
        return None


Remedy: TypeAlias = Callable[[Any, Facts, HintWording], str | None]
"""A function that returns the hint of an error of its class, or None."""


def _remedy(error: TimeNetError, facts: Facts, wording: HintWording) -> str | None:
    """Return the hint of the most specific class of the error that has a remedy, or None."""
    for cls in type(error).__mro__:
        remedy = _REMEDIES.get(cls)
        if remedy is not None:
            return remedy(error, facts, wording)
    return None


def _not_found(facts: Facts, wording: HintWording) -> str | None:
    """Return a hint that names a registry, a build or a connector package for the requested release, or None."""
    if facts.wanted is None:
        return None
    if facts.request is Operation.BUILD:
        # Only an installed connector builds the requested release.
        return None if builders.has_backends() else f"Install {_PACKAGE}."
    return _missing(facts.wanted.dataset_id, facts.wanted.version, facts, wording)


def _missing(dataset_id: str, version: str | None, facts: Facts, wording: HintWording) -> str | None:
    """Return a hint that names a registry or a build for a release that no source of the call provides."""
    if facts.request is Operation.PUBLISH:
        # The release must be in the local registry first.
        return _into_local_registry(dataset_id, version, facts, wording)
    return f"{wording.add_registry()}{_or_build(dataset_id, version, facts, wording)}."


def _or_build(dataset_id: str, version: str | None, facts: Facts, wording: HintWording) -> str:
    """Return the second half of a sentence that names the build of a release, or an empty string."""
    if facts.request is Operation.DOWNLOAD:
        # A download never builds.
        return ""
    if not builders.has_backends():
        return f", or install {_PACKAGE}"
    if _lists_connectors(facts) or not _builds(dataset_id, version):
        return ""
    return wording.or_build_from_connectors(facts)


def _not_built(release: DatasetRef, facts: Facts, wording: HintWording) -> str:
    """Return a hint that names the build of a release that only a connector provides, or a registry version."""
    build = wording.build(release.dataset_id, facts)
    if facts.request is Operation.READ:
        return f"Read its card with get_card('{release.dataset_id}'), or build it with {build}."
    held = None
    if (
        facts.request is Operation.DOWNLOAD
        and facts.wanted is not None
        and facts.wanted.dataset_id == release.dataset_id
    ):
        # A registry can hold another version of the requested release.
        registries = [source for source in facts.sources if isinstance(source, BaseRegistry)]
        answer = resolve_release(registries, release.dataset_id, None)
        held = DatasetRef(dataset_id=release.dataset_id, version=answer.version) if isinstance(answer, Found) else None
    if held is None:
        return f"Build it with {build}."
    download = wording.download(held, facts)
    return f"Build it with {build}, or download the version that a registry holds with {download}."


def _version(error: TimeNetVersionUnavailableError, facts: Facts, wording: HintWording) -> str:
    """Return a hint that names the version that the installed connector builds."""
    if facts.request is Operation.DOWNLOAD:
        # A download never builds, so the version of the connector needs a build first.
        return _not_built(DatasetRef(dataset_id=error.release.dataset_id, version=error.declared), facts, wording)
    return f"Ask for version {error.declared}, or leave out the version."


def _access(release: DatasetRef, facts: Facts, wording: HintWording) -> str:
    """Return a hint that names the local build of a release that no source serves, or a registry for it."""
    dest = facts.dest
    if dest is not None and dest.requires_open_access:
        if facts.request is Operation.PUBLISH:
            return f"{dest} serves open releases only. Publish it to a registry of your own, such as a local directory."
        refused = wording.publish_to_open_only(dest)
        if refused is not None:
            return refused
    build = wording.build(release.dataset_id, facts)
    card = builders.find_card(release.dataset_id)
    if card is None or card.dataset_version != release.version:
        other = "" if card is None else f"The installed connector builds version {card.dataset_version}. "
        return f"{other}Install the connector package that builds {release}, then build it locally with {build}."
    if facts.request is Operation.DOWNLOAD or _lists_connectors(facts):
        return f"Build it locally with {build}."
    return wording.build_locally_from_connectors(facts)


def _damaged(release: DatasetRef, facts: Facts, wording: HintWording) -> str | None:
    """Return a hint that names the copy or the build that replaces a damaged copy in the local registry, or None."""
    local_registry = facts.local_registry
    held = None if local_registry is None else local_registry.find_manifest(release)
    if local_registry is None or held is None or local_registry.damage(release, held) is None:
        return None
    remedy = _into_local_registry(release.dataset_id, str(release.version), facts, wording, again=True)
    if remedy is not None:
        return remedy
    return wording.copy_intact(release, facts)


def _into_local_registry(
    dataset_id: str, version: str | None, facts: Facts, wording: HintWording, *, again: bool = False
) -> str | None:
    """Return a hint that names the copy from a registry, else the build, that puts a release in the local registry."""
    local_registry = facts.local_registry
    if local_registry is None:
        return None
    wanted = str(RequestedRelease(dataset_id, version))
    download = (
        f"Copy {'an intact release' if again else 'it'} into the local registry with {wording.download(wanted, facts)}."
    )
    # A publish asks no registry, so its hint names only what the local facts show.
    publish = facts.request is Operation.PUBLISH
    if not publish and isinstance(resolve_release(_registries(local_registry, facts), dataset_id, version), Found):
        return download
    if _builds(dataset_id, version):
        if again:
            return f"Build it again with {wording.build(dataset_id, facts, force=True)}."
        return f"Build it into the local registry with {wording.build(wanted, facts)}."
    # Without a connector, a publish names the download, but no registry was asked if it holds the release.
    return download if publish and not again else None


def _conflict(error: TimeNetVersionConflictError, facts: Facts) -> str:
    """Return a hint that names the delete of the local copy for a consumer, or a new version for a producer."""
    local_registry = facts.local_registry
    if local_registry is not None and error.registry_location == local_registry.location:
        # The local registry holds another copy, so a consumer takes the copy of the registry instead.
        return f"Delete {release_dir(local_registry.root, error.release)} to take the copy of {error.source}."
    if facts.request is Operation.BUILD and facts.dest is not None:
        # Only timenet-build build has a destination for a build: --publish-to.
        return (
            "Build it again with `--force` to compare a fresh build with the published release, or give a changed "
            "release a new version."
        )
    return "A published version cannot change, so a changed release needs a new version."


def _bad_manifest(error: TimeNetManifestError, facts: Facts) -> str | None:
    """Return a hint that names the delete that lets a download or a build replace a release, or None."""
    local_registry, path = facts.local_registry, error.path
    release = None if local_registry is None or path is None else local_registry.release_at(path)
    if path is None or release is None:
        return None
    directory = path.parent
    # A download or a build reads the manifest first, so it fails until the directory is gone.
    again = f"Delete {directory} and download or build {release} again"
    return f"{again}, then clear again." if facts.request is Operation.CLEAR else f"{again}."


def _format_version(error: TimeFFormatVersionError) -> str | None:
    """Return a hint that names the upgrade that reads a newer format, or None for an older one."""
    return "Upgrade the timenet package to read it." if error.found > error.supported else None


def _bad_origin(release: DatasetRef, facts: Facts, wording: HintWording) -> str:
    """Return a hint that names the forced download and the forced build that replace a release and its record."""
    download = wording.download(release, facts, force=True)
    build = wording.build(str(release), facts, force=True)
    return (
        f"Get the release again with {download} if it came from a registry, or with {build} if it was built. "
        "Both replace the release and its record."
    )


def _unreachable(facts: Facts, wording: HintWording) -> str | None:
    """Return a hint that names the local copy or a registry instead of one that the call cannot reach."""
    if facts.dest is not None:
        return None
    leave = wording.leave_unreachable()
    held = None
    wanted = facts.wanted
    local_registry = facts.local_registry
    if facts.request in {Operation.READ, Operation.DOWNLOAD} and local_registry is not None and wanted is not None:
        # Only "latest" asks every source. The local registry answers an exact version first.
        answer = local_registry.resolve(wanted.dataset_id, None) if wanted.version == LATEST_VERSION else None
        held = DatasetRef(dataset_id=wanted.dataset_id, version=answer.version) if isinstance(answer, Found) else None
    if held is None:
        return f"{leave[0].upper()}{leave[1:]}."
    return f"Ask for {held} to read the copy in the local registry, or {leave}."


def _builds(dataset_id: str, version: str | None) -> bool:
    """Return whether an installed connector builds a release."""
    return isinstance(resolve_release([InstalledConnectors()], dataset_id, version), Found)


def _lists_connectors(facts: Facts) -> bool:
    """Return whether the sources of a call list the installed connectors."""
    return any(isinstance(source, InstalledConnectors) for source in facts.sources)


def _connectors(facts: Facts) -> str:
    """Return what an SDK call adds to its sources to build a release."""
    if facts.local_registry is not None:
        return "InstalledConnectors()"
    return "LocalRegistry(<path>) and InstalledConnectors()"


def _registries(local_registry: LocalRegistry, facts: Facts) -> list[BaseRegistry]:
    """Return the registries of the call other than the local registry, which the suggested download reads."""
    return [
        source
        for source in facts.sources
        if isinstance(source, BaseRegistry) and not is_same_registry(source, local_registry)
    ]


_REMEDIES: dict[type[TimeNetError], Remedy] = {
    TimeNetVersionUnavailableError: _version,
    TimeNetNotBuiltError: lambda error, facts, wording: _not_built(error.release, facts, wording),
    TimeNetMissingParentError: lambda error, facts, wording: _missing(
        error.missing.dataset_id, str(error.missing.version), facts, wording
    ),
    TimeNetDatasetNotFoundError: lambda _, facts, wording: _not_found(facts, wording),
    TimeNetAccessError: lambda error, facts, wording: _access(error.dataset, facts, wording),
    TimeNetFileError: lambda error, facts, wording: _damaged(error.release, facts, wording),
    TimeNetVersionConflictError: lambda error, facts, _: _conflict(error, facts),
    TimeNetManifestError: lambda error, facts, _: _bad_manifest(error, facts),
    TimeFFormatVersionError: lambda error, _, __: _format_version(error),
    TimeNetOriginError: lambda error, facts, wording: _bad_origin(error.release, facts, wording),
    TimeNetIsolationError: lambda _, __, wording: (
        f"Build in this interpreter with {wording.run_in_this_interpreter()}."
    ),
    TimeNetUnreachableError: lambda _, facts, wording: _unreachable(facts, wording),
    TimeNetNoCatalogError: lambda _, __, wording: f"Read an exact release with {wording.read_exact_release()}.",
}
"""The remedy of each error class. A subclass without its own entry gets the remedy of its nearest base class."""
