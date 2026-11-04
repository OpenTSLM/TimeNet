"""Find, read, and download datasets with :class:`TimeNet`.

Clients search their sources in order. Builds and downloads go into the first local registry that is
not read-only.
"""

# The public API has a method named ``list``. Deferred annotations keep the type hint
# ``list[str]`` resolved to the builtin type, not to the method.
from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
import shlex
from typing import TYPE_CHECKING, TypeAlias, TypeVar

from timenet.builders import BuildOptions
from timenet.config import settings
from timenet.coordinator.build import run_plan
from timenet.coordinator.plan import Lookups, plan_download
from timenet.coordinator.read import raise_not_found, read_release
from timenet.dataset import TimeFDataset
from timenet.errors import (
    TimeFValidationError,
    TimeNetAccessError,
    TimeNetDatasetNotFoundError,
    TimeNetError,
    TimeNetFileError,
    TimeNetMissingParentError,
    TimeNetVersionUnavailableError,
)
from timenet.manifest import Manifest
from timenet.progress import ProgressCallback
from timenet.reader import LoadPlan, TimeFReader
from timenet.registry import BaseRegistry, LocalRegistry, open_registry
from timenet.registry._paths import release_dir
from timenet.registry.local import is_same_registry
from timenet.sources import (
    Found,
    FoundInConnector,
    FoundInRegistry,
    InstalledConnectors,
    NotFound,
    NotServed,
    ReleaseSource,
    resolve_release,
)
from timenet.types import DatasetMetadata, Domain, License, Task
from timenet.types.references import DatasetRef, split_ref


if TYPE_CHECKING:
    from timenet.torch import TimeFTorchDataset


# This code is at module scope, where `list` is the builtin type. The class has a method
# named ``list``. This placement stops that method name from shadowing the builtin in type
# annotations.
T = TypeVar("T")
_Metadatas: TypeAlias = list[DatasetMetadata]
_OrList: TypeAlias = T | list[T] | None


def default_local_registry(*, local_registry: str | Path | None = None) -> LocalRegistry:
    """Open the default local registry for downloaded and built releases.

    Args:
        local_registry: The local registry directory (``TIMENET_LOCAL_REGISTRY``, default ``~/.timenet/registry``).

    Returns:
        A writable local registry.

    Raises:
        TimeFValidationError: If the local registry path or another setting is invalid.
        TimeNetRegistryError: If ``local_registry`` names a remote registry.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by settings)
    return LocalRegistry(settings(local_registry=local_registry).local_registry_dir)


def default_sources(
    *,
    local_registry: str | Path | None = None,
    remote_registry: str | Path | None = None,
    auto_build: bool | None = None,
) -> list[ReleaseSource]:
    """Return the local registry, the remote registry and optional installed connectors in lookup order.

    A ``None`` argument uses the current environment setting or its default.

    Args:
        local_registry: The local registry directory (``TIMENET_LOCAL_REGISTRY``, default ``~/.timenet/registry``).
        remote_registry: A registry URL or path (``TIMENET_REMOTE_REGISTRY``, default ``timenet://``).
        auto_build: Whether to end with ``InstalledConnectors()`` (``TIMENET_AUTO_BUILD``, default true).

    Returns:
        A new list of sources in lookup order.

    Raises:
        TimeFValidationError: If an argument or environment setting is invalid.
        TimeNetRegistryError: If ``local_registry`` names a remote registry, or the remote registry cannot be opened.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by settings and open_registry)
    cfg = settings(local_registry=local_registry, remote_registry=remote_registry, auto_build=auto_build)
    store = LocalRegistry(cfg.local_registry_dir)
    remote = open_registry(cfg.remote_registry, label="TIMENET_REMOTE_REGISTRY")
    return [store, remote, *([InstalledConnectors()] if cfg.auto_build else [])]


def _resolve_ref(dataset_id: str, version: str | None) -> tuple[str, str | None]:
    """Split an ``org/id@version`` ref, then combine it with an explicit ``version``.

    Args:
        dataset_id: A dataset id. It can have a suffix of ``@<version>`` or ``@latest``.
        version: An explicit version, ``"latest"``, or ``None`` or ``""`` for no version.

    Returns:
        The bare dataset id and the requested version: exact, ``"latest"``, or ``None`` for no version.

    Raises:
        TimeFValidationError: The ref and ``version`` both give a version.
    """
    ref_id, ref_version = split_ref(dataset_id)
    if ref_version is not None and version:
        raise TimeFValidationError(f"version given twice: '@{ref_version}' in the id and version={version!r}")
    return ref_id, ref_version if ref_version is not None else (version or None)


def as_release_sources(items: Sequence[ReleaseSource | str | Path], *, keyword: str) -> list[ReleaseSource]:
    """Convert registry URLs and paths to release sources.

    Existing source objects keep their identity and position. A path opens a writable local registry.

    Args:
        items: A sequence of sources, registry URLs, or local paths.
        keyword: The argument name to show in errors.

    Returns:
        The sources in their original order, or an empty list for empty input.

    Raises:
        TimeFValidationError: If ``items`` is not a valid list.
        TimeNetRegistryError: If a registry URL or path is blank or cannot be opened.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by open_registry
    if isinstance(items, (str, os.PathLike)):
        raise TimeFValidationError(f"{keyword} takes a list of sources. Pass {keyword}=[{str(items)!r}].")
    if not isinstance(items, Sequence):
        raise TimeFValidationError(f"{keyword} takes a list of sources. Pass {keyword}=[{type(items).__name__}(...)].")
    opened: list[ReleaseSource] = []
    for index, item in enumerate(items):
        if isinstance(item, (str, os.PathLike)):
            path = item if isinstance(item, str) else Path(item)
            opened.append(open_registry(path, label=f"{keyword}[{index}]"))
        elif isinstance(item, (BaseRegistry, InstalledConnectors)):
            opened.append(item)
        else:
            raise TimeFValidationError(f"{keyword}[{index}] is not a source, a registry URL or a path: {item!r}")
    return opened


class TimeNet:
    """Browse, download, and read TimeNet datasets."""

    def __init__(self, sources: Sequence[ReleaseSource | str | Path] | None = None) -> None:
        """Create a client that searches dataset sources in order.

        Sources accept registries, registry URLs or paths, and ``InstalledConnectors()``.
        A path opens a writable ``LocalRegistry``. Builds land in the first ``LocalRegistry`` without
        ``read_only=True``, which ``InstalledConnectors()`` requires.

        Args:
            sources: Ordered sources, registry URLs or paths, or ``None`` for :func:`default_sources`.

        Raises:
            TimeFValidationError: If ``sources`` is not a non-empty list of valid sources, or has installed
                connectors but no writable local registry.
        """
        if sources is None:
            self._sources = tuple(default_sources())
        else:
            if isinstance(sources, (str, os.PathLike)):
                shown = repr(str(sources))
                raise TimeFValidationError(
                    f"sources takes a list of sources. Pass sources=[{shown}], or "
                    f"sources=[{shown}, InstalledConnectors()] to build what the path lacks"
                )
            self._sources = tuple(as_release_sources(sources, keyword="sources"))
            if not self._sources:
                raise TimeFValidationError("sources is empty. Pass at least one source, or None for the default list")
        self._store = next(
            (source for source in self._sources if isinstance(source, LocalRegistry) and not source.read_only), None
        )
        if self._store is None and any(isinstance(source, InstalledConnectors) for source in self._sources):
            raise TimeFValidationError(
                "InstalledConnectors() needs a writable local registry to build into. "
                "Add a LocalRegistry without read_only=True to the sources"
            )

    def list(self) -> _Metadatas:
        """Return metadata for every dataset in the selected registry.

        The client uses the first registry other than the local registry, or the local registry if it is the only one.

        Returns:
            One :class:`~timenet.types.DatasetMetadata` per dataset.
        """
        return self._catalog().list_datasets()

    def get(self, dataset_id: str, version: str | None = None) -> Manifest:
        """Return a dataset version's manifest from the selected registry.

        The client uses the first registry other than the local registry, or the local registry if it is the only one.

        Args:
            dataset_id: The dataset ID.
            version: The version string, or ``None`` for the latest.

        Returns:
            The dataset's manifest.
        """
        dataset_id, version = _resolve_ref(dataset_id, version)
        return self._catalog().get_manifest(dataset_id, version)

    def search(  # ruff: ignore[too-many-arguments]
        self,
        *,
        query: _OrList[str] = None,
        domain: _OrList[Domain] = None,
        task: _OrList[type[Task]] = None,
        license: _OrList[License] = None,
        time_series_spec: _OrList[str] = None,
        dataset_id: _OrList[str] = None,
        tag: _OrList[str] = None,
        limit: int = 100,
    ) -> _Metadatas:
        """Search the selected registry for datasets that match the filters.

        The client uses the first registry other than the local registry, or the local registry if it is the only one.

        Args:
            query: Free text terms for the name, description, and tags.
            domain: Match any of these domains.
            task: Match any of these task classes in the schema.
            license: Match any of these licenses.
            time_series_spec: Match all of these ``spec_type`` values.
            dataset_id: Match only these IDs.
            tag: Match all of these tags.
            limit: The maximum number of results.

        Returns:
            The matching dataset metadata.
        """
        return self._catalog().search(
            query=query,
            domain=domain,
            task=task,
            license=license,
            time_series_spec=time_series_spec,
            dataset_id=dataset_id,
            tag=tag,
            limit=limit,
        )

    def download(
        self,
        dataset_id: str,
        version: str | None = None,
        *,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> Path:
        """Copy a dataset release and its payload sources into the local registry.

        The release resolves as in :meth:`open`, without a build. The release and each payload source that the
        local registry lacks or holds damaged come from the first registry other than the local registry that
        provides them.

        Args:
            dataset_id: The dataset ID, optionally followed by ``@version``.
            version: An exact version, ``"latest"``, or ``None``, as in :meth:`open`.
            force: Copy the release again, even if the local registry holds a complete copy.
            progress: Receives step events and byte counts for each copied release.

        Returns:
            The release directory in the local registry.

        Raises:
            TimeFValidationError: If the sources lack a writable local registry, or both arguments specify a version.
            TimeNetAccessError: If a required release is only listed or a remote release is not open.
            TimeNetDatasetNotFoundError: If no registry other than the local registry provides the release.
            TimeNetVersionUnavailableError: If the version is missing and its connector builds another version.
            TimeNetMissingParentError: If no registry provides a missing payload source.
            TimeNetVersionConflictError: If the local registry holds a release of the copy with different files.
            TimeNetFileError: If a copied file is damaged or no registry can replace a damaged local copy.
            TimeFFormatError: If payload sources have conflicting file records or form a dependency cycle.
        """  # ruff: ignore[docstring-extraneous-exception] (raised by the plan and the copies)
        dataset_id, version = _resolve_ref(dataset_id, version)
        store = self._store
        if store is None:
            # Without a local registry, a download has no place to go.
            raise TimeFValidationError(
                "download() needs a writable local registry to copy into. "
                "Add a LocalRegistry without read_only=True to the sources"
            )
        lookups = Lookups()
        answer = lookups.resolve(self._sources, dataset_id, version)
        if isinstance(answer, NotServed):
            # A registry lists the release, but no source serves its files.
            error = answer.access_error(dataset_id)
            error.hint = _access_hint(dataset_id, self._sources)
            raise error
        if isinstance(answer, NotFound):
            # No source provides the release.
            try:
                raise_not_found(dataset_id, version, self._sources)
            except TimeNetVersionUnavailableError:
                # An installed connector builds another version, and the error names it.
                raise
            except TimeNetDatasetNotFoundError as error:
                raise TimeNetDatasetNotFoundError(f"{error} Add a registry that holds it to the sources.") from error
        release = DatasetRef(dataset_id=dataset_id, version=answer.version)
        try:
            plan = plan_download(release, store, sources=self._sources, force=force, lookups=lookups)
        except TimeNetFileError as error:
            # The local registry holds a damaged copy that no other registry provides.
            error.hint = self._file_hint(error)
            raise
        run_plan(plan, store, options=BuildOptions(progress=progress))
        return release_dir(store.root, release)

    def open(self, dataset_id: str, version: str | None = None) -> TimeFReader:
        """Open a dataset version for lazy reads, where it is.

        Opening reads the version's manifest, but neither its catalog nor any value. The reader's
        reads leave stored values behind value handles unless their load plan names them. Close
        the reader after use, or use it as a context manager. Remote backends open payload sources
        from their manifests upfront and read payload ranges on demand, reusing complete cached
        releases.

        The call reads each release where it is, and copies nothing. Without a version, it reads the
        newest version of the first source that serves the dataset. An exact version comes from the
        first source that serves it, and ``"latest"`` is the highest version of any source. Each
        payload source of the version resolves on its own, at its exact version. If the call builds
        the version, the build also needs the declared parents of the version and their payload
        sources.

        If only ``InstalledConnectors()`` provides a release, the call builds it into the local registry first.
        The call builds a payload source with its own connector, at the exact version of the
        source. The build gets the pins of the version. The build runs in staging. It publishes the
        source only if it reproduces the pinned files.

        A local copy is damaged if its manifest is present but a file is missing or has another
        size. The call checks each release that it opens in a local registry before it reads or
        builds anything. It never reads past a damaged copy to a later source, and it never builds
        a release over a damaged copy.

        Args:
            dataset_id: The dataset id.
            version: An exact version, ``"latest"``, or ``None`` for no version.

        Returns:
            A reader for the version.

        Raises:
            TimeNetAccessError: If a registry lists the version or a payload source but does not
                serve its files, and no later source provides it.
            TimeNetDatasetNotFoundError: If no source provides the version. The message says which
                source to add.
            TimeNetVersionUnavailableError: If no source provides the version, and the installed
                connector declares another version. The build does not start then.
            TimeNetInvalidCardError: If the card of a connector cannot be read.
            TimeNetMissingParentError: If no source provides a payload source, or a release that a
                build needs. The message says which source to add.
            TimeNetFileError: If a release that the call opens in a local registry has a missing
                file or a file of another size. If the damaged copy is in the local registry, the message says
                how to replace it: with the :meth:`download` that copies an intact release from a
                registry, else with a new build, if an installed connector builds the release.
            TimeFFormatError: If a payload source differs from the files that the version pins,
                for example because its build does not reproduce them.
            TimeNetBuildError: If a build runs but fails.
        """  # ruff: ignore[docstring-extraneous-exception] (raised by read_release and the builder)
        dataset_id, version = _resolve_ref(dataset_id, version)
        try:
            return TimeFReader(read_release(dataset_id, version, sources=self._sources, store=self._store))
        except TimeNetAccessError as error:
            error.hint = _access_hint(dataset_id, self._sources)
            raise
        except TimeNetFileError as error:
            error.hint = self._file_hint(error)
            raise
        except TimeNetMissingParentError as error:
            error.hint = _not_provided_hint(self._sources)
            raise
        except TimeNetDatasetNotFoundError as error:
            # A subclass, such as TimeNetVersionUnavailableError, names its own fix.
            if type(error) is not TimeNetDatasetNotFoundError:
                raise
            raise TimeNetDatasetNotFoundError(
                f"{str(error).rstrip('.')}. {_not_provided_hint(self._sources)}"
            ) from error

    def load(self, dataset_id: str, version: str | None = None) -> TimeFDataset:
        """Read a whole dataset version into memory, with every stored value loaded.

        The dataset needs no storage afterwards: its reader is closed before it is returned. For a
        large dataset, use :meth:`open` and read what you need instead.

        Args:
            dataset_id: The dataset id.
            version: An exact version, ``"latest"``, or ``None``, as in :meth:`open`.

        Returns:
            The dataset with every signal array, time offset, and text value loaded.

        Raises:
            TimeNetAccessError: If a registry lists the version or a payload source but does not
                serve its files, and no later source provides it.
            TimeNetDatasetNotFoundError: If no source provides the version.
            TimeNetVersionUnavailableError: If no source provides the version, and the installed
                connector declares another version. The build does not start then.
            TimeNetInvalidCardError: If the card of a connector cannot be read.
            TimeNetMissingParentError: If no source provides a payload source, or a release that a
                build needs.
            TimeNetFileError: If a release that the call opens in a local registry has a missing
                file or a file of another size.
            TimeFFormatError: If a payload source differs from the files that the version pins.
            TimeNetBuildError: If a build runs but fails.
        """  # ruff: ignore[docstring-extraneous-exception] (all come from open())
        with self.open(dataset_id, version) as reader:
            return reader.read(load=LoadPlan.ALL_EAGER)

    def _catalog(self) -> BaseRegistry:
        """Select a registry from the sources.

        Returns:
            The first registry other than the local registry, or the local registry if no other registry exists.

        Raises:
            TimeFValidationError: If no source is a registry.
        """
        # TODO(adrian): Remove temporary glue which maps these methods onto one registry until
        # list() and search() ask every source. This registry gives the results of the client of
        # one registry, from before the list of sources.
        registries = [source for source in self._sources if isinstance(source, BaseRegistry)]
        registry = next((registry for registry in registries if registry is not self._store), self._store)
        if registry is None:
            raise TimeFValidationError("list(), get() and search() need a registry in the sources")
        return registry

    def _file_hint(self, error: TimeNetFileError) -> str | None:
        """Suggest how to replace a damaged release in the local registry.

        Args:
            error: The error that names the damaged release.

        Returns:
            A repair hint, or ``None`` if the local registry has no damaged copy or the lookup raises a TimeNet error.
        """
        store = self._store
        if store is None:
            return None
        damaged = error.release
        try:
            held = store.find_manifest(damaged)
            if held is None or store.damage(damaged, held) is None:
                # The local registry holds no damaged copy, so the error is about another registry.
                return None
            found = _copy_source(self._sources, store, damaged)
        except TimeNetError:
            # The hint must not hide the error that it explains, for example if a registry is offline.
            return None
        return _repair_hint(damaged, found, store)

    def open_torch(
        self, dataset_id: str, version: str | None = None, *, load: LoadPlan = LoadPlan.ALL_EAGER
    ) -> TimeFTorchDataset:
        """Open a dataset version as a read-only PyTorch ``Dataset`` that reads it a batch at a time.

        The view holds an open reader from :meth:`open`. Close the view with its ``close()``
        method, or use it in a ``with`` block. This method needs the ``torch`` extra
        (``pip install 'timenet[torch]'``). The code imports the torch view lazily, so base users do
        not need torch installed.

        Args:
            dataset_id: The dataset id.
            version: An exact version, ``"latest"``, or ``None``, as in :meth:`open`.
            load: The plan that each batch loads with. A text-only plan, for example, skips the
                arrays of candidate and target Records.

        Returns:
            A :class:`~timenet.torch.TimeFTorchDataset` over a reader of the version, which reads
            and loads one batch of records at a time.
        """
        from timenet.torch import TimeFTorchDataset  # ruff: ignore[import-outside-top-level]

        return TimeFTorchDataset.from_reader(self.open(dataset_id, version), load=load)


def _not_provided_hint(sources: Sequence[ReleaseSource]) -> str:
    """Suggest sources for a missing dataset release.

    Args:
        sources: The configured sources.

    Returns:
        Instructions to add a registry or installed connectors.
    """
    if any(isinstance(source, InstalledConnectors) for source in sources):
        return "Add a registry that holds it to the sources."
    return "Add a registry that holds it to the sources, or add InstalledConnectors() to build it."


def _access_hint(dataset_id: str, sources: Sequence[ReleaseSource]) -> str:
    """Suggest how to obtain a release whose files are unavailable.

    Args:
        dataset_id: The dataset ID without a version suffix.
        sources: The configured sources.

    Returns:
        A hint with options to add a source or build the release locally.
    """
    build = f"`timenet-build build {dataset_id}`"
    if any(isinstance(source, InstalledConnectors) for source in sources):
        return (
            "No source serves its files, and no installed connector builds this release. "
            f"Install its connector package and build it locally with {build}."
        )
    return f"No source serves its files. Add InstalledConnectors() to the sources, or build it locally with {build}."


def _copy_source(sources: Sequence[ReleaseSource], store: LocalRegistry, release: DatasetRef) -> Found | None:
    """Find the answer of a release source outside the local registry.

    Args:
        sources: The sources to search, in order.
        store: The local registry to exclude.
        release: The dataset ID and exact version.

    Returns:
        The answer of the source that provides or builds the release, or ``None`` if none exists.
    """
    others = [source for source in sources if not is_same_registry(source, store)]
    answer = resolve_release(others, release.dataset_id, str(release.version))
    return answer if isinstance(answer, Found) else None


def _repair_hint(release: DatasetRef, found: Found | None, store: LocalRegistry) -> str:
    """Suggest how to replace a damaged release in the local registry.

    Args:
        release: The damaged release.
        found: The answer of the available source, or ``None`` if none exists.
        store: The local registry.

    Returns:
        A message with a download or build command, or the reason replacement is unavailable.
    """
    if isinstance(found, FoundInRegistry) and not is_same_registry(found.registry, store):
        # A registry other than the local registry provides the release.
        return f"Copy an intact release into the local registry with download('{release}')."
    if isinstance(found, FoundInConnector):
        # Only an installed connector provides the release.
        command = f"timenet-build build {release.dataset_id} --local-registry {shlex.quote(str(store.root))} --force"
        return f"Build it again with `{command}`."
    # Only the local registry holds the release.
    return "No registry other than the local registry provides it, so download() cannot replace it."
