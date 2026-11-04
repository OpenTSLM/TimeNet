"""Find, read, and download datasets with :class:`TimeNet`.

Clients search their sources in order. Builds and downloads go into the local store.
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
from timenet.config import check_store, settings
from timenet.coordinator.build import run_plan
from timenet.coordinator.plan import Lookups, plan_download
from timenet.coordinator.read import raise_not_found, read_release
from timenet.dataset import TimeFDataset
from timenet.errors import (
    TimeFValidationError,
    TimeNetAccessError,
    TimeNetDatasetNotFoundError,
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


def default_store(*, local_registry: str | Path | None = None) -> LocalRegistry:
    """Open the default local store for downloaded and built releases.

    Args:
        local_registry: The store directory (``TIMENET_LOCAL_REGISTRY``, default ``~/.timenet/registry``).

    Returns:
        A local registry with ``store=True``.

    Raises:
        TimeFValidationError: If the store path or another setting is invalid.
        TimeNetRegistryError: If ``local_registry`` names a remote registry.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by settings)
    return LocalRegistry(settings(local_registry=local_registry).local_registry_dir, store=True)


def default_sources(
    *,
    local_registry: str | Path | None = None,
    remote_registry: str | Path | None = None,
    auto_build: bool | None = None,
) -> list[ReleaseSource]:
    """Return the store, a registry and optional installed connectors in lookup order.

    Remote registries reuse complete matching releases in the store.
    A ``None`` argument uses the current environment setting or its default.

    Args:
        local_registry: The store directory (``TIMENET_LOCAL_REGISTRY``, default ``~/.timenet/registry``).
        remote_registry: A registry URL or path (``TIMENET_REMOTE_REGISTRY``, default ``timenet://``).
        auto_build: Whether to end with ``InstalledConnectors()`` (``TIMENET_AUTO_BUILD``, default true).

    Returns:
        A new list of sources in lookup order.

    Raises:
        TimeFValidationError: If an argument or environment setting is invalid.
        TimeNetRegistryError: If the local store is remote or the other registry cannot be opened.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by settings and open_registry)
    store = default_store(local_registry=local_registry)
    cfg = settings(remote_registry=remote_registry, auto_build=auto_build)
    remote = open_registry(cfg.remote_registry, store=store.root, label="TIMENET_REMOTE_REGISTRY")
    return [store, remote, *([InstalledConnectors()] if cfg.auto_build else [])]


def _resolve_ref(dataset_id: str, version: str | None) -> tuple[str, str | None]:
    """Split an ``org/id@version`` ref, then combine it with an explicit ``version``.

    Args:
        dataset_id: A dataset id. It can have a suffix of ``@<version>`` or ``@latest``.
        version: An explicit version, or ``None`` for the latest version.

    Returns:
        The bare dataset id and the resolved version. ``None`` means the latest version.

    Raises:
        TimeFValidationError: The ref and ``version`` both give a version.
    """
    ref_id, ref_version = split_ref(dataset_id)
    if ref_version is not None and version is not None:
        raise TimeFValidationError(f"version given twice: '@{ref_version}' in the id and version={version!r}")
    return ref_id, ref_version if ref_version is not None else version


def as_release_sources(
    items: Sequence[ReleaseSource | str | Path], *, keyword: str, store: str | Path | None = None
) -> list[ReleaseSource]:
    """Check a list of sources, and open each string or path in it as a registry.

    A string or a :class:`~pathlib.Path` is a registry: a URL, a ``file://`` URI, or a local path. A
    source stays as it is.

    Args:
        items: A sequence of sources, registry URLs, or local paths.
        keyword: The argument name to show in errors.
        store: The store directory whose copies of the same releases a registry URL reads, or ``None`` for none.

    Returns:
        The sources, in order. An empty list stays empty.

    Raises:
        TimeFValidationError: If ``items`` is not a valid list, or ``store`` overlaps ``TIMENET_CACHE``.
        TimeNetRegistryError: If a registry URL or path is blank or cannot be opened.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by open_registry
    if isinstance(items, (str, os.PathLike)) or not isinstance(items, Sequence):
        shown = repr(str(items)) if isinstance(items, (str, os.PathLike)) else f"{type(items).__name__}(...)"
        if keyword == "sources" and isinstance(items, (str, os.PathLike)):
            raise TimeFValidationError(
                f"sources takes a list of sources. Pass sources=[{shown}], or "
                f"sources=[LocalRegistry({shown}, store=True), InstalledConnectors()] to build what the path lacks"
            )
        raise TimeFValidationError(f"{keyword} takes a list of sources. Pass {keyword}=[{shown}].")
    if store is not None:
        check_store(Path(store))
    opened: list[ReleaseSource] = []
    for index, item in enumerate(items):
        # A path has a method named resolve, so it passes the protocol check. Paths come first.
        if isinstance(item, (str, os.PathLike)):
            path = item if isinstance(item, str) else Path(item)
            opened.append(open_registry(path, store=store, label=f"{keyword}[{index}]"))
        elif isinstance(item, ReleaseSource):
            opened.append(item)
        else:
            raise TimeFValidationError(f"{keyword}[{index}] is not a source, a registry URL or a path: {item!r}")
    return opened


def _store_root(items: object) -> Path | None:
    """Return the root of the store among the sources of a list that are not strings or paths.

    Args:
        items: The value of ``sources``, other than ``None``.

    Returns:
        The root of the first :class:`~timenet.registry.LocalRegistry` with ``store=True``, or ``None``
        if the list holds none or is not a list.
    """
    if isinstance(items, (str, os.PathLike)) or not isinstance(items, Sequence):
        return None
    return next((item.root for item in items if isinstance(item, LocalRegistry) and item.is_store), None)


class TimeNet:
    """Browse, download, and read TimeNet datasets."""

    def __init__(self, sources: Sequence[ReleaseSource | str | Path] | None = None) -> None:
        """Create a client with an ordered list of dataset sources.

        ``InstalledConnectors()`` requires one ``LocalRegistry`` with ``store=True`` to hold builds.
        A registry URL reads the copies of the same releases in that store, and without a store none.

        Args:
            sources: Ordered registries, URLs, paths or connectors, or ``None`` for :func:`default_sources`.

        Raises:
            TimeFValidationError: If ``sources`` is not a valid list, has a wrong number of stores,
                or has a store that overlaps ``TIMENET_CACHE``.
        """
        if sources is None:
            self._sources = tuple(default_sources())
        else:
            self._sources = tuple(as_release_sources(sources, keyword="sources", store=_store_root(sources)))
            if not self._sources:
                raise TimeFValidationError("sources is empty. Pass at least one source, or None for the default list")
        stores = [source for source in self._sources if isinstance(source, LocalRegistry) and source.is_store]
        if len(stores) > 1:
            raise TimeFValidationError(
                f"sources has {len(stores)} stores: {', '.join(str(store) for store in stores)}. "
                "Pass store=True to at most one LocalRegistry"
            )
        self._store = stores[0] if stores else None
        if self._store is None and any(isinstance(source, InstalledConnectors) for source in self._sources):
            raise TimeFValidationError(
                "InstalledConnectors() builds into the store, so the sources need one. "
                "Add LocalRegistry(<path>, store=True) to the sources"
            )

    def list(self) -> _Metadatas:
        """Return metadata for every dataset in the selected registry.

        The client uses the first registry other than the build store, or the store if it is the only registry.

        Returns:
            One :class:`~timenet.types.DatasetMetadata` per dataset.
        """
        return self._catalog().list_datasets()

    def get(self, dataset_id: str, version: str | None = None) -> Manifest:
        """Return a dataset version's manifest from the selected registry.

        The client uses the first registry other than the build store, or the store if it is the only registry.

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

        The client uses the first registry other than the build store, or the store if it is the only registry.

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
        """Copy a dataset release and its payload sources into the store.

        The release resolves as in :meth:`open`, without a build. The release and each payload source that the
        store lacks or holds damaged come from the first registry other than the store that provides them.

        Args:
            dataset_id: The dataset ID, optionally followed by ``@version``.
            version: The version string, or ``None`` for the latest.
            force: Copy the release again, even if the store holds a complete copy.
            progress: Receives step events and byte counts for each copied release.

        Returns:
            The release directory in the store.

        Raises:
            TimeFValidationError: If the sources lack a store or both arguments specify a version.
            TimeNetAccessError: If a required release is only listed or a remote release is not open.
            TimeNetDatasetNotFoundError: If no registry other than the store provides the release.
            TimeNetVersionUnavailableError: If the version is missing and its connector builds another version.
            TimeNetMissingParentError: If no registry provides a missing payload source.
            TimeNetVersionConflictError: If the store holds a release of the copy with different files.
            TimeNetFileError: If a copied file is damaged or no registry can replace a damaged local copy.
            TimeFFormatError: If payload sources have conflicting file records or form a dependency cycle.
        """  # ruff: ignore[docstring-extraneous-exception] (raised by the plan and the copies)
        dataset_id, version = _resolve_ref(dataset_id, version)
        store = self._store
        if store is None:
            # Without a store, a download has no place to go.
            raise TimeFValidationError(
                "download() copies into the store; add LocalRegistry(<path>, store=True) to the sources"
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
            # The store holds a damaged copy that no other registry provides.
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

        The call reads each release where it is, and copies nothing. An exact version comes from
        the first source that provides it. "Latest" is the highest version that any source
        provides. Each payload source of the version resolves in the same way, on its own, at its
        exact version. If the call builds the version, the build also needs the declared parents of
        the version and their payload sources.

        If only ``InstalledConnectors()`` provides a release, the call builds it into the store first.
        The call builds a payload source with its own connector, at the exact version of the
        source. The build gets the pins of the version. The build runs in staging. It publishes the
        source only if it reproduces the pinned files.

        A local copy is damaged if its manifest is present but a file is missing or has another
        size. The call checks each release that it opens in a local registry before it reads or
        builds anything. It never reads past a damaged copy to a later source, and it never builds
        a release over a damaged copy.

        Args:
            dataset_id: The dataset id.
            version: The version string, or ``None`` for the latest.

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
                file or a file of another size. If the damaged copy is in the store, the message says
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
            version: The version string, or ``None`` for the latest.

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
            The first registry other than the store, or the store if no other registry exists.

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
        """Suggest how to replace a damaged release in the store.

        Args:
            error: The error that names the damaged release.

        Returns:
            A repair hint, or ``None`` if the store has no damaged copy or the lookup fails.
        """
        store = self._store
        if store is None:
            return None
        damaged = error.release
        try:
            held = store.find_manifest(damaged)
            if held is None or store.damage(damaged, held) is None:
                # The store holds no damaged copy, so the error is about another registry.
                return None
            entry = _copy_source(self._sources, store, damaged)
        except Exception:
            # The hint must not hide the error that it explains, for example if a registry is offline.
            return None
        return _repair_hint(damaged, entry, store)

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
            version: The version string, or ``None`` for the latest.
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


def _copy_source(sources: Sequence[ReleaseSource], store: LocalRegistry, release: DatasetRef) -> ReleaseSource | None:
    """Find a release source outside the store.

    Args:
        sources: The sources to search, in order.
        store: The store to exclude.
        release: The dataset ID and exact version.

    Returns:
        The source that provides or builds the release, or ``None`` if none exists.
    """
    others = [source for source in sources if not is_same_registry(source, store)]
    answer = resolve_release(others, release.dataset_id, str(release.version))
    return answer.entry if isinstance(answer, Found) else None


def _repair_hint(release: DatasetRef, entry: ReleaseSource | None, store: LocalRegistry) -> str:
    """Suggest how to replace a damaged release in the store.

    Args:
        release: The damaged release.
        entry: The available source, or ``None`` if none exists.
        store: The store.

    Returns:
        A message with a download or build command, or the reason replacement is unavailable.
    """
    if isinstance(entry, BaseRegistry) and not is_same_registry(entry, store):
        # A registry other than the store provides the release.
        return f"Copy an intact release into the store with download('{release}')."
    if isinstance(entry, InstalledConnectors):
        # Only an installed connector provides the release.
        command = f"timenet-build build {release.dataset_id} --out {shlex.quote(str(store.root))} --force"
        return f"Build it again with `{command}`."
    # Only the store holds the release.
    return "No registry other than the store provides it, so download() cannot replace it."
