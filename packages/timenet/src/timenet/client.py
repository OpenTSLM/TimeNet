"""The :class:`TimeNet` class is the SDK's single entry point for using TimeNet from code.

A client reads releases through an ordered list of sources, and builds a release that only
``InstalledConnectors()`` provides into its first local registry that is not read-only.
:func:`default_sources` makes the list of ``TimeNet()``.
"""

# The public API has a method named ``list``. Deferred annotations keep the type hint
# ``list[str]`` resolved to the builtin type, not to the method.
from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, TypeVar

from timenet.config import settings
from timenet.coordinator.read import read_release
from timenet.dataset import TimeFDataset
from timenet.errors import (
    TimeFValidationError,
    TimeNetAccessError,
    TimeNetDatasetNotFoundError,
    TimeNetMissingParentError,
)
from timenet.manifest import Manifest
from timenet.reader import LoadPlan, TimeFReader
from timenet.registry import BaseRegistry, LocalRegistry, open_registry
from timenet.registry._paths import release_dir
from timenet.registry.base import ProgressCallback
from timenet.sources import InstalledConnectors, ReleaseSource
from timenet.types import DatasetMetadata, Domain, License, Task
from timenet.types.access import require_open_access
from timenet.types.references import DatasetRef, split_ref


if TYPE_CHECKING:
    from timenet.torch import TimeFTorchDataset


# This code is at module scope, where `list` is the builtin type. The class has a method
# named ``list``. This placement stops that method name from shadowing the builtin in type
# annotations.
T = TypeVar("T")
_Metadatas: TypeAlias = list[DatasetMetadata]
_OrList: TypeAlias = T | list[T] | None


def default_sources(*, remote_registry: str | Path | None = None) -> list[ReleaseSource]:
    """Return the default sources: the store, a registry, and the installed connectors.

    The store is ``$TIMENET_REGISTRY`` if it names a local path, else ``<TIMENET_HOME>/registry``.
    The registry is ``remote_registry``, else ``$TIMENET_REGISTRY`` if it is not local, else
    ``timenet://``, and a local ``$TIMENET_REGISTRY`` without ``remote_registry`` leaves it out.

    Args:
        remote_registry: The registry after the store, as a URL, a ``file://`` URI or a local path.

    Returns:
        A new list, made from the environment at the time of the call.
    """
    # TODO(adrian): Remove temporary glue which maps TIMENET_REGISTRY and TIMENET_HOME onto the
    # list until the variables of the defaults (TIMENET_LOCAL_REGISTRY, TIMENET_REMOTE_REGISTRY,
    # TIMENET_AUTO_BUILD) replace them.
    cfg = settings()
    configured = None if cfg.registry is None else open_registry(cfg.registry)
    local = configured if isinstance(configured, LocalRegistry) else None
    store = LocalRegistry(cfg.registry_path if local is None else local.root)
    if remote_registry is not None:
        remote = open_registry(remote_registry)
    elif local is not None:
        # A client of a local $TIMENET_REGISTRY read no other registry, and neither does this list.
        return [store, InstalledConnectors()]
    else:
        remote = configured if configured is not None else open_registry("timenet://")
    return [store, remote, InstalledConnectors()]


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


def _open_sources(sources: Sequence[ReleaseSource | str | Path] | None) -> tuple[ReleaseSource, ...]:
    """Validate sources and open registry URLs and paths.

    Args:
        sources: The value of ``sources``, other than ``None``.

    Returns:
        The sources, in order.

    Raises:
        TimeFValidationError: If sources are not a nonempty sequence of valid sources, registry URLs, or paths.
        TimeNetRegistryError: If a registry URL or path is blank or cannot be opened.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by open_registry
    if isinstance(sources, (str, os.PathLike)):
        shown = repr(str(sources))
        raise TimeFValidationError(
            f"sources takes a list of sources. Pass sources=[{shown}], or "
            f"sources=[{shown}, InstalledConnectors()] to build what the path lacks"
        )
    if not isinstance(sources, Sequence):
        raise TimeFValidationError(f"sources takes a list of sources. Pass sources=[{type(sources).__name__}(...)]")
    if not sources:
        raise TimeFValidationError("sources is empty. Pass at least one source, or None for the default list")
    opened: list[ReleaseSource] = []
    for index, item in enumerate(sources):
        if isinstance(item, (str, os.PathLike)):
            opened.append(open_registry(item if isinstance(item, str) else Path(item), label=f"sources[{index}]"))
        elif isinstance(item, (BaseRegistry, InstalledConnectors)):
            opened.append(item)
        else:
            raise TimeFValidationError(f"sources[{index}] is not a source, a registry URL or a path: {item!r}")
    return tuple(opened)


class TimeNet:
    """Browse, download, and read TimeNet datasets."""

    def __init__(self, sources: Sequence[ReleaseSource | str | Path] | None = None) -> None:
        """Create a client that reads releases through an ordered list of sources.

        A source is a registry, a registry URL or path, or ``InstalledConnectors()`` (see
        :mod:`timenet.sources`). A path opens a writable :class:`~timenet.registry.LocalRegistry`.
        Builds land in the first local registry that is not ``read_only=True``, and a list with
        ``InstalledConnectors()`` needs one.

        Args:
            sources: The sources, in order. ``None`` means :func:`default_sources`.

        Raises:
            TimeFValidationError: If ``sources`` is not a valid list, or has ``InstalledConnectors()``
                without a writable local registry.
        """
        self._sources = tuple(default_sources()) if sources is None else _open_sources(sources)
        self._store = next(
            (source for source in self._sources if isinstance(source, LocalRegistry) and not source.read_only), None
        )
        if self._store is None and any(isinstance(source, InstalledConnectors) for source in self._sources):
            raise TimeFValidationError(
                "InstalledConnectors() needs a writable local registry to build into. "
                "Add a LocalRegistry without read_only=True to the sources"
            )

    def list(self) -> _Metadatas:
        """Return the metadata of every dataset in one registry of the sources.

        The registry is the first registry of the sources that is not the local registry of the
        builds, else that local registry.

        Returns:
            One :class:`~timenet.types.DatasetMetadata` per dataset.
        """
        return self._catalog().list_datasets()

    def get(self, dataset_id: str, version: str | None = None) -> Manifest:
        """Return a dataset's manifest from the registry that :meth:`list` asks.

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
        """Search the registry that :meth:`list` asks.

        This method mirrors :meth:`~timenet.registry.BaseRegistry.search`.

        Args:
            query: Free text terms for the name, description, and tags.
            domain: Keep datasets that share any of these domains.
            task: Keep datasets whose schema includes any of these task classes.
            license: Keep datasets with any of these licenses.
            time_series_spec: Keep datasets that declare all of these ``spec_type`` values.
            dataset_id: Keep only these ids.
            tag: Keep datasets that declare all of these tags.
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
        progress_cb: ProgressCallback | None = None,
    ) -> Path:
        """Fetch a version and each of its payload sources from the registry that :meth:`list` asks.

        The copy lands in ``$TIMENET_STORAGE``, else in ``<TIMENET_HOME>/storage``. Parents that
        only the build needs are not downloaded.

        Args:
            dataset_id: The dataset ID.
            version: The version string, or ``None`` for the latest.
            force: Download again, even if an up-to-date copy already exists.
            progress_cb: Called with each file's byte count as it lands, for a progress display.

        Returns:
            The local ``<storage>/<dataset_id>/<version>/`` directory.
        """
        # TODO(adrian): Remove temporary glue which maps download() onto a copy from one registry
        # into the storage directory, as before the list of sources, until a download into the
        # local registry replaces it.
        dataset_id, version = _resolve_ref(dataset_id, version)
        registry = self._catalog()
        self._reject_unhosted_access(dataset_id, version)
        manifest = registry.get_manifest(dataset_id, version)
        resolved = str(manifest.metadata.dataset_version)
        target = release_dir(settings().storage_dir, DatasetRef.parse(f"{dataset_id}@{resolved}"))
        # The registry owns the fetch: the base implementation stages each file and swaps atomically,
        # and a remote registry overrides it to resolve and stream every file in parallel.
        registry.download_version(dataset_id, resolved, target, force=force, manifest=manifest, progress_cb=progress_cb)
        return target

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
                file or a file of another size.
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
        """Return the registry that :meth:`list`, :meth:`get`, :meth:`search`, and :meth:`download` ask.

        Returns:
            The first registry of the sources that is not the local registry of the builds, else
            that local registry.

        Raises:
            TimeFValidationError: If the sources hold no registry.
        """
        # TODO(adrian): Remove temporary glue which maps these methods onto one registry until
        # list() and search() ask every source. This registry gives the results of the client of
        # one registry, from before the list of sources.
        registries = [source for source in self._sources if isinstance(source, BaseRegistry)]
        registry = next((registry for registry in registries if registry is not self._store), self._store)
        if registry is None:
            raise TimeFValidationError("list(), get(), search() and download() need a registry in the sources")
        return registry

    # TODO(adrian): Remove temporary glue which checks access only for the interim download()
    # until a later change replaces download().
    def _reject_unhosted_access(self, dataset_id: str, version: str | None) -> None:
        """Enforce the registry's open access requirement.

        Args:
            dataset_id: The dataset ID without a version suffix.
            version: The resolved version, or ``None`` for the latest.

        Raises:
            TimeNetAccessError: If the dataset lacks open access and the registry requires it.
        """
        registry = self._catalog()
        if not registry.requires_open_access:
            return
        try:
            manifest = registry.get_manifest(dataset_id, version)
            require_open_access(manifest.metadata, manifest.lineage)
        except TimeNetAccessError as error:
            error.hint = _unhosted_hint(dataset_id)
            raise

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
    """Tell the caller which source can provide a release that no source provides.

    Args:
        sources: The sources of the client.

    Returns:
        The hint as a full sentence. It names ``InstalledConnectors()`` only if the sources lack it.
    """
    if any(isinstance(source, InstalledConnectors) for source in sources):
        return "Add a registry that holds it to the sources."
    return "Add a registry that holds it to the sources, or add InstalledConnectors() to build it."


def _access_hint(dataset_id: str, sources: Sequence[ReleaseSource]) -> str:
    """Tell the caller how to get a release that a registry lists but does not serve.

    Args:
        dataset_id: The bare dataset id.
        sources: The sources of the client.

    Returns:
        The hint as full sentences. It asks for ``InstalledConnectors()`` only if the sources lack it.
    """
    build = f"`timenet-build build {dataset_id}`"
    if any(isinstance(source, InstalledConnectors) for source in sources):
        return (
            "No source serves its files, and no installed connector builds this release. "
            f"Install its connector package and build it locally with {build}."
        )
    return f"No source serves its files. Add InstalledConnectors() to the sources, or build it locally with {build}."


# TODO(adrian): Remove temporary glue which gives a hint only for the interim download() until a
# later change replaces download().
def _unhosted_hint(dataset_id: str) -> str:
    """Tell the caller how to get a dataset whose files the download registry does not serve.

    Returns:
        A sentence with the dataset's build command.
    """
    return f"TimeNet does not host its data, so build it locally with `timenet-build build {dataset_id}`."
