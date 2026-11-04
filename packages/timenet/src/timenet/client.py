"""The :class:`TimeNet` class is the SDK's single entry point for using TimeNet from code.

This class wraps a :class:`~timenet.registry.BaseRegistry` (the catalog) and a local storage path
(the download cache). It exposes these methods: ``list``, ``get``, ``search``, ``download``,
``open``, ``load``, and ``open_torch``. ``open`` and ``load`` read a release where it is, through
the read coordinator in :mod:`timenet.coordinator.read`. Against a local registry, they build a
release that only an installed connector provides. Against a remote registry, they never run
connector code.
"""

# The public API has a method named ``list``. Deferred annotations keep the type hint
# ``list[str]`` resolved to the builtin type, not to the method.
from __future__ import annotations

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
_Sources: TypeAlias = list[ReleaseSource]


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


class TimeNet:
    """Browse, download, and read datasets from a registry."""

    def __init__(
        self, registry: str | Path | BaseRegistry | None = None, *, storage_path: str | Path | None = None
    ) -> None:
        """Open a client for a registry.

        The client selects the registry in this order: the ``registry`` argument, then
        ``$TIMENET_REGISTRY``, then the hosted TimeNet registry (``timenet://``).

        Args:
            registry: A registry instance, URL, ``file://`` URI, or local path.
            storage_path: The directory for cached downloads. The default is
                ``$TIMENET_STORAGE``, or ``<TIMENET_HOME>/storage`` if that variable is not set.
        """
        given_registry = registry if isinstance(registry, BaseRegistry) else None
        cfg = settings(
            registry=None if given_registry is not None or registry is None else str(registry),
            storage=storage_path,
        )
        if given_registry is not None:
            self._registry = given_registry
        elif cfg.registry is not None:
            self._registry = open_registry(cfg.registry, cache_dir=cfg.storage_dir)
        else:
            self._registry = open_registry("timenet://", cache_dir=cfg.storage_dir)
        self._storage = cfg.storage_dir

    def list(self) -> _Metadatas:
        """Return the metadata of every dataset in the registry.

        Returns:
            One :class:`~timenet.types.DatasetMetadata` per dataset.
        """
        return self._registry.list_datasets()

    def get(self, dataset_id: str, version: str | None = None) -> Manifest:
        """Return a dataset's manifest.

        Args:
            dataset_id: The dataset id.
            version: The version string, or ``None`` for the latest.

        Returns:
            The dataset's manifest.
        """
        dataset_id, version = _resolve_ref(dataset_id, version)
        return self._registry.get_manifest(dataset_id, version)

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
        """Search the registry. This method mirrors :meth:`~timenet.registry.BaseRegistry.search`.

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
        return self._registry.search(
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
        """Fetch a version and each of its payload sources into local storage.

        Build-only parents remain recorded in the manifest and are not downloaded.

        Args:
            dataset_id: The dataset id.
            version: The version string, or ``None`` for the latest.
            force: Download again, even if an up-to-date copy already exists.
            progress_cb: Called with each file's byte count as it lands, for a progress display.

        Returns:
            The local ``<storage>/<dataset_id>/<version>/`` directory.
        """
        dataset_id, version = _resolve_ref(dataset_id, version)
        self._reject_unhosted_access(dataset_id, version)
        manifest = self._registry.get_manifest(dataset_id, version)
        resolved = str(manifest.metadata.dataset_version)
        target = release_dir(self._storage, DatasetRef.parse(f"{dataset_id}@{resolved}"))
        # The registry owns the fetch: the base implementation stages each file and swaps atomically,
        # and a remote registry overrides it to resolve and stream every file in parallel.
        self._registry.download_version(
            dataset_id, resolved, target, force=force, manifest=manifest, progress_cb=progress_cb
        )
        return target

    def open(
        self,
        dataset_id: str,
        version: str | None = None,
        *,
        auto_build: bool = True,
        fetch_from: str | BaseRegistry | None = None,
    ) -> TimeFReader:
        """Open a dataset version for lazy reads, where it is.

        Opening reads the version's manifest, but neither its catalog nor any value. The reader's
        reads leave stored values behind value handles unless their load plan names them. Close
        the reader after use, or use it as a context manager. Remote backends open payload sources
        from their manifests upfront and read payload ranges on demand, reusing complete cached
        releases.

        The call reads each release where it is, and copies nothing. A local client asks its own
        registry first, then the registry that ``fetch_from`` names, then the installed connectors
        if ``auto_build`` is set. Without a version, the call reads the newest version of the first
        of them that serves the dataset. An exact version comes from the first of them that serves
        it, and ``"latest"`` is the highest version that any of them provides. Each payload source
        of the version resolves on its own, at its exact version. If the call builds the version,
        the build also needs the declared parents of the version and their payload sources.

        If only a connector provides a release, the call builds it into the local registry first.
        The call builds a payload source with its own connector, at the exact version of the
        source. The build gets the pins of the version. The build runs in staging. It publishes the
        source only if it reproduces the pinned files.

        A local copy is damaged if its manifest is present but a file is missing or has another
        size. The call checks each release that it opens in a local registry before it reads or
        builds anything. It never reads past a damaged copy to a later source, and it never builds
        a release over a damaged copy. Against a remote registry, ``open`` never runs
        connector code. If a remote client gets ``fetch_from``, ``open`` raises an error.

        Args:
            dataset_id: The dataset id.
            version: An exact version, ``"latest"``, or ``None`` for no version.
            auto_build: Build a release that only an installed connector provides. Set false to
                raise instead.
            fetch_from: A registry, or its URI, that a local client reads a release from if its own
                registry does not provide it.

        Returns:
            A reader for the version.

        Raises:
            TimeNetAccessError: If a registry that is not local lists the version or a payload
                source but does not serve its files, and no later source provides it.
            TimeNetDatasetNotFoundError: If no source provides the version.
            TimeNetVersionUnavailableError: If no source provides the version, and the installed
                connector declares another version. The build does not start then.
            TimeNetInvalidCardError: If the card of a connector cannot be read.
            TimeNetMissingParentError: If no source provides a payload source, or a release that a
                build needs. The message names each keyword that the call did not use.
            TimeNetFileError: If a release that the call opens in a local registry has a missing
                file or a file of another size.
            TimeFFormatError: If a payload source differs from the files that the version pins,
                for example because its build does not reproduce them.
            TimeNetBuildError: If a build runs but fails.
            TimeFValidationError: If a remote client gets ``fetch_from``.
        """  # ruff: ignore[docstring-extraneous-exception] (raised by read_release and the builder)
        dataset_id, version = _resolve_ref(dataset_id, version)
        sources, store = self._read_sources(auto_build=auto_build, fetch_from=fetch_from)
        # Maps fetch_from and auto_build onto hints for a local client.
        try:
            return TimeFReader(read_release(dataset_id, version, sources=sources, store=store))
        except TimeNetAccessError as error:
            if store is None:
                error.hint = _unhosted_hint(dataset_id)
            elif not auto_build:
                error.hint = "Build it from its connector with auto_build=True."
            raise
        except TimeNetMissingParentError as error:
            if store is not None:
                error.hint = _missing_hint(auto_build=auto_build, fetch_from=fetch_from)
            raise
        except TimeNetDatasetNotFoundError as error:
            hint = None if store is None else _missing_hint(auto_build=auto_build, fetch_from=fetch_from)
            # A subclass, such as TimeNetVersionUnavailableError, names its own fix.
            if hint is None or type(error) is not TimeNetDatasetNotFoundError:
                raise
            raise TimeNetDatasetNotFoundError(f"{str(error).rstrip('.')}. {hint}") from error

    def load(
        self,
        dataset_id: str,
        version: str | None = None,
        *,
        auto_build: bool = True,
        fetch_from: str | BaseRegistry | None = None,
    ) -> TimeFDataset:
        """Read a whole dataset version into memory, with every stored value loaded.

        The dataset needs no storage afterwards: its reader is closed before it is returned. For a
        large dataset, use :meth:`open` and read what you need instead.

        Args:
            dataset_id: The dataset id.
            version: An exact version, ``"latest"``, or ``None``, as in :meth:`open`.
            auto_build: Build a release that only an installed connector provides. Set false to
                raise instead.
            fetch_from: A registry, or its URI, that a local client reads a release from if its own
                registry does not provide it.

        Returns:
            The dataset with every signal array, time offset, and text value loaded.

        Raises:
            TimeNetAccessError: If a registry that is not local lists the version or a payload
                source but does not serve its files, and no later source provides it.
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
            TimeFValidationError: If a remote client gets ``fetch_from``.
        """  # ruff: ignore[docstring-extraneous-exception] (all come from open())
        with self.open(dataset_id, version, auto_build=auto_build, fetch_from=fetch_from) as reader:
            return reader.read(load=LoadPlan.ALL_EAGER)

    def _read_sources(
        self, *, auto_build: bool, fetch_from: str | BaseRegistry | None
    ) -> tuple[_Sources, LocalRegistry | None]:
        """Return the sources of a read, in order, and the local registry that its builds go to.

        Maps ``fetch_from`` and ``auto_build`` onto a list of sources for the read coordinator. A
        local client reads from its own registry, then from ``fetch_from``, then from the installed
        connectors if ``auto_build`` is set. Its builds go to its own registry. A remote client
        reads only from its own registry, and it never builds.

        Returns:
            The sources, and the local registry, or ``None`` for a remote client.

        Raises:
            TimeFValidationError: If a remote client gets ``fetch_from``.
        """
        registry = self._registry
        if not isinstance(registry, LocalRegistry):
            if fetch_from is not None:
                raise TimeFValidationError(
                    "fetch_from adds a registry to the reads of a local registry; a remote client reads from its "
                    "own registry"
                )
            return [registry], None
        source = open_registry(fetch_from) if isinstance(fetch_from, str) else fetch_from
        sources: _Sources = [registry]
        if source is not None:
            sources.append(source)
        if auto_build:
            sources.append(InstalledConnectors())
        return sources, registry

    def _reject_unhosted_access(self, dataset_id: str, version: str | None) -> None:
        """Check dataset access against the registry's requirements.

        Args:
            dataset_id: The dataset ID without a version suffix.
            version: The version string, or ``None`` for the latest.

        Raises:
            TimeNetAccessError: If the dataset or its lineage lacks open access and the registry requires it.
        """
        if not self._registry.requires_open_access:
            return
        try:
            manifest = self._registry.get_manifest(dataset_id, version)
            require_open_access(manifest.metadata, manifest.lineage)
        except TimeNetAccessError as error:
            error.hint = _unhosted_hint(dataset_id)
            raise

    def open_torch(
        self,
        dataset_id: str,
        version: str | None = None,
        *,
        auto_build: bool = True,
        fetch_from: str | BaseRegistry | None = None,
        load: LoadPlan = LoadPlan.ALL_EAGER,
    ) -> TimeFTorchDataset:
        """Open a dataset version as a read-only PyTorch ``Dataset`` that reads it a batch at a time.

        The view holds an open reader from :meth:`open`. Close the view with its ``close()``
        method, or use it in a ``with`` block. This method needs the ``torch`` extra
        (``pip install 'timenet[torch]'``). The code imports the torch view lazily, so base users do
        not need torch installed.

        Args:
            dataset_id: The dataset id.
            version: An exact version, ``"latest"``, or ``None``, as in :meth:`open`.
            auto_build: Build a release that only an installed connector provides. Set false to
                raise instead.
            fetch_from: A registry, or its URI, that a local client reads a release from if its own
                registry does not provide it.
            load: The plan that each batch loads with. A text-only plan, for example, skips the
                arrays of candidate and target Records.

        Returns:
            A :class:`~timenet.torch.TimeFTorchDataset` over a reader of the version, which reads
            and loads one batch of records at a time.
        """
        from timenet.torch import TimeFTorchDataset  # ruff: ignore[import-outside-top-level]

        return TimeFTorchDataset.from_reader(
            self.open(dataset_id, version, auto_build=auto_build, fetch_from=fetch_from), load=load
        )


def _unhosted_hint(dataset_id: str) -> str:
    """Tell a remote client how to get a dataset whose data its registry does not serve.

    Returns:
        The hint as a full sentence.
    """
    return f"TimeNet does not host its data, so build it locally with `timenet-build build {dataset_id}`."


def _missing_hint(*, auto_build: bool, fetch_from: str | BaseRegistry | None) -> str | None:
    """Name the keywords of :meth:`TimeNet.open` that can get a missing release.

    Returns:
        The hint as a full sentence, or ``None`` if the call already used both keywords.
    """
    ways = []
    if fetch_from is None:
        ways.append("read it from a registry that holds it with fetch_from=<registry>")
    if not auto_build:
        ways.append("build it from its connector with auto_build=True")
    if not ways:
        return None
    hint = " or ".join(ways)
    return f"{hint[0].upper()}{hint[1:]}."
