"""The :class:`TimeNet` class is the SDK's single entry point for using TimeNet from code.

This class wraps a :class:`~timenet.registry.BaseRegistry` (the catalog) and a local storage path
(the download cache). It exposes these methods: ``list``, ``get``, ``search``, ``download``,
``open``, ``load``, and ``open_torch``. Against a local registry, ``open`` and ``load`` build a
dataset the registry does not have when an installed package registers a connector for its id;
missing parents are built only when ``build_parents`` is enabled.
"""

# The public API has a method named ``list``. Deferred annotations keep the type hint
# ``list[str]`` resolved to the builtin type, not to the method.
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, TypeVar

from timenet.builders import build_exact, find_builder
from timenet.config import settings
from timenet.dataset import TimeFDataset
from timenet.errors import TimeFValidationError, TimeNetAccessError, TimeNetDatasetNotFoundError
from timenet.manifest import Manifest
from timenet.reader import LoadPlan, TimeFReader
from timenet.registry import BaseRegistry, LocalRegistry, open_registry
from timenet.registry._paths import release_dir
from timenet.registry.base import ProgressCallback
from timenet.registry.dependencies import prepare_local_copy
from timenet.types import Access, DatasetMetadata, Domain, License, Task
from timenet.types.references import DatasetRef, is_latest, split_ref


if TYPE_CHECKING:
    from timenet.torch import TimeFTorchDataset


# This code is at module scope, where `list` is the builtin type. The class has a method
# named ``list``. This placement stops that method name from shadowing the builtin in type
# annotations.
T = TypeVar("T")
_Metadatas: TypeAlias = list[DatasetMetadata]
_OrList: TypeAlias = T | list[T] | None


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


class TimeNet:
    """Browse a registry and fetch datasets to local storage."""

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

    def search(  # noqa: PLR0913
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
        fetch_parents: str | None = None,
        build_parents: bool = False,
    ) -> TimeFReader:
        """Prepare a dataset version's payload sources, then open it for lazy reads.

        Opening prepares the version and reads its manifest, but neither its catalog nor any value.
        The reader's reads leave stored values behind value handles unless their load plan names
        them. Close the reader after use, or use it as a context manager. Remote backends open
        payload sources from their manifests upfront and read payload ranges on demand, reusing
        complete cached releases. Only the requested catalog is downloaded, on the first structural
        read. Local registries require payload sources to be present unless fetching or building is
        enabled. Build-only parents are prepared when a connector runs.

        Against a local registry, a dataset the registry does not have is built first, if some
        installed package registers a connector for its id. Set ``auto_build`` false to fail fast
        instead of starting a download and a build. A remote registry raises as before.

        Args:
            dataset_id: The dataset id.
            version: The version string, or ``None`` for the latest.
            auto_build: Build a missing local dataset from its connector. Set false to raise instead.
            fetch_parents: Registry URI to fetch missing payload sources or declared parents from.
            build_parents: Build missing payload sources, or parents needed when building a release.

        Returns:
            A reader for the version.

        Raises:
            TimeNetAccessError: If the dataset is not open access and the registry is not local.
            TimeNetDatasetNotFoundError: If the version is absent and nothing builds it, or the connector
                declares a different version than the one asked for.
            TimeNetBuildError: If the build runs but fails.
            TimeFValidationError: If a remote read supplies a second registry through ``fetch_parents``.
        """  # noqa: DOC502 (TimeNetAccessError comes from _reject_unhosted_access, TimeNetBuildError from the builder)
        dataset_id, version = _resolve_ref(dataset_id, version)
        self._reject_unhosted_access(dataset_id, version)
        try:
            manifest = self._registry.get_manifest(dataset_id, version)
        except TimeNetDatasetNotFoundError as miss:
            if not auto_build or not isinstance(self._registry, LocalRegistry):
                raise
            builder = find_builder(dataset_id)
            if builder is None:
                raise
            # A connector produces one declared version. Reject a pin it cannot satisfy before the
            # build runs, so a wrong pin fails fast instead of after a full build. Sentinels are
            # not pins, so a fresh build satisfies them.
            if not is_latest(version):
                declared = builder.declared_version(dataset_id)
                if declared is not None and version != declared:
                    raise TimeNetDatasetNotFoundError(
                        f"the connector for {dataset_id!r} builds version {declared}, not the requested {version}"
                    ) from miss
            builder.build(
                dataset_id,
                self._registry.root,
                build_parents=build_parents,
                fetch_parents=fetch_parents,
            )
            manifest = self._registry.get_manifest(dataset_id, version)
        root = self._registry.local_root
        reference = DatasetRef(dataset_id=dataset_id, version=manifest.metadata.dataset_version)
        fetch_from = None if isinstance(self._registry, LocalRegistry) else self._registry
        if fetch_parents is not None:
            if not isinstance(self._registry, LocalRegistry):
                raise TimeFValidationError(
                    "fetch_parents selects a source for a local registry; remote reads use their configured registry"
                )
            fetch_from = open_registry(fetch_parents)
        if not isinstance(self._registry, LocalRegistry) and not build_parents:
            return TimeFReader(self._registry.open_version(dataset_id, str(reference.version)))
        handle = prepare_local_copy(
            root,
            reference,
            manifest=manifest,
            fetch_from=fetch_from,
            build=(lambda ref: build_exact(ref, root, fetch_parents=fetch_parents)) if build_parents else None,
        )
        return TimeFReader(handle)

    def load(
        self,
        dataset_id: str,
        version: str | None = None,
        *,
        auto_build: bool = True,
        fetch_parents: str | None = None,
        build_parents: bool = False,
    ) -> TimeFDataset:
        """Read a whole dataset version into memory, with every stored value loaded.

        The dataset needs no storage afterwards: its reader is closed before it is returned. For a
        large dataset, use :meth:`open` and read what you need instead.

        Args:
            dataset_id: The dataset id.
            version: The version string, or ``None`` for the latest.
            auto_build: Build a missing local dataset from its connector. Set false to raise instead.
            fetch_parents: Registry URI to fetch missing payload sources or declared parents from.
            build_parents: Build missing payload sources, or parents needed when building a release.

        Returns:
            The dataset with every signal array, time offset, and text value loaded.

        Raises:
            TimeNetAccessError: If the dataset is not open access and the registry is not local.
            TimeNetDatasetNotFoundError: If the version is absent and nothing builds it, or the connector
                declares a different version than the one asked for.
            TimeNetBuildError: If the build runs but fails.
            TimeFValidationError: If a remote read supplies a second registry through ``fetch_parents``.
        """  # noqa: DOC502 (all come from open())
        with self.open(
            dataset_id, version, auto_build=auto_build, fetch_parents=fetch_parents, build_parents=build_parents
        ) as reader:
            return reader.read(load=LoadPlan.ALL_EAGER)

    def _reject_unhosted_access(self, dataset_id: str, version: str | None) -> None:
        """Raise for a non-open dataset read from a registry that does not host its data.

        Credentialed and restricted datasets cannot be redistributed, so TimeNet never serves their
        bytes. Such a dataset is build-your-own: a local registry holds only what the user built, so a
        local read is fine, but a hosted registry can only point the user at where to obtain access.

        Args:
            dataset_id: The bare dataset id.
            version: The resolved version, or ``None`` for the latest.

        Raises:
            TimeNetAccessError: If the dataset is not open and the registry is not a local one.
        """
        if isinstance(self._registry, LocalRegistry):
            return
        metadata = self._registry.get_manifest(dataset_id, version).metadata
        if metadata.access is Access.OPEN:
            return
        raise TimeNetAccessError(
            f"{dataset_id!r} is {metadata.access.value}: TimeNet does not host its data. Get access at "
            f"{metadata.access_url}, then build it locally with `timenet-build build {dataset_id}`."
        )

    def open_torch(
        self,
        dataset_id: str,
        version: str | None = None,
        *,
        auto_build: bool = True,
        fetch_parents: str | None = None,
        build_parents: bool = False,
    ) -> TimeFTorchDataset:
        """Open a dataset version as a read-only PyTorch ``Dataset`` that reads it a batch at a time.

        The view holds an open reader from :meth:`open`. Close the view with its ``close()``
        method, or use it in a ``with`` block. This method needs the ``torch`` extra
        (``pip install 'timenet[torch]'``). The code imports the torch view lazily, so base users do
        not need torch installed.

        Args:
            dataset_id: The dataset id.
            version: The version string, or ``None`` for the latest.
            auto_build: Build a missing local dataset from its connector. Set false to raise instead.

        Returns:
            A :class:`~timenet.torch.TimeFTorchDataset` over a reader of the version, which reads
            and loads one batch of records at a time.
        """
        from timenet.torch import TimeFTorchDataset  # noqa: PLC0415

        return TimeFTorchDataset.from_reader(
            self.open(
                dataset_id, version, auto_build=auto_build, fetch_parents=fetch_parents, build_parents=build_parents
            )
        )
