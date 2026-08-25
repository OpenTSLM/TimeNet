"""The :class:`TimeNet` class is the SDK's single entry point for using TimeNet from code.

This class wraps a :class:`~timenet.registry.BaseRegistry` (the catalog) and a local storage path
(the download cache). It exposes these methods: ``list``, ``get``, ``search``, ``download``, and
``load``. This class never runs connector code. The build side produces datasets.
"""

# The public API has a method named ``list``. Deferred annotations keep the type hint
# ``list[str]`` resolved to the builtin type, not to the method.
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, TypeVar

from timenet.config import settings
from timenet.dataset import TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.refs import split_ref
from timenet.registry import BaseRegistry, LocalRegistry, RemoteRegistry, open_registry
from timenet.types import DatasetMetadata, Domain, License, Task


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
        ``$TIMENET_REGISTRY``, then the local default registry (``<TIMENET_HOME>/registry``).

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
            self._registry = LocalRegistry(cfg.registry_path)
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

    def download(self, dataset_id: str, version: str | None = None, *, force: bool = False) -> Path:
        """Fetch a dataset version's files into local storage. Return its directory.

        Args:
            dataset_id: The dataset id.
            version: The version string, or ``None`` for the latest.
            force: Download again, even if an up-to-date copy already exists.

        Returns:
            The local ``<storage>/<dataset_id>/<version>/`` directory.
        """
        dataset_id, version = _resolve_ref(dataset_id, version)
        manifest = self._registry.get_manifest(dataset_id, version)
        resolved = str(manifest.metadata.dataset_version)
        target = self._storage / dataset_id / resolved
        # The registry owns the fetch: the base implementation stages each file and swaps atomically,
        # and a remote registry overrides it to resolve and stream every file in parallel.
        self._registry.download_version(dataset_id, resolved, target, force=force, manifest=manifest)
        return target

    def load(self, dataset_id: str, version: str | None = None, *, download_mode: str | None = None) -> TimeFDataset:
        """Read the dataset into memory through the registry's storage handle.

        This method does not download the whole dataset. The reader loads each series only
        when code uses it. It loads data from the registry through the handle that
        :meth:`~timenet.registry.BaseRegistry.open_version` returns. To get an on-disk cache,
        use :meth:`download`.

        Args:
            dataset_id: The dataset id.
            version: The version string, or ``None`` for the latest.
            download_mode: For a remote registry, ``"full"`` or ``"on_demand"`` to override the
                default download mode; ignored for local/S3 registries.

        Returns:
            The dataset with lazy, per-series loaders that use the registry handle.
        """
        dataset_id, version = _resolve_ref(dataset_id, version)
        if download_mode is not None and isinstance(self._registry, RemoteRegistry):
            handle = self._registry.open_version(dataset_id, version, mode=download_mode)
        else:
            handle = self._registry.open_version(dataset_id, version)
        return TimeFReader(handle).read()

    def load_torch(self, dataset_id: str, version: str | None = None) -> TimeFTorchDataset:
        """Download the dataset if needed, then return it as a read-only PyTorch ``Dataset``.

        This method needs the ``torch`` extra (``pip install 'timenet[torch]'``). The code
        imports the torch view lazily, so base users do not need torch installed.

        Args:
            dataset_id: The dataset id.
            version: The version string, or ``None`` for the latest.

        Returns:
            A :class:`~timenet.torch.TimeFTorchDataset` over the loaded dataset.
        """
        from timenet.torch import TimeFTorchDataset  # noqa: PLC0415

        # Force the full download: a DataLoader pickles the handle to its workers, and the on-demand
        # handle wraps a live httpx client that cannot pickle; only the local-filesystem handle survives.
        return TimeFTorchDataset(self.load(dataset_id, version, download_mode="full"))
