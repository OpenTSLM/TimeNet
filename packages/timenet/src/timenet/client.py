"""The :class:`TimeNet` SDK: the single entry point for using TimeNet from code.

Wraps a :class:`~timenet.registry.BaseRegistry` (the catalog) and a local storage path (the download
cache), exposing ``list`` / ``get`` / ``search`` / ``download`` / ``load``. It never runs connector
code; producing datasets is the curation side.
"""

# The public API has a method named ``list``; deferred annotations keep ``list[str]`` type hints
# resolving to the builtin rather than the method.
from __future__ import annotations

from pathlib import Path
import shutil
from typing import TYPE_CHECKING
import uuid

from timenet.config import settings
from timenet.dataset import TimeFDataset
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.refs import split_ref
from timenet.registry import BaseRegistry, LocalRegistry, open_registry
from timenet.types import DatasetMetadata, Domain, License, Task


if TYPE_CHECKING:
    from timenet.torch import TimeFTorchDataset


# Defined at module scope, where `list` is the builtin (the class has a method named ``list`` that
# would otherwise shadow it in type annotations).
type _Metadatas = list[DatasetMetadata]
type _OrList[T] = T | list[T] | None


def _resolve_ref(dataset_id: str, version: str | None) -> tuple[str, str | None]:
    """Split an ``org/id@version`` ref and reconcile it with an explicit ``version``.

    Args:
        dataset_id: A dataset id, optionally suffixed with ``@<version>`` or ``@latest``.
        version: An explicit version, or ``None`` for the latest.

    Returns:
        The bare dataset id and the resolved version (``None`` = latest).

    Raises:
        TimeFValidationError: If a version is given both in the ref and as ``version``.
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
        """Open a client against a registry.

        Registry selection order: the ``registry`` argument, then ``$TIMENET_REGISTRY``, then the local
        default registry (``<TIMENET_HOME>/registry``).

        Args:
            registry: A registry instance, URL, ``file://`` URI, or local path.
            storage_path: Where downloads are cached (defaults to ``$TIMENET_STORAGE`` or
                ``<TIMENET_HOME>/storage``).
        """
        given_registry = registry if isinstance(registry, BaseRegistry) else None
        cfg = settings(
            registry=None if given_registry is not None or registry is None else str(registry),
            storage=storage_path,
        )
        if given_registry is not None:
            self._registry = given_registry
        elif cfg.registry is not None:
            self._registry = open_registry(cfg.registry)
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

    def search(
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
        """Search the registry. Mirrors :meth:`~timenet.registry.BaseRegistry.search`.

        Args:
            query: Free-text terms over name/description/tags.
            domain: Keep datasets sharing any of these domains.
            task: Keep datasets whose schema includes any of these task classes.
            license: Keep datasets with any of these licenses.
            time_series_spec: Keep datasets declaring all of these ``spec_type`` values.
            dataset_id: Keep only these ids.
            tag: Keep datasets declaring all of these tags.
            limit: Maximum number of results.

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
        """Fetch a dataset version's files into local storage and return its directory.

        Args:
            dataset_id: The dataset id.
            version: The version string, or ``None`` for the latest.
            force: Re-download even if an up-to-date copy already exists.

        Returns:
            The local ``<storage>/<dataset_id>/<version>/`` directory.
        """
        dataset_id, version = _resolve_ref(dataset_id, version)
        manifest = self._registry.get_manifest(dataset_id, version)
        resolved = str(manifest.metadata.dataset_version)
        target = self._storage / dataset_id / resolved
        if (target / "manifest.json").exists() and not force:
            return target

        files = manifest.files
        relpaths = [files.samples, files.annotations, files.time_series_index, *files.tasks, *files.time_series]
        # Fetch into a staging dir and swap it in atomically, so an interrupted (re-)download never
        # leaves a half-written copy in place of a good one — the live target is replaced only once
        # every file (manifest.json last) has landed.
        staging_parent = self._storage / dataset_id
        # A hard kill (SIGKILL/power loss) skips the finally below, so its staging dir lingers. Sweep
        # any stale <version>.tmp-* sibling before staging a fresh copy (mirrors the writer).
        if staging_parent.is_dir():
            for entry in staging_parent.glob(f"{resolved}.tmp-*"):
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
        staging = staging_parent / f"{resolved}.tmp-{uuid.uuid4().hex}"
        try:
            for relpath in relpaths:
                self._fetch(dataset_id, resolved, relpath, staging)
            self._fetch(dataset_id, resolved, "manifest.json", staging)  # commit marker last
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            staging.replace(target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return target

    def load(self, dataset_id: str, version: str | None = None) -> TimeFDataset:
        """Download if needed, then read the dataset into memory.

        Args:
            dataset_id: The dataset id.
            version: The version string, or ``None`` for the latest.

        Returns:
            The dataset with lazy per-series loaders backed by the local copy.
        """
        target = self.download(dataset_id, version)
        return TimeFReader(target).read()

    def load_torch(self, dataset_id: str, version: str | None = None) -> TimeFTorchDataset:
        """Download if needed and return the dataset as a read-only PyTorch ``Dataset``.

        Requires the ``torch`` extra (``pip install 'timenet[torch]'``); the torch view is imported
        lazily so base users don't need torch.

        Args:
            dataset_id: The dataset id.
            version: The version string, or ``None`` for the latest.

        Returns:
            A :class:`~timenet.torch.TimeFTorchDataset` over the loaded dataset.
        """
        from timenet.torch import TimeFTorchDataset

        return TimeFTorchDataset(self.load(dataset_id, version))

    def _fetch(self, dataset_id: str, version: str, relpath: str, target: Path) -> None:
        """Copy one file from the registry into the local ``target`` directory.

        Raises:
            TimeFFormatError: If ``relpath`` would write outside ``target`` (e.g. it contains ``..``).
        """
        destination = target / relpath
        if not destination.resolve().is_relative_to(target.resolve()):
            raise TimeFFormatError(f"manifest file path {relpath!r} escapes the download directory")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._registry.open_file(dataset_id, version, relpath) as source, destination.open("wb") as sink:
            shutil.copyfileobj(source, sink)
