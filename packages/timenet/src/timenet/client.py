"""Find, inspect, download, and read TimeF datasets.

:class:`TimeNet` checks sources in order and uses :func:`default_sources` by default.
"""

# The public API has a method named ``list``. Deferred annotations keep the type hint
# ``list[str]`` resolved to the builtin type, not to the method.
from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
import os
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, TypeVar

from timenet._hints import Facts, Operation, RequestedRelease, explain
from timenet._warnings import warn_at_caller
from timenet.builders import BuildOptions
from timenet.config import TimeNetSettings, settings
from timenet.coordinator.build import run_plan
from timenet.coordinator.plan import Lookups, plan_download
from timenet.coordinator.read import raise_not_found, read_release
from timenet.dataset import TimeFDataset
from timenet.errors import CatalogWarning, TimeFValidationError, TimeNetNoCatalogError, TimeNetNotBuiltError
from timenet.manifest import Manifest
from timenet.progress import ProgressCallback
from timenet.reader import LoadPlan, TimeFReader
from timenet.registry import BaseRegistry, LocalRegistry, open_registry
from timenet.registry._paths import release_dir
from timenet.sources import FoundInConnector, InstalledConnectors, NotFound, NotServed, ReleaseSource
from timenet.types import DatasetFilter, DatasetMetadata, Domain, License, ListedRelease, Task
from timenet.types.references import DatasetRef, split_ref


if TYPE_CHECKING:
    from timenet.torch import TimeFTorchDataset


# This code is at module scope, where `list` is the builtin type. The class has a method
# named ``list``. This placement stops that method name from shadowing the builtin in type
# annotations.
T = TypeVar("T")
_Metadatas: TypeAlias = list[DatasetMetadata]
_Listed: TypeAlias = list[ListedRelease]
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
    return [LocalRegistry(cfg.local_registry_dir), *_sources_after_local_registry(cfg)]


def _sources_after_local_registry(cfg: TimeNetSettings, remote: BaseRegistry | None = None) -> list[ReleaseSource]:
    """Return the remote registry and the optional installed connectors of the settings, in lookup order.

    Args:
        cfg: The settings.
        remote: The remote registry, already open, or ``None`` for the remote registry of the settings.

    Returns:
        The sources that follow the local registry.

    Raises:
        TimeNetRegistryError: If the remote registry cannot be opened.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by open_registry)
    if remote is None:
        remote = open_registry(cfg.remote_registry, label="TIMENET_REMOTE_REGISTRY")
    return [remote, *([InstalledConnectors()] if cfg.auto_build else [])]


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
        self._local_registry = next(
            (source for source in self._sources if isinstance(source, LocalRegistry) and not source.read_only), None
        )
        if self._local_registry is None and any(isinstance(source, InstalledConnectors) for source in self._sources):
            raise TimeFValidationError(
                "InstalledConnectors() needs a writable local registry to build into. "
                "Add a LocalRegistry without read_only=True to the sources"
            )

    def list(self) -> _Metadatas:
        """List each dataset at its highest version across all sources, including connector cards.

        A listed version can be newer than the one that :meth:`open` reads, which skips a version that a registry
        does not serve and, without a version, takes the first source that serves the dataset.
        A source without a catalog is left out with a :class:`~timenet.errors.CatalogWarning`. Source errors stop
        the call.

        Returns:
            One :class:`~timenet.types.DatasetMetadata` per dataset, sorted by dataset ID.

        Raises:
            TimeNetNoCatalogError: If no source has a catalog.
            TimeNetInvalidCardError: If an installed backend cannot list its dataset IDs.
        """  # ruff: ignore[docstring-extraneous-exception] (raised by the backend)
        with self._explain(Operation.READ):
            return [release.card for release in self._listed()]

    def _listed(self) -> _Listed:
        """Return each dataset at its highest version across the sources, sorted by dataset ID.

        For equal versions, the earlier source keeps the row. A connector card has no registry. A source without a
        catalog is left out with a :class:`~timenet.errors.CatalogWarning`.

        Returns:
            One listed release per dataset.

        Raises:
            TimeNetNoCatalogError: If no source has a catalog.
        """
        if not any(source.has_catalog for source in self._sources):
            names = ", ".join(str(source) for source in self._sources)
            raise TimeNetNoCatalogError(f"none of the sources has a catalog: {names}")
        kept: dict[str, ListedRelease] = {}
        for source in self._sources:
            if not source.has_catalog:
                warn_at_caller(CatalogWarning(f"{source} has no catalog, so the list leaves it out."))
                continue
            registry = source if isinstance(source, BaseRegistry) else None
            for card in source.list_datasets():
                held = kept.get(card.dataset_id)
                if held is None or card.dataset_version > held.card.dataset_version:
                    kept[card.dataset_id] = ListedRelease(card, registry)
        return [kept[dataset_id] for dataset_id in sorted(kept)]

    def get_manifest(self, dataset_id: str, version: str | None = None) -> Manifest:
        """Return the manifest of the release that :meth:`open` reads, without a copy or a build.

        For a release that a registry lists but does not serve, return its listed manifest with access links.

        Args:
            dataset_id: The dataset ID, optionally ending with ``@<version>`` or ``@latest``.
            version: An exact version, ``"latest"``, or ``None``, as in :meth:`open`.

        Returns:
            The release's manifest.

        Raises:
            TimeNetNotBuiltError: If the release needs a build to create its manifest.
            TimeNetDatasetNotFoundError: If no source provides the release.
            TimeNetVersionUnavailableError: If the version is unavailable and a connector declares another version.
            TimeNetAccessError: If a registry only lists the release and provides no manifest.
            TimeNetInvalidCardError: If the connector's card is unreadable or invalid.
            TimeFValidationError: If ``dataset_id`` and ``version`` both give a version.
        """  # ruff: ignore[docstring-extraneous-exception] (raised by the resolution of the release)
        dataset_id, version = _resolve_ref(dataset_id, version)
        with self._explain(Operation.READ, dataset_id, version):
            found = self._manifest_or_connector(dataset_id, version)
            if isinstance(found, FoundInConnector):
                raise TimeNetNotBuiltError(found.builder.release, found.builder.card)
        return found

    def get_card(self, dataset_id: str, version: str | None = None) -> DatasetMetadata:
        """Return the card of the release that :meth:`open` reads, without a copy or a build.

        The card comes from the manifest or, for an unbuilt release, from the installed connector.

        Args:
            dataset_id: The dataset ID, optionally ending with ``@<version>`` or ``@latest``.
            version: An exact version, ``"latest"``, or ``None``, as in :meth:`open`.

        Returns:
            The release's metadata card.

        Raises:
            TimeNetDatasetNotFoundError: If no source provides the release.
            TimeNetVersionUnavailableError: If the version is unavailable and a connector declares another version.
            TimeNetAccessError: If a registry only lists the release and provides no manifest.
            TimeNetInvalidCardError: If the connector's card is unreadable or invalid.
            TimeFValidationError: If ``dataset_id`` and ``version`` both give a version.
        """  # ruff: ignore[docstring-extraneous-exception] (raised by the resolution of the release)
        dataset_id, version = _resolve_ref(dataset_id, version)
        with self._explain(Operation.READ, dataset_id, version):
            found = self._manifest_or_connector(dataset_id, version)
            return found.builder.card if isinstance(found, FoundInConnector) else found.metadata

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
        """Search the rows of :meth:`list`: each dataset at its highest version across all sources.

        So the search checks only the highest version of each dataset, and an older version that matches is not found.
        A ``task`` or ``time_series_spec`` filter reads the manifest of a row only if its card matches, and the search
        stops at ``limit`` matches. Connector cards lack schemas, so they never match these filters.
        A source without a catalog is left out with a :class:`~timenet.errors.CatalogWarning`. Source errors stop
        the call.

        Args:
            query: Match any term in the name, description, or tags, ignoring case.
            domain: Match any of these domains.
            task: Match any of these task classes in the schema.
            license: Match any of these licenses.
            time_series_spec: Match all of these ``spec_type`` values.
            dataset_id: Match any of these IDs.
            tag: Match all of these tags.
            limit: The maximum number of results.

        Returns:
            The matching dataset metadata, sorted by dataset ID.

        Raises:
            TimeFValidationError: If ``limit`` is negative.
            TimeNetNoCatalogError: If no source has a catalog.
            TimeNetInvalidCardError: If an installed backend cannot list its dataset IDs.
        """  # ruff: ignore[docstring-extraneous-exception] (raised by the backend)
        wanted = DatasetFilter.of(
            query=query,
            domain=domain,
            task=task,
            license=license,
            time_series_spec=time_series_spec,
            dataset_id=dataset_id,
            tag=tag,
            limit=limit,
        )
        with self._explain(Operation.READ):
            return wanted.select(self._listed())

    def download(
        self,
        dataset_id: str,
        version: str | None = None,
        *,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> Path:
        """Copy a release and its payload sources into the local registry.

        The release resolves as in :meth:`open`, without a build. Missing or damaged releases come from the first
        other registry that provides them, with a provenance record that names it.

        Args:
            dataset_id: The dataset ID.
            version: An exact version, ``"latest"``, or ``None``, as in :meth:`open`.
            force: Copy the release again, even if the local registry holds a complete copy.
            progress: Receives step events and byte counts for each copied release.

        Returns:
            The release directory in the local registry.

        Raises:
            TimeFValidationError: If the sources have no writable local registry.
            TimeNetAccessError: If a required release is only listed or lacks open access in a remote registry.
            TimeNetNotBuiltError: If only installed connectors provide the release.
            TimeNetDatasetNotFoundError: If no registry other than the local registry provides the release.
            TimeNetVersionUnavailableError: If the version is missing and its connector builds another version.
            TimeNetMissingParentError: If no registry provides a missing payload source.
            TimeNetVersionConflictError: If the local registry holds the same release with different files.
            TimeNetFileError: If a stored copy cannot be repaired or a copied file is damaged.
            TimeFFormatError: If a payload source breaks a pin or the payload sources form a cycle.
        """  # ruff: ignore[docstring-extraneous-exception] (raised by the plan and the copies)
        dataset_id, version = _resolve_ref(dataset_id, version)
        local_registry = self._local_registry
        if local_registry is None:
            # Without a local registry, a download has no place to go.
            raise TimeFValidationError(
                "download() needs a writable local registry to copy into. "
                "Add a LocalRegistry without read_only=True to the sources"
            )
        with self._explain(Operation.DOWNLOAD, dataset_id, version):
            lookups = Lookups()
            answer = lookups.resolve(self._sources, dataset_id, version)
            if isinstance(answer, NotServed):
                # A registry lists the release, but no source serves its files.
                raise answer.access_error(dataset_id)
            if isinstance(answer, NotFound):
                # No source provides the release.
                raise_not_found(dataset_id, version, self._sources)
            release = DatasetRef(dataset_id=dataset_id, version=answer.version)
            plan = plan_download(release, local_registry, sources=self._sources, force=force, lookups=lookups)
            run_plan(plan, local_registry, options=BuildOptions(progress=progress))
        return release_dir(local_registry.root, release)

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
        with self._explain(Operation.READ, dataset_id, version):
            return TimeFReader(
                read_release(dataset_id, version, sources=self._sources, target_registry=self._local_registry)
            )

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

    def _explain(
        self, request: Operation, dataset_id: str | None = None, version: str | None = None
    ) -> AbstractContextManager[None]:
        """Return a context that gives a TimeNet error raised in it a hint in the current wording."""
        wanted = None if dataset_id is None else RequestedRelease(dataset_id, version)
        return explain(Facts(self._sources, self._local_registry, request=request, wanted=wanted))

    def _manifest_or_connector(self, dataset_id: str, version: str | None) -> Manifest | FoundInConnector:
        """Resolve a release to its manifest, or to the answer of the connector that provides it.

        Args:
            dataset_id: The dataset ID without a version suffix.
            version: An exact version, ``"latest"``, or ``None``, as in :meth:`open`.

        Returns:
            The manifest, or the connector answer if only an installed connector provides the release.

        Raises:
            TimeNetDatasetNotFoundError: If no source provides the release.
            TimeNetAccessError: If a registry only lists the release and provides no manifest.
        """  # ruff: ignore[docstring-extraneous-exception] (raised by raise_not_found)
        lookups = Lookups()
        answer = lookups.resolve(self._sources, dataset_id, version)
        if isinstance(answer, NotFound):
            raise_not_found(dataset_id, version, self._sources)
        if isinstance(answer, NotServed):
            if answer.manifest is None:
                raise answer.access_error(dataset_id)
            return answer.manifest
        if isinstance(answer, FoundInConnector):
            return answer
        manifest = lookups.manifest(answer.registry, DatasetRef(dataset_id=dataset_id, version=answer.version))
        if manifest is None:
            # The registry no longer holds the release that it listed.
            raise_not_found(dataset_id, version, self._sources)
        return manifest

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
