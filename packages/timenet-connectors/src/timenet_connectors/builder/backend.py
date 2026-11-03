"""The build backend the SDK reaches through the ``timenet.builders`` entry point."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from timenet.builders import BuildOptions
from timenet.config import settings
from timenet.engine import run_pipeline
from timenet.errors import TimeNetDatasetNotFoundError, TimeNetVersionUnavailableError
from timenet.manifest import PayloadSource
from timenet.progress import BytesMoved, ProgressCallback, ShardWritten
from timenet.registry import BaseRegistry, LocalRegistry
from timenet.types import DatasetMetadata
from timenet.types.references import DatasetRef
from timenet.writer.progress import ProgressStage, WriteProgressEvent
from timenet_connectors.builder.cli import build_one
from timenet_connectors.builder.env import run_isolated_build
from timenet_connectors.discovery import connector_dir, has_connector, known_ids, resolve
from timenet_connectors.download import DownloadProgress, progress_sink


class ConnectorBuilder:
    """Build a dataset with its connector, in an environment built from the connector's requirements."""

    def dataset_ids(self) -> list[str]:  # ruff: ignore[no-self-use] (BuilderBackend protocol method)
        """List this package's dataset IDs without importing connectors.

        Returns:
            The sorted dataset IDs. A card that cannot be read is skipped with a warning.
        """
        return known_ids()

    def knows(self, dataset_id: str) -> bool:  # ruff: ignore[no-self-use] (BuilderBackend protocol method)
        """Report whether a connector package exists for the id, without importing it.

        Args:
            dataset_id: The dataset id.

        Returns:
            Whether a connector exists.
        """
        return has_connector(dataset_id)

    def declared_version(self, dataset_id: str) -> str | None:
        """Read the version the connector's card declares, without importing or building it.

        Args:
            dataset_id: The dataset id.

        Returns:
            The declared version string, or ``None`` if no connector exists for the dataset.

        Raises:
            TimeNetInvalidCardError: If the card of the connector cannot be read or is not a mapping.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by card()
        card = self.card(dataset_id)
        return None if card is None else str(card.dataset_version)

    def card(self, dataset_id: str) -> DatasetMetadata | None:  # ruff: ignore[no-self-use] (protocol)
        """Read the connector's ``dataset.yaml`` card, without importing or building the connector.

        Args:
            dataset_id: The dataset id.

        Returns:
            The validated card, or ``None`` if no connector exists for the dataset.

        Raises:
            TimeNetInvalidCardError: If the card cannot be read or is not a mapping.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by DatasetMetadata.from_yaml
        try:
            directory = connector_dir(dataset_id)
        except LookupError:
            return None
        return DatasetMetadata.from_yaml(directory / "dataset.yaml")

    def build(  # ruff: ignore[too-many-arguments, no-self-use] (protocol)
        self,
        dataset_id: str,
        root: Path,
        *,
        force: bool = False,
        build_parents: bool = False,
        fetch_from: str | BaseRegistry | None = None,
        pinned_by: Mapping[DatasetRef, PayloadSource] | None = None,
    ) -> Path:
        """Build a dataset in an isolated environment.

        Set ``TIMENET_ISOLATION=off`` to run in the current interpreter.

        Args:
            dataset_id: The dataset id.
            root: The output registry directory.
            force: Rebuild an existing version.
            build_parents: Build missing parent releases from their installed connectors.
            fetch_from: The registry, or its URI, to copy missing parent releases from.
            pinned_by: Releases that read this dataset's files, with file pins the build must match.

        Returns:
            The committed version directory.
        """
        return Path(
            build_one(
                dataset_id,
                root,
                fetch_from=fetch_from,
                build_parents=build_parents,
                force=force,
                pinned_by=pinned_by,
            )
        )

    def build_prepared(
        self,
        release: DatasetRef,
        store: LocalRegistry,
        *,
        force: bool,
        pinned_by: Mapping[DatasetRef, PayloadSource],
        options: BuildOptions,
    ) -> None:
        """Build a release in a local registry.

        The registry must already contain all parent releases and their payload sources.
        ``options.isolation``, else ``TIMENET_ISOLATION``, set to ``"off"`` builds in this interpreter.

        Args:
            release: The exact dataset ID and version to build.
            store: The local registry to build into.
            force: Rebuild even if ``store`` already contains the release.
            pinned_by: Releases that depend on this release, with file pins that the build must preserve.
            options: Cache and output settings, with a progress callback for builds in this interpreter.

        Raises:
            TimeNetDatasetNotFoundError: If no connector exists for the dataset.
            TimeNetVersionUnavailableError: If the requested version differs from the connector's card.
            TimeNetBuildError: If the isolated build fails.
        """  # ruff: ignore[docstring-extraneous-exception] - TimeNetBuildError is raised by run_isolated_build
        card = self.card(release.dataset_id)
        if card is None:
            raise TimeNetDatasetNotFoundError(f"no connector builds {release.dataset_id!r}")
        if card.dataset_version != release.version:
            raise TimeNetVersionUnavailableError(release, card.dataset_version)
        if settings(isolation=options.isolation).isolation == "on":
            # The parent reads only the card: the connector and its dependencies exist only in the child.
            run_isolated_build(
                release,
                store.root,
                force=force,
                keep_cache=options.keep_cache,
                quiet=options.quiet,
                pinned_by=pinned_by,
            )
            return
        _build_in_process(release, store.root, force=force, pinned_by=pinned_by, options=options)


def _build_in_process(
    release: DatasetRef,
    root: Path,
    *,
    force: bool,
    pinned_by: Mapping[DatasetRef, PayloadSource],
    options: BuildOptions,
) -> None:
    """Build a release in the current interpreter.

    Args:
        release: The dataset ID and version declared in the connector's card.
        root: The local registry directory to build into.
        force: Rebuild even if ``root`` already contains the release.
        pinned_by: Releases that depend on this release, with file pins that the build must preserve.
        options: Cache settings and a progress callback.
    """
    connector = resolve(release.dataset_id)()
    relay = None if options.progress is None else _ProgressRelay(release, options.progress)
    # A connector reports its downloads through the ambient progress sink. Without a sink, a
    # downloader skips the work that it does only to report progress.
    with progress_sink(None if relay is None else relay.download):
        run_pipeline(
            connector,
            root,
            progress_cb=None if relay is None else relay.write,
            force=force,
            keep_cache=options.keep_cache,
            pinned_by=pinned_by,
        )


@dataclass(frozen=True)
class _ProgressRelay:
    """Convert download and write updates into build progress events."""

    release: DatasetRef
    report: ProgressCallback

    def download(self, event: DownloadProgress) -> None:
        """Report downloaded bytes as a :class:`~timenet.progress.BytesMoved` event.

        Args:
            event: The download progress update.
        """
        self.report(BytesMoved(event.url, event.downloaded, event.total))

    def write(self, event: WriteProgressEvent) -> None:
        """Report a completed shard as a :class:`~timenet.progress.ShardWritten` event.

        Args:
            event: The writer progress update.
        """
        if event.stage is ProgressStage.SHARD_FINALIZED:
            self.report(ShardWritten(self.release, event.completed))
