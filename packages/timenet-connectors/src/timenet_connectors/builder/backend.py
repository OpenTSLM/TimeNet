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
        """Build exactly this release into the store.

        The store must already hold each parent of the release and their payload
        sources. This method does not plan. The build runs in a separate environment
        with the requirements of the connector, unless ``TIMENET_ISOLATION=off``,
        which builds in this interpreter.

        Args:
            release: The exact dataset ID and version to build.
            store: The local registry to build into.
            force: Build the release again, even if ``store`` already holds it.
            pinned_by: The releases that already read files of this release, with their pins. The
                build publishes the release only if it reproduces each pinned file.
            options: The options of the build. An in-process build sends a
                :class:`~timenet.progress.BytesMoved` event for the downloads and a
                :class:`~timenet.progress.ShardWritten` event for each finished shard to
                ``options.progress``. An isolated child prints its own progress, unless
                ``options.quiet`` is set.

        Raises:
            TimeNetDatasetNotFoundError: If no connector exists for the dataset.
            TimeNetVersionUnavailableError: If the card of the connector declares another version
                than ``release``. Nothing runs then. It is also a ``TimeFValidationError``.
            TimeNetBuildError: If the isolated child fails.
        """  # ruff: ignore[docstring-extraneous-exception] - TimeNetBuildError is raised by run_isolated_build
        card = self.card(release.dataset_id)
        if card is None:
            raise TimeNetDatasetNotFoundError(f"no connector builds {release.dataset_id!r}")
        if card.dataset_version != release.version:
            raise TimeNetVersionUnavailableError(release, card.dataset_version)
        if settings().isolation == "on":
            # Do not call resolve() here. It imports the connector, which needs the dependencies that
            # only the environment of the child has.
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
    """Build the release in this interpreter, with the ``values_backend`` of its connector.

    Args:
        release: The release to build. Its version is the version that the card declares.
        root: The directory of the local registry to build into.
        force: Build the release again, even if ``root`` already holds it.
        pinned_by: The releases that already read files of this release, with their pins.
        options: The options of the build.
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
    """Sends the download and writer progress of an in-process build as progress events of the build."""

    release: DatasetRef
    report: ProgressCallback

    def download(self, event: DownloadProgress) -> None:
        """Send the progress of a download as a :class:`~timenet.progress.BytesMoved` event.

        Args:
            event: The progress of the download.
        """
        self.report(BytesMoved(event.url, event.downloaded, event.total))

    def write(self, event: WriteProgressEvent) -> None:
        """Send each finished shard as a :class:`~timenet.progress.ShardWritten` event.

        Args:
            event: The progress of the writer.
        """
        if event.stage is ProgressStage.SHARD_FINALIZED:
            self.report(ShardWritten(self.release, event.completed))
