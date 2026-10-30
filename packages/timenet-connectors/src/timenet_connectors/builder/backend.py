"""The build backend the SDK reaches through the ``timenet.builders`` entry point."""

from collections.abc import Mapping
from pathlib import Path

from timenet.errors import TimeNetInvalidCardError
from timenet.manifest import PayloadSource
from timenet.registry import BaseRegistry
from timenet.types import DatasetMetadata
from timenet.types.references import DatasetRef
from timenet_connectors.builder.cli import build_one
from timenet_connectors.discovery import has_connector, read_card


class ConnectorBuilder:
    """Builds a dataset by running its connector in an environment built from its requirements."""

    def knows(self, dataset_id: str) -> bool:  # noqa: PLR6301 (BuilderBackend protocol method)
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
            The declared version string, or ``None`` if no readable card exists.
        """
        card = self.card(dataset_id)
        return None if card is None else str(card.dataset_version)

    def card(self, dataset_id: str) -> DatasetMetadata | None:  # noqa: PLR6301 (protocol)
        """Read the connector's ``dataset.yaml`` card, without importing or building the connector.

        Args:
            dataset_id: The dataset id.

        Returns:
            The validated card, or ``None`` if no readable card exists.
        """
        try:
            return read_card(dataset_id)
        except (LookupError, TimeNetInvalidCardError):
            return None

    def build(  # noqa: PLR0913, PLR6301 (protocol)
        self,
        dataset_id: str,
        root: Path,
        *,
        force: bool = False,
        values_backend: str | None = None,
        build_parents: bool = False,
        fetch_from: str | BaseRegistry | None = None,
        pinned_by: Mapping[DatasetRef, PayloadSource] | None = None,
    ) -> Path:
        """Build the dataset, in an isolated environment unless isolation is turned off.

        ``TIMENET_ISOLATION=off`` builds in this interpreter instead, as ``--no-isolation`` does on
        the CLI. That is also the recursion guard: an isolated child carries the setting, so a
        connector that loads another dataset from inside one does not re-exec uv forever. The build
        runs through :func:`~timenet_connectors.builder.cli.build_one`, which prepares missing parent
        releases first, each the same way as this build.

        Args:
            dataset_id: The dataset id.
            root: The output registry directory.
            force: Rebuild even if the version is already built.
            values_backend: Storage backend for the values plane. Defaults to the connector's value.
            build_parents: Build missing parent releases from their installed connectors.
            fetch_from: The registry, or its URI, to copy missing parent releases from.
            pinned_by: The releases that already read files of this release, with their pins. The
                build publishes the release only if it reproduces each pinned file.

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
                values_backend=values_backend,
                pinned_by=pinned_by,
            )
        )
