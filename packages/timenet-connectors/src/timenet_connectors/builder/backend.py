"""The build backend the SDK reaches through the ``timenet.builders`` entry point."""

from collections.abc import Mapping
from pathlib import Path

from timenet.manifest import PayloadSource
from timenet.registry import BaseRegistry
from timenet.types import DatasetMetadata
from timenet.types.references import DatasetRef
from timenet_connectors.builder.cli import build_one
from timenet_connectors.discovery import connector_dir, has_connector


class ConnectorBuilder:
    """Builds a dataset by running its connector in an environment built from its requirements."""

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
