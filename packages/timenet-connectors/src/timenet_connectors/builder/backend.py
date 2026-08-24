"""The curation backend the SDK reaches through the ``timenet.curators`` entry point."""

from pathlib import Path

from timenet.config import settings
from timenet.engine import run_pipeline
from timenet_connectors.curate.env import run_isolated
from timenet_connectors.discovery import connector_dir, has_connector, resolve


class ConnectorCurator:
    """Builds a dataset by running its connector in an environment built from its requirements."""

    def knows(self, dataset_id: str) -> bool:  # noqa: PLR6301 (CuratorBackend protocol method)
        """Report whether a connector package exists for the id, without importing it.

        Args:
            dataset_id: The dataset id.

        Returns:
            Whether a connector exists.
        """
        return has_connector(dataset_id)

    def declared_version(self, dataset_id: str) -> str | None:  # noqa: PLR6301 (protocol)
        """Read the version the connector's card declares, without importing or building it.

        Args:
            dataset_id: The dataset id.

        Returns:
            The declared version string, or ``None`` if no readable card exists.
        """
        from timenet.errors import InvalidCardError  # noqa: PLC0415
        from timenet.types import DatasetMetadata  # noqa: PLC0415

        try:
            card = DatasetMetadata.from_yaml(connector_dir(dataset_id) / "dataset.yaml")
        except (LookupError, InvalidCardError):
            return None
        return str(card.dataset_version)

    def build(self, dataset_id: str, root: Path, *, force: bool = False) -> Path:  # noqa: PLR6301 (protocol)
        """Build the dataset, in an isolated environment unless isolation is turned off.

        ``TIMENET_ISOLATION=off`` builds in this interpreter instead, as ``--no-isolation`` does on
        the CLI. That is also the recursion guard: an isolated child carries the setting, so a
        connector that loads another dataset from inside one does not re-exec uv forever.

        Args:
            dataset_id: The dataset id.
            root: The output registry directory.
            force: Rebuild even if the version is already curated.

        Returns:
            The committed version directory.
        """
        if settings().isolation == "off":
            return run_pipeline(resolve(dataset_id)(), root, force=force)
        return run_isolated(dataset_id, root, force=force)
