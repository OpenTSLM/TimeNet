"""The build backend the SDK reaches through the ``timenet.builders`` entry point."""

from pathlib import Path

from timenet.config import settings
from timenet.engine import run_pipeline
from timenet.errors import TimeNetInvalidCardError
from timenet.types import DatasetMetadata
from timenet_connectors.builder.env import run_isolated
from timenet_connectors.discovery import has_connector, read_card, resolve


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

    def build(  # noqa: PLR6301 (protocol)
        self,
        dataset_id: str,
        root: Path,
        *,
        force: bool = False,
        values_backend: str | None = None,
    ) -> Path:
        """Build the dataset, in an isolated environment unless isolation is turned off.

        ``TIMENET_ISOLATION=off`` builds in this interpreter instead, as ``--no-isolation`` does on
        the CLI. That is also the recursion guard: an isolated child carries the setting, so a
        connector that loads another dataset from inside one does not re-exec uv forever.

        Args:
            dataset_id: The dataset id.
            root: The output registry directory.
            force: Rebuild even if the version is already built.
            values_backend: Storage backend for the values plane. Defaults to the connector's value.

        Returns:
            The committed version directory.
        """
        if settings().isolation == "off":
            connector = resolve(dataset_id)()
            resolved_backend = connector.values_backend if values_backend is None else values_backend
            return run_pipeline(connector, root, force=force, values_backend=resolved_backend)
        return Path(run_isolated(dataset_id, root, force=force, values_backend=values_backend))
