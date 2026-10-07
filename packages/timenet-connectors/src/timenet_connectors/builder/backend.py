"""The build backend the SDK reaches through the ``timenet.builders`` entry point."""

from pathlib import Path

from timenet.builders import find_builder
from timenet.config import settings
from timenet.engine import run_pipeline
from timenet.errors import TimeNetBuildError, TimeNetDatasetNotFoundError
from timenet.registry import LocalRegistry
from timenet.types import DatasetMetadata
from timenet_connectors.builder.env import run_isolated
from timenet_connectors.discovery import connector_dir, has_connector, resolve


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

    def declared_version(self, dataset_id: str) -> str | None:  # noqa: PLR6301 (protocol)
        """Read the version the connector's card declares, without importing or building it.

        Args:
            dataset_id: The dataset id.

        Returns:
            The declared version string, or ``None`` if no readable card exists.
        """
        from timenet.errors import TimeNetInvalidCardError  # noqa: PLC0415

        try:
            card = DatasetMetadata.from_yaml(connector_dir(dataset_id) / "dataset.yaml")
        except (LookupError, TimeNetInvalidCardError):
            return None
        return str(card.dataset_version)

    def build(
        self,
        dataset_id: str,
        root: Path,
        *,
        force: bool = False,
        values_backend: str | None = None,
    ) -> Path:
        """Build the dataset and any missing parents.

        The build uses an isolated environment unless ``TIMENET_ISOLATION=off``.

        Args:
            dataset_id: The dataset id.
            root: The output registry directory.
            force: Rebuild even if the version is already built.
            values_backend: Values storage backend. Defaults to the connector's backend.

        Returns:
            The committed version directory.
        """
        return self._build(dataset_id, root, force=force, values_backend=values_backend, ancestors=())

    @staticmethod
    def _build(
        dataset_id: str,
        root: Path,
        *,
        force: bool,
        values_backend: str | None,
        ancestors: tuple[str, ...],
    ) -> Path:
        """Build a dataset after its direct parents.

        Returns:
            The committed version directory.
        """
        build_parents(dataset_id, root, ancestors=ancestors)
        if settings().isolation == "off":
            connector = resolve(dataset_id)()
            resolved_backend = connector.values_backend if values_backend is None else values_backend
            return run_pipeline(connector, root, force=force, values_backend=resolved_backend)
        return Path(run_isolated(dataset_id, root, force=force, values_backend=values_backend))


def build_parents(dataset_id: str, root: Path, *, ancestors: tuple[str, ...] = ()) -> None:
    """Build the parents declared in the connector's card that are not yet present.

    Args:
        dataset_id: The child dataset id.
        root: The output registry directory.
        ancestors: Dataset ids in the current build chain, for cycle detection.

    Raises:
        TimeNetBuildError: If the parent declarations contain a cycle.
        TimeNetDatasetNotFoundError: If a required parent version cannot be built.
    """
    if dataset_id in ancestors:
        cycle = " -> ".join((*ancestors, dataset_id))
        raise TimeNetBuildError(f"dataset dependency graph contains a cycle: {cycle}")
    metadata = DatasetMetadata.from_yaml(connector_dir(dataset_id) / "dataset.yaml")
    registry = LocalRegistry(root)
    for parent in metadata.parents:
        parent_id, version = parent.dataset_id, str(parent.version)
        if registry.exists(parent_id, version):
            continue
        builder = find_builder(parent_id)
        if builder is None:
            raise TimeNetDatasetNotFoundError(f"parent {parent} is missing from {root} and no connector can build it")
        declared = builder.declared_version(parent_id)
        if declared is not None and declared != version:
            raise TimeNetDatasetNotFoundError(f"parent {parent} is pinned exactly, but its connector builds {declared}")
        if isinstance(builder, ConnectorBuilder):
            builder._build(parent_id, root, force=False, values_backend=None, ancestors=(*ancestors, dataset_id))
        else:
            builder.build(parent_id, root)
        if not registry.exists(parent_id, version):
            raise TimeNetDatasetNotFoundError(f"builder did not produce required parent {parent}")
