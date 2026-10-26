"""Open the parent releases a connector build declares."""

from dataclasses import dataclass, field

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry.base import BaseRegistry
from timenet.registry.dependencies import prepare_dependency_graph
from timenet.registry.local import LocalRegistry
from timenet.types import DatasetMetadata
from timenet.types.reference_keys import InputAlias
from timenet.types.references import DatasetRef


@dataclass
class BuildContext:
    """The declared parent releases of one connector build, opened by input alias.

    The context never fetches from elsewhere or builds a parent. ``timenet-build`` prepares missing
    parents before the build starts, so a parent that is still missing here is an error.
    """

    references: dict[InputAlias, DatasetRef] = field(default_factory=dict)
    inputs: dict[InputAlias, TimeFReader] = field(default_factory=dict)

    @classmethod
    def open(cls, metadata: DatasetMetadata, registry: BaseRegistry) -> "BuildContext":
        """Open each declared input from the output registry before connector conversion starts.

        A local registry must hold every parent complete. Remote parents resolve their physical
        source manifests upfront and read catalogs and payloads only when conversion needs them.

        Args:
            metadata: The child dataset card with exact input references.
            registry: The registry the child is written to.

        Returns:
            Opened readers keyed by input alias.

        Raises:
            TimeNetDatasetNotFoundError: If the registry does not hold a declared parent.
        """  # noqa: DOC502 - registry preparation raises the documented error
        context = cls(references=dict(metadata.inputs))
        try:
            for alias, ref in metadata.inputs.items():
                version = (
                    prepare_dependency_graph(registry.root, ref)
                    if isinstance(registry, LocalRegistry)
                    else registry.open_version(ref.dataset_id, str(ref.version))
                )
                context.inputs[alias] = TimeFReader(version)
        except BaseException:
            context.close()
            raise
        return context

    def check_imports(self, dataset: TimeFDataset) -> None:
        """Require the dataset to import each declared input under its alias, and nothing else.

        ``import_from`` records the card, lineage, and access requirement of every parent it copies
        from. Matching those parent cards against the declared inputs catches a connector that ignores an input,
        imports an undeclared one, or imports a parent under a different alias.

        Args:
            dataset: The composed dataset.

        Raises:
            TimeFValidationError: If the imported parents differ from the declared inputs.
        """
        imported = {alias: card.source for alias, card in dataset.parent_cards.items()}
        if imported != self.references:
            raise TimeFValidationError(
                f"{dataset.metadata.dataset_id} declares inputs {_describe(self.references)} but imports "
                f"{_describe(imported)}; import each declared input under its declared alias"
            )

    def close(self) -> None:
        """Release the parent readers once the writer has consumed them."""
        for parent in self.inputs.values():
            parent.close()

    def __enter__(self) -> "BuildContext":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _describe(references: dict[InputAlias, DatasetRef]) -> str:
    return "{" + ", ".join(f"{alias}: {ref}" for alias, ref in sorted(references.items())) + "}"
