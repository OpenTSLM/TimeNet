"""A local registry with the demo release and a child that reads one of its Records."""

from pathlib import Path

from timenet.dataset import TimeFDataset
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, open_local_version
from timenet.registry._paths import release_dir
from timenet.testing import make_dataset
from timenet.types.reference_keys import ParentAlias
from timenet.types.references import DatasetRef


def _child_of_the_demo(root: Path) -> tuple[LocalRegistry, DatasetRef]:
    """Store the demo dataset and ``test/child``, which reads one of its Records, in a local registry.

    Returns:
        The registry, and the demo release, which is the payload source of the child.
    """
    parent = make_dataset()
    registry = LocalRegistry(root)
    registry.put_dataset(parent)
    reference = DatasetRef(dataset_id=parent.metadata.dataset_id, version=parent.metadata.dataset_version)
    child = TimeFDataset(
        metadata=parent.metadata.model_copy(
            update={"dataset_id": "test/child", "parents": {ParentAlias("parent"): reference}}
        )
    )
    with TimeFReader(open_local_version(release_dir(root, reference))) as reader:
        child.import_from(reader, alias=ParentAlias("parent"), records=[parent.records[0].id])
    registry.put_dataset(child)
    return registry, reference
