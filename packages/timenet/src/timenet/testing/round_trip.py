"""Store a dataset, verify the stored release, and compare what a reader gets back."""

from pathlib import Path

from timenet.dataset import TimeFDataset
from timenet.errors import TimeFValidationError
from timenet.reader import LoadPlan, TimeFReader
from timenet.registry import LocalRegistry
from timenet.testing.builders import assert_datasets_equal


def assert_round_trips(dataset: TimeFDataset, registry_dir: Path) -> TimeFDataset:
    """Write ``dataset`` through a local registry, verify the release, and check that reading preserves it.

    A connector test calls this on the dataset that ``convert()`` or ``compose()`` returned. The
    helper writes the dataset into a local registry, runs
    :meth:`~timenet.reader.TimeFReader.verify` on the release, reads the release back with every
    value loaded, and compares it with :func:`~timenet.testing.assert_datasets_equal`.

    Args:
        dataset: The dataset to store.
        registry_dir: The directory of the local registry, such as ``tmp_path / "registry"``. For a
            dataset from ``compose()``, the registry must already hold the parent releases.

    Returns:
        The dataset read back from the release, for further assertions.

    Raises:
        TimeFValidationError: If the registry already holds this dataset id and version. ``store`` would then
            keep the old release, and the helper would compare that one instead.
        TimeNetFileError: If a file of the release is missing or differs from the manifest.
        AssertionError: If the release reads back different from ``dataset``.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by verify and assert_datasets_equal
    registry = LocalRegistry(registry_dir)
    dataset_id, version = dataset.metadata.dataset_id, str(dataset.metadata.dataset_version)
    if registry.exists(dataset_id, version):
        raise TimeFValidationError(
            f"the registry at {registry_dir} already holds {dataset_id}@{version}; give each round trip "
            f"its own registry directory"
        )
    version = registry.store(dataset)
    with TimeFReader(registry.open_version(dataset.metadata.dataset_id, version)) as reader:
        reader.verify()
        restored = reader.read(load=LoadPlan.ALL_EAGER)
    assert_datasets_equal(dataset, restored)
    return restored
