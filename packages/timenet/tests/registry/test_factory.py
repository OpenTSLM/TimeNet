import pytest

from timenet.errors import TimeNetDatasetNotFoundError
from timenet.registry import LocalRegistry


def test_an_unknown_dataset_names_the_registry_and_the_dataset(tmp_path):
    with pytest.raises(TimeNetDatasetNotFoundError) as caught:
        LocalRegistry(tmp_path).get_manifest("nope/missing")
    assert str(caught.value) == f"local registry {tmp_path} holds no version of 'nope/missing'"
