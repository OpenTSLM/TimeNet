from pydantic import ValidationError
import pytest

from timenet.types import DatasetRef, ParentDataset, Version


def test_dataset_reference_formats_and_orders_by_id_then_version():
    older = DatasetRef(dataset_id="physionet/ptb-xl", version=Version(1, 2, 3))
    newer = DatasetRef(dataset_id="physionet/ptb-xl", version=Version(1, 10, 0))
    other = DatasetRef(dataset_id="timenet/hello-world", version=Version(0, 1, 0))
    assert str(older) == "physionet/ptb-xl@1.2.3"
    assert sorted([other, newer, older]) == [older, newer, other]


def test_parent_dataset_exposes_its_exact_reference():
    parent = ParentDataset(alias="ptbxl", dataset_id="physionet/ptb-xl", version=Version(1, 0, 0))
    assert parent.dataset == DatasetRef(dataset_id="physionet/ptb-xl", version=Version(1, 0, 0))


@pytest.mark.parametrize("alias", ["PTBXL", "1parent", "has space", "has.dot"])
def test_parent_alias_is_safe_identifier(alias):
    with pytest.raises(ValidationError, match="alias"):
        ParentDataset(alias=alias, dataset_id="org/name", version=Version(1, 0, 0))
