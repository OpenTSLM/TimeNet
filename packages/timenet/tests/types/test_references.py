import pytest

from timenet.errors import TimeFValidationError
from timenet.types import DatasetRef, ObjectKind, ObjectRef, ParentDataset, Version


def test_dataset_reference_round_trip():
    reference = DatasetRef("physionet/ptb-xl", Version(1, 2, 3))
    assert str(reference) == "physionet/ptb-xl@1.2.3"
    assert DatasetRef.parse(str(reference)) == reference


def test_object_reference_round_trip_escapes_local_id():
    reference = ObjectRef(
        DatasetRef("physionet/ptb-xl", Version(1, 0, 0)),
        ObjectKind.RECORD,
        "records/00001:low resolution",
    )
    assert ObjectRef.parse(str(reference)) == reference
    assert str(reference).endswith("#record:records%2F00001%3Alow%20resolution")


@pytest.mark.parametrize("alias", ["PTBXL", "1parent", "has space", "has.dot"])
def test_parent_alias_is_safe_identifier(alias):
    with pytest.raises(TimeFValidationError, match="parent alias"):
        ParentDataset(alias, DatasetRef("org/name", Version(1, 0, 0)))
