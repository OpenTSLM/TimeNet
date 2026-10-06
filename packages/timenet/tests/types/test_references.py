import pickle

import jsonschema
from pydantic import ValidationError
import pytest

from timenet.types import DatasetRef, Version


def test_dataset_reference_formats_and_orders_by_id_then_version():
    older = DatasetRef(dataset_id="physionet/ptb-xl", version=Version(1, 2, 3))
    newer = DatasetRef(dataset_id="physionet/ptb-xl", version=Version(1, 10, 0))
    other = DatasetRef(dataset_id="timenet/hello-world", version=Version(0, 1, 0))
    assert str(older) == "physionet/ptb-xl@1.2.3"
    assert sorted([other, newer, older]) == [older, newer, other]


def test_exact_reference_string_round_trips_and_preserves_identity():
    reference = DatasetRef.model_validate("physionet/ptb-xl@1.0.0")
    expected = DatasetRef(dataset_id="physionet/ptb-xl", version=Version(1, 0, 0))
    assert reference == expected
    assert hash(reference) == hash(expected)
    assert reference.model_dump(mode="json") == "physionet/ptb-xl@1.0.0"
    assert DatasetRef.model_validate_json(reference.model_dump_json()) == reference
    assert pickle.loads(pickle.dumps(reference)) == reference
    jsonschema.validate(reference.model_dump(mode="json"), DatasetRef.model_json_schema())


@pytest.mark.parametrize(
    "reference",
    ["org/name", "org/name@latest", "org/name@", "org/name@1.0", "org/name@1.0.0@2.0.0", "name@1.0.0"],
)
def test_exact_reference_rejects_unpinned_or_invalid_versions(reference):
    with pytest.raises(ValidationError):
        DatasetRef.model_validate(reference)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(reference, DatasetRef.model_json_schema())
