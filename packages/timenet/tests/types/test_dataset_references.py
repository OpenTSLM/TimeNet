import pytest

from timenet.errors import TimeFValidationError
from timenet.types.references import DatasetRef


def test_exact_reference_rejects_unpinned_versions():
    ref = DatasetRef.parse("physionet/ptb-xl@1.0.0")
    assert str(ref) == "physionet/ptb-xl@1.0.0"
    for invalid in ("physionet/ptb-xl", "physionet/ptb-xl@latest"):
        with pytest.raises(TimeFValidationError, match="exact"):
            DatasetRef.parse(invalid)


@pytest.mark.parametrize("value", ["org/name@", "org/name@1.0", "a@b@1.0.0", "@1.0.0"])
def test_exact_reference_reports_malformed_references_as_timenet_errors(value):
    with pytest.raises(TimeFValidationError):
        DatasetRef.parse(value)


def test_exact_reference_reports_an_invalid_dataset_id_as_a_timenet_error():
    with pytest.raises(TimeFValidationError, match="invalid dataset reference"):
        DatasetRef.parse("not-a-dataset-id@1.0.0")
