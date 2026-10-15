import pytest

from timenet.errors import TimeFValidationError
from timenet.types.references import DatasetRef


def test_exact_reference_rejects_unpinned_versions():
    ref = DatasetRef.parse("physionet/ptb-xl@1.0.0")
    assert str(ref) == "physionet/ptb-xl@1.0.0"
    for invalid in ("physionet/ptb-xl", "physionet/ptb-xl@latest"):
        with pytest.raises(TimeFValidationError, match="exact"):
            DatasetRef.parse(invalid)
