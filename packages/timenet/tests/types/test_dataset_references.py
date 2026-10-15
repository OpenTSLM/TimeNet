import pytest

from timenet.errors import TimeFValidationError
from timenet.types.references import DatasetRef, manifest_digest


def test_exact_reference_and_canonical_manifest_digest():
    ref = DatasetRef.parse("physionet/ptb-xl@1.0.0")
    assert str(ref) == "physionet/ptb-xl@1.0.0"
    for invalid in ("physionet/ptb-xl", "physionet/ptb-xl@latest"):
        with pytest.raises(TimeFValidationError, match="exact"):
            DatasetRef.parse(invalid)
    assert manifest_digest({"b": 2, "a": 1}) == manifest_digest({"a": 1, "b": 2})
    assert manifest_digest({"a": 1}) != manifest_digest({"a": 2})
