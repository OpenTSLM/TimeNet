from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict
import pytest

from timenet.errors import TimeFValidationError
from timenet_connectors.metadata import RecordMetadata


class _Sampling(StrEnum):
    RANDOM = "random"


class _Device(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str


class _DatasetMetadata(RecordMetadata):
    version: Literal[2] = 2
    grams: float
    sampling: _Sampling
    watch: _Device
    strap: _Device | None = None


def test_dataset_metadata_is_one_flat_typed_object():
    metadata = _DatasetMetadata(subject_id="S1", grams=45, sampling=_Sampling.RANDOM, watch=_Device(name="w"))
    stored = metadata.to_json()
    assert stored == {"subject_id": "S1", "version": 2, "grams": 45.0, "sampling": "random", "watch": {"name": "w"}}
    restored = _DatasetMetadata.from_json(stored)
    assert restored == metadata
    assert restored.sampling is _Sampling.RANDOM
    # A key the class doesn't declare, or a version it doesn't read, is rejected rather than misread.
    with pytest.raises(TimeFValidationError, match="invalid _DatasetMetadata"):
        _DatasetMetadata.from_json({**stored, "note": "x"})
    with pytest.raises(TimeFValidationError, match="version"):
        _DatasetMetadata.from_json({**stored, "version": 1})


def test_a_dataset_cannot_redefine_a_shared_field():
    with pytest.raises(TimeFValidationError, match="redefines the shared metadata field 'subject_id'"):

        class _Clash(RecordMetadata):
            subject_id: int = 0  # ty: ignore[invalid-attribute-override]
