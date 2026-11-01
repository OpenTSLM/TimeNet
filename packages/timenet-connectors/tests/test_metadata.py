from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError
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
    windows: tuple[int, ...] = ()


def _metadata() -> _DatasetMetadata:
    return _DatasetMetadata(grams=45, sampling=_Sampling.RANDOM, watch=_Device(name="w"), windows=(1, 2))


def test_metadata_is_stored_as_one_flat_object_and_read_back():
    metadata = _metadata()
    stored = metadata.to_json()
    assert stored == {
        "version": 2,
        "grams": 45.0,
        "sampling": "random",
        "watch": {"name": "w"},
        "strap": None,
        "windows": [1, 2],
    }
    restored = _DatasetMetadata.from_json(stored)
    assert restored == metadata
    assert restored.sampling is _Sampling.RANDOM


def test_from_json_rejects_an_unknown_key():
    with pytest.raises(TimeFValidationError, match="invalid _DatasetMetadata"):
        _DatasetMetadata.from_json({**_metadata().to_json(), "note": "x"})


def test_from_json_rejects_another_version():
    with pytest.raises(TimeFValidationError, match="version"):
        _DatasetMetadata.from_json({**_metadata().to_json(), "version": 1})


def test_from_json_rejects_a_missing_key_instead_of_using_its_default():
    stored = dict(_metadata().to_json())
    del stored["version"]
    with pytest.raises(TimeFValidationError, match=r"lacks \['version'\]"):
        _DatasetMetadata.from_json(stored)


def test_from_json_rejects_a_value_of_the_wrong_type_instead_of_converting_it():
    with pytest.raises(TimeFValidationError, match="grams"):
        _DatasetMetadata.from_json({**_metadata().to_json(), "grams": "45"})


def test_a_value_of_the_wrong_type_or_a_nan_is_rejected_when_building():
    with pytest.raises(ValidationError, match="grams"):
        _DatasetMetadata(grams="45", sampling=_Sampling.RANDOM, watch=_Device(name="w"))  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValidationError, match="finite"):
        _DatasetMetadata(grams=float("nan"), sampling=_Sampling.RANDOM, watch=_Device(name="w"))


class _Sensor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    serial: str | None


class _NullableMetadata(RecordMetadata):
    battery: float | None
    unit: str | None = "mg/dL"
    sensor: _Sensor


def test_a_none_value_is_stored_and_read_back():
    # A required nullable field, a None that overrides a non-None default, and a None inside a nested model.
    metadata = _NullableMetadata(battery=None, unit=None, sensor=_Sensor(serial=None))
    stored = metadata.to_json()
    assert stored == {"battery": None, "unit": None, "sensor": {"serial": None}}
    assert _NullableMetadata.from_json(stored) == metadata
