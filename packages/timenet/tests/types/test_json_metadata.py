import math
from typing import cast

import numpy as np
from pydantic import ValidationError
import pytest

from timenet.dataset import OrdinalAxis, Record, Signal, Source
from timenet.format.control_writer import _json
from timenet.json import JsonMapping, validate_json_mapping, validate_json_value
from timenet.types import Annotation, AnswerTask, TimeSeriesSpec


_SPEC = TimeSeriesSpec(spec_type="value", name="Value", unit_value=None)


def _objects(metadata):
    return (
        lambda: Annotation(key="fact", value=1, metadata=metadata),
        lambda: Annotation(key="fact", value=1, occurrence_metadata=metadata),
        lambda: AnswerTask(metadata=metadata),
        lambda: Record(metadata=metadata),
        lambda: Source(name="source", metadata=metadata),
        lambda: Signal(spec=_SPEC, time_axis=OrdinalAxis(), name="signal", data=[1.0], metadata=metadata),
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {"bad": (1, 2)},
        {1: "not a string key"},
        {"bad": {1, 2}},
        {"bad": b"bytes"},
        {"bad": np.float64(1.0)},
        {"bad": math.nan},
        {"bad": math.inf},
        {"nested": [{"bad": np.float64(1.0)}]},
        {"nested": [{"bad": math.nan}]},
        {"nested": [{"bad": math.inf}]},
    ],
)
def test_metadata_fields_reject_non_json_values(metadata):
    for construct in _objects(metadata):
        with pytest.raises(ValidationError):
            construct()


def test_metadata_is_detached_from_the_callers_nested_containers():
    nested = [{"value": 1}]
    metadata = cast("JsonMapping", {"nested": nested})
    record = Record(metadata=metadata)
    nested[0]["value"] = 2
    assert record.metadata == {"nested": [{"value": 1}]}


def test_writer_json_validation_catches_post_construction_mutation():
    record = Record(metadata={"valid": True})
    record.metadata["later"] = {1, 2}  # type: ignore - simulate post-construction corruption
    with pytest.raises(ValidationError):
        _json(record.metadata)


@pytest.mark.parametrize("value", [None, True, 1, 1.0, "text", [1, {"value": False}]])
def test_json_values_keep_native_types(value):
    validated = validate_json_value(value)
    assert validated == value
    assert type(validated) is type(value)


def test_json_validation_reports_cycles():
    metadata = {}
    metadata["cycle"] = metadata
    with pytest.raises(ValidationError, match="recursion"):
        validate_json_mapping(metadata)
