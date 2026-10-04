import math
from typing import cast

import numpy as np
from pydantic import ValidationError
import pytest

from timenet._json import JsonMapping
from timenet.dataset import OrdinalAxis, Record, Signal, Source
from timenet.format.control_writer import _json
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
