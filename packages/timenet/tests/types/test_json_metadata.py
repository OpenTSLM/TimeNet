import copy
import json
import math
import pickle
from typing import cast

import numpy as np
from pydantic import ValidationError
import pytest

from timenet.dataset import OrdinalAxis, Record, Signal, Source
from timenet.format.control_writer import _json
from timenet.json import JsonMapping, freeze_json_mapping, validate_json_mapping, validate_json_value
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


def _frozen_sample() -> JsonMapping:
    return freeze_json_mapping({"nested": [1, {"flag": True}], "note": "text"})


def _nested_list(frozen: JsonMapping) -> list:
    nested = frozen["nested"]
    assert isinstance(nested, list)
    return nested


_DICT_CHANGES = {
    "setitem": lambda value: value.__setitem__("added", 1),
    "delitem": lambda value: value.__delitem__("note"),
    "ior": lambda value: value.__ior__({"added": 1}),
    "clear": lambda value: value.clear(),
    "pop": lambda value: value.pop("note"),
    "popitem": lambda value: value.popitem(),
    "setdefault": lambda value: value.setdefault("added", 1),
    "update": lambda value: value.update(added=1),
}
_LIST_CHANGES = {
    "setitem": lambda value: value.__setitem__(0, 2),
    "slice": lambda value: value.__setitem__(slice(0, 1), [2]),
    "delitem": lambda value: value.__delitem__(0),
    "iadd": lambda value: value.__iadd__([2]),
    "imul": lambda value: value.__imul__(2),
    "append": lambda value: value.append(2),
    "clear": lambda value: value.clear(),
    "extend": lambda value: value.extend([2]),
    "insert": lambda value: value.insert(0, 2),
    "pop": lambda value: value.pop(),
    "remove": lambda value: value.remove(1),
    "reverse": lambda value: value.reverse(),
    "sort": lambda value: value.sort(),
}


@pytest.mark.parametrize("change", _DICT_CHANGES.values(), ids=_DICT_CHANGES.keys())
def test_a_frozen_json_object_refuses_every_change(change):
    frozen = _frozen_sample()
    with pytest.raises(TypeError, match="cannot change"):
        change(frozen)
    assert frozen == {"nested": [1, {"flag": True}], "note": "text"}


@pytest.mark.parametrize("change", _LIST_CHANGES.values(), ids=_LIST_CHANGES.keys())
def test_a_frozen_json_array_refuses_every_change(change):
    frozen = _frozen_sample()
    with pytest.raises(TypeError, match="cannot change"):
        change(_nested_list(frozen))
    assert frozen == {"nested": [1, {"flag": True}], "note": "text"}


def test_frozen_json_is_detached_from_the_source_at_every_level():
    source = {"nested": [{"value": 1}]}
    frozen = freeze_json_mapping(source)
    source["nested"][0]["value"] = 2
    source["nested"].append({"value": 3})
    assert frozen == {"nested": [{"value": 1}]}
    item = _nested_list(frozen)[0]
    assert isinstance(item, dict)
    with pytest.raises(TypeError, match="cannot change"):
        item["value"] = 3


@pytest.mark.parametrize(
    "copy_value", [lambda value: pickle.loads(pickle.dumps(value)), copy.deepcopy], ids=["pickle", "deepcopy"]
)
def test_frozen_json_keeps_its_json_and_stays_frozen_after_a_copy(copy_value):
    source = {"nested": [1, {"flag": True}], "note": "text"}
    frozen = freeze_json_mapping(source)
    assert json.dumps(frozen, sort_keys=True) == json.dumps(source, sort_keys=True)
    restored = copy_value(frozen)
    assert restored == source
    with pytest.raises(TypeError, match="cannot change"):
        restored["added"] = True
    with pytest.raises(TypeError, match="cannot change"):
        _nested_list(restored).clear()
    item = _nested_list(restored)[1]
    assert isinstance(item, dict)
    with pytest.raises(TypeError, match="cannot change"):
        item["flag"] = False


@pytest.mark.parametrize("source", [{"bad": math.nan}, {1: "value"}, {"bad": (1, 2)}, {"bad": np.float64(1.0)}])
def test_freezing_preserves_strict_json_validation(source):
    with pytest.raises(ValidationError):
        freeze_json_mapping(source)
