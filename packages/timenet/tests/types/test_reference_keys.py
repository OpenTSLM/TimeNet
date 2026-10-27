import json
import pickle

from pydantic import ValidationError
import pytest

from timenet.errors import TimeFValidationError
from timenet.types import DatasetMetadata, License, Version
from timenet.types.reference_keys import ParentAlias
from timenet.types.references import DatasetRef


_PTBXL = DatasetRef.parse("physionet/ptb-xl@1.0.0")

_CARD = """
dataset_id: physionet/ecg-qa-cot
dataset_version: 1.0.0
name: ECG-QA CoT
description: Questions over PTB-XL recordings.
license: MIT
parents:
  ptbxl: physionet/ptb-xl@1.0.0
"""


def _metadata(**overrides):
    values = {
        "dataset_id": "test/child",
        "dataset_version": Version(1, 0, 0),
        "name": "Child",
        "description": "d",
        "license": License.MIT,
    }
    values.update(overrides)
    return DatasetMetadata.model_validate(values)


def test_a_parent_alias_is_a_str_that_code_can_tell_apart():
    alias = ParentAlias("ptbxl")
    assert isinstance(alias, str)
    assert not isinstance("ptbxl", ParentAlias)
    assert alias == "ptbxl"
    assert {"ptbxl": 1}[alias] == 1
    assert repr(alias) == "'ptbxl'"
    assert type(str(alias)) is str
    assert type(pickle.loads(pickle.dumps(alias))) is ParentAlias


@pytest.mark.parametrize("value", ["", " ", "\t\n", None, 7])
def test_a_parent_alias_must_be_a_nonblank_string(value):
    with pytest.raises(TimeFValidationError, match="a parent alias must be a nonblank string"):
        ParentAlias(value)


def test_a_card_validates_its_parent_keys_into_aliases(tmp_path):
    path = tmp_path / "card.yaml"
    path.write_text(_CARD)
    from_yaml = DatasetMetadata.from_yaml(path)
    from_json = DatasetMetadata.model_validate_json(json.dumps(from_yaml.model_dump(mode="json")))
    for metadata in (from_yaml, from_json):
        assert metadata.parents == {"ptbxl": _PTBXL}
        assert [type(alias) for alias in metadata.parents] == [ParentAlias]


def test_parent_aliases_serialize_as_plain_strings():
    metadata = _metadata(parents={ParentAlias("ptbxl"): _PTBXL})
    assert [type(alias) for alias in metadata.model_dump()["parents"]] == [str]
    assert list(json.loads(metadata.model_dump_json())["parents"]) == ["ptbxl"]
    assert DatasetMetadata.model_validate_json(metadata.model_dump_json()) == metadata


@pytest.mark.parametrize("alias", ["", "  ", 1])
def test_a_card_rejects_a_blank_or_non_string_parent_alias(alias):
    with pytest.raises(ValidationError, match="parents"):
        _metadata(parents={alias: "physionet/ptb-xl@1.0.0"})
