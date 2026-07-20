import pytest

from timenet.errors import InvalidCardError
from timenet.types import DatasetMetadata


_VALID_CARD = """
yaml_schema_version: 1
dataset_id: hello_world
dataset_version: 1.0.0
name: Hello World
description: A synthetic demo dataset.
license: CC-BY-4.0
domains:
  - general
tags: [demo, synthetic]
"""


def _write(tmp_path, text):
    path = tmp_path / "card.yaml"
    path.write_text(text)
    return path


def test_from_yaml_round_trips_a_valid_card(tmp_path):
    m = DatasetMetadata.from_yaml(_write(tmp_path, _VALID_CARD))
    assert m.dataset_id == "hello_world"
    assert m.name == "Hello World"
    assert [str(d) for d in m.domains] == ["general"]
    assert m.tags == ("demo", "synthetic")


def test_from_yaml_rejects_a_card_missing_required_fields(tmp_path):
    with pytest.raises(InvalidCardError, match="failed validation"):
        DatasetMetadata.from_yaml(_write(tmp_path, "dataset_id: hello_world\n"))


def test_from_yaml_rejects_a_non_mapping(tmp_path):
    with pytest.raises(InvalidCardError, match="must be a YAML mapping"):
        DatasetMetadata.from_yaml(_write(tmp_path, "- just\n- a\n- list\n"))


def test_from_yaml_rejects_a_bad_enum_value(tmp_path):
    bad = _VALID_CARD.replace("license: CC-BY-4.0", "license: NOT-A-LICENSE")
    with pytest.raises(InvalidCardError):
        DatasetMetadata.from_yaml(_write(tmp_path, bad))
