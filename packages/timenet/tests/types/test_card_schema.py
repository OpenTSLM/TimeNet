import jsonschema
import pytest

from timenet.errors import InvalidCardError
from timenet.schemas import DATASET_CARD_SCHEMA
from timenet.types import DatasetMetadata, Domain, License, Version


_VALID_CARD = """\
yaml_schema_version: 1
dataset_id: demo/thing
dataset_version: 1.2.3
name: Demo
description: A demo dataset.
license: CC-BY-4.0
domains:
  - general
  - health
tags:
  - a
  - b
source_url: https://example.com/demo
"""

_MINIMAL_CARD = "dataset_id: demo/thing\ndataset_version: 1.0.0\nname: D\ndescription: d\nlicense: MIT\n"


def _write(tmp_path, text):
    path = tmp_path / "dataset.yaml"
    path.write_text(text)
    return path


def test_schema_is_well_formed():
    jsonschema.Draft202012Validator.check_schema(DATASET_CARD_SCHEMA)


def test_from_yaml_loads_a_valid_card(tmp_path):
    assert DatasetMetadata.from_yaml(_write(tmp_path, _VALID_CARD)) == DatasetMetadata(
        dataset_id="demo/thing",
        dataset_version=Version(1, 2, 3),
        name="Demo",
        description="A demo dataset.",
        license=License.CC_BY_4_0,
        domains=(Domain.GENERAL, Domain.HEALTH),
        tags=("a", "b"),
        source_url="https://example.com/demo",
    )


def test_from_yaml_applies_defaults_for_optional_fields(tmp_path):
    m = DatasetMetadata.from_yaml(_write(tmp_path, _MINIMAL_CARD))
    assert (m.domains, m.tags, m.source_url, m.yaml_schema_version) == ((), (), None, 1)


@pytest.mark.parametrize(
    "bad",
    [
        "dataset_version: 1.0.0\nname: D\ndescription: d\nlicense: MIT\n",  # missing dataset_id
        _MINIMAL_CARD.replace("MIT", "Nope"),  # unknown license
        _MINIMAL_CARD.replace("demo/thing", "noslash"),  # id without a slash
        _MINIMAL_CARD.replace("1.0.0", "1.0"),  # not a full semver
        _MINIMAL_CARD + "extra: x\n",  # unknown key (additionalProperties: false)
    ],
)
def test_from_yaml_rejects_invalid_cards(tmp_path, bad):
    with pytest.raises(InvalidCardError):
        DatasetMetadata.from_yaml(_write(tmp_path, bad))


def test_from_yaml_rejects_non_mapping(tmp_path):
    with pytest.raises(InvalidCardError):
        DatasetMetadata.from_yaml(_write(tmp_path, "- just\n- a\n- list\n"))


def test_from_yaml_rejects_missing_file(tmp_path):
    with pytest.raises(InvalidCardError):
        DatasetMetadata.from_yaml(tmp_path / "does_not_exist.yaml")
