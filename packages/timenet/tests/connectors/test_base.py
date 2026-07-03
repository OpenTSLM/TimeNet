from abc import ABC
from pathlib import Path

import pytest

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.testing import make_dataset


def test_base_connector_is_abstract():
    assert issubclass(BaseConnector, ABC)
    assert BaseConnector.__abstractmethods__ == frozenset({"download", "convert"})


def test_base_connector_cannot_instantiate():
    with pytest.raises(TypeError):
        BaseConnector()


_DEMO_CARD = """
dataset_id: demo/thing
dataset_version: 1.0.0
name: Demo
description: d
license: MIT
"""


def test_concrete_connector_implements_contract(tmp_path):
    card = tmp_path / "card.yaml"
    card.write_text(_DEMO_CARD)

    class DemoConnector(BaseConnector[str]):
        CARD = card

        def download(self, cache_dir: Path) -> list[str]:
            return ["ref"]

        def convert(self, raw_refs: list[str]) -> TimeFDataset:
            return make_dataset()

    connector = DemoConnector()
    assert connector.metadata().dataset_id == "demo/thing"
    assert connector.download(tmp_path) == ["ref"]
    assert isinstance(connector.convert(["ref"]), TimeFDataset)


def test_card_defaults_to_dataset_yaml_beside_the_connector(tmp_path, monkeypatch):
    # With CARD unset, the card is read from dataset.yaml in the connector module's folder.
    (tmp_path / "dataset.yaml").write_text(_DEMO_CARD)
    module = tmp_path / "conn.py"
    module.write_text("x = 1\n")

    class DemoConnector(BaseConnector[str]):
        def download(self, cache_dir: Path) -> list[str]:
            return []

        def convert(self, raw_refs: list[str]) -> TimeFDataset:
            return make_dataset()

    monkeypatch.setattr("inspect.getfile", lambda _cls: str(module))
    assert DemoConnector().metadata().dataset_id == "demo/thing"
