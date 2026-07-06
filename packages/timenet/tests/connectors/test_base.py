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


def test_concrete_connector_implements_contract(tmp_path):
    card = tmp_path / "dataset.yaml"
    card.write_text("dataset_id: demo/thing\ndataset_version: 1.0.0\nname: Demo\ndescription: d\nlicense: MIT\n")

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
