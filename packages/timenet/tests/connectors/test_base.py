from abc import ABC
from pathlib import Path

import pytest

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.testing import make_dataset


def test_base_connector_has_no_abstract_methods():
    assert issubclass(BaseConnector, ABC)
    assert BaseConnector.__abstractmethods__ == frozenset()


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


def _write_card(tmp_path: Path) -> Path:
    card = tmp_path / "dataset.yaml"
    card.write_text("dataset_id: demo/thing\ndataset_version: 1.0.0\nname: Demo\ndescription: d\nlicense: MIT\n")
    return card


def test_async_connector_download_bridges_to_download_async(tmp_path):
    card = _write_card(tmp_path)

    class AsyncConnector(BaseConnector[str]):
        CARD = card

        async def download_async(self, cache_dir: Path) -> list[str]:
            return ["async-ref"]

        def convert(self, raw_refs: list[str]) -> TimeFDataset:
            return make_dataset()

    # The engine only ever calls the sync download(); it must drive the async impl to completion.
    assert AsyncConnector().download(tmp_path) == ["async-ref"]


def test_connector_implementing_neither_download_raises(tmp_path):
    card = _write_card(tmp_path)

    class NoDownload(BaseConnector[str]):
        CARD = card

        def convert(self, raw_refs: list[str]) -> TimeFDataset:
            return make_dataset()

    with pytest.raises(TypeError, match="download"):
        NoDownload()


def test_connector_implementing_neither_convert_nor_compose_raises(tmp_path):
    card = _write_card(tmp_path)

    class NoConvert(BaseConnector[str]):
        CARD = card

        def download(self, cache_dir: Path) -> list[str]:
            return ["ref"]

    with pytest.raises(TypeError, match="convert"):
        NoConvert()
