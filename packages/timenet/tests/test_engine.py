from pathlib import Path

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.engine import run_pipeline
from timenet.manifest import Manifest
from timenet.testing import make_dataset
from timenet.types import DatasetMetadata


class _DemoConnector(BaseConnector[str]):
    def metadata(self) -> DatasetMetadata:
        return make_dataset().metadata

    def download(self, cache_dir: Path) -> list[str]:
        return ["ref"]

    def convert(self, raw_refs: list[str]) -> TimeFDataset:
        return make_dataset()


def test_store_writes_a_readable_layout(tmp_path):
    connector = _DemoConnector()
    dataset = connector.convert(connector.download(tmp_path))
    version_dir = connector.store(dataset, tmp_path)
    assert (version_dir / "manifest.json").exists()
    assert version_dir == tmp_path / "hello_world" / "1.0.0"


def test_store_derives_schema_if_needed(tmp_path):
    connector = _DemoConnector()
    dataset = connector.convert(connector.download(tmp_path))
    assert dataset.schema is None
    connector.store(dataset, tmp_path)  # should derive schema itself
    assert dataset.schema is not None


def test_run_pipeline_end_to_end(tmp_path):
    version_dir = run_pipeline(_DemoConnector(), tmp_path, cache_dir=tmp_path / "cache")
    manifest = Manifest.from_json((version_dir / "manifest.json").read_text())
    assert manifest.counts.samples == 3
    assert manifest.dataset_id == "hello_world"


class _CountingConnector(_DemoConnector):
    def __init__(self) -> None:
        self.downloads = 0

    def download(self, cache_dir: Path) -> list[str]:
        self.downloads += 1
        return ["ref"]


def test_run_pipeline_is_idempotent(tmp_path):
    connector = _CountingConnector()
    first = run_pipeline(connector, tmp_path, cache_dir=tmp_path / "cache")
    second = run_pipeline(connector, tmp_path, cache_dir=tmp_path / "cache")
    assert first == second
    assert connector.downloads == 1  # the second run skipped download/convert/store


def test_run_pipeline_force_rebuilds(tmp_path):
    connector = _CountingConnector()
    run_pipeline(connector, tmp_path, cache_dir=tmp_path / "cache")
    version_dir = run_pipeline(connector, tmp_path, cache_dir=tmp_path / "cache", force=True)
    assert connector.downloads == 2
    assert (version_dir / "manifest.json").exists()
