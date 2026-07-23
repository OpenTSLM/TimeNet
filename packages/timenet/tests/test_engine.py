from pathlib import Path
import tempfile

from timenet.config import settings
from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.engine import run_pipeline, store_dataset
from timenet.manifest import Manifest
from timenet.testing import make_dataset


def _write_demo_card() -> Path:
    """Write a dataset.yaml mirroring ``make_dataset()``'s metadata, for the demo connector's card.

    The demo connector declares its identity in a card like a real connector, rather than overriding
    ``metadata()``. Generating the card from ``make_dataset().metadata`` keeps the two in step as the
    fixture's id/version change across the stack.

    Returns:
        Path to the written card.
    """
    m = make_dataset().metadata
    lines = [
        f"dataset_id: {m.dataset_id}",
        f"dataset_version: {m.dataset_version}",
        f'name: "{m.name}"',
        f'description: "{m.description}"',
        f"license: {m.license}",
    ]
    if m.domains:
        lines.append("domains:")
        lines += [f"  - {d}" for d in m.domains]
    if m.tags:
        lines.append("tags:")
        lines += [f"  - {t}" for t in m.tags]
    card = Path(tempfile.mkdtemp()) / "dataset.yaml"
    card.write_text("\n".join(lines) + "\n")
    return card


_DEMO_CARD = _write_demo_card()


class _DemoConnector(BaseConnector[str]):
    CARD = _DEMO_CARD

    def download(self, cache_dir: Path) -> list[str]:
        return ["ref"]

    def convert(self, raw_refs: list[str]) -> TimeFDataset:
        return make_dataset()


def test_store_writes_a_readable_layout(tmp_path):
    connector = _DemoConnector()
    dataset = connector.convert(connector.download(tmp_path))
    version_dir = store_dataset(dataset, tmp_path)
    assert (version_dir / "manifest.json").exists()
    assert version_dir == tmp_path / "hello_world" / "1.0.0"


def test_store_derives_schema_if_needed(tmp_path):
    connector = _DemoConnector()
    dataset = connector.convert(connector.download(tmp_path))
    assert dataset.schema is None
    store_dataset(dataset, tmp_path)  # should derive schema itself
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


def test_clean_cache_keeps_caller_supplied_dir(tmp_path):
    cache = tmp_path / "mine"
    run_pipeline(_DemoConnector(), tmp_path / "root", cache_dir=cache, clean_cache=True)
    assert cache.is_dir()  # a caller-owned cache_dir is never deleted, even with clean_cache


def test_clean_cache_removes_the_auto_created_default(tmp_path, monkeypatch):
    for var in ("TIMENET_STORAGE", "TIMENET_CACHE", "TIMENET_REGISTRY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    run_pipeline(_DemoConnector(), tmp_path / "root", clean_cache=True)  # cache_dir=None -> we own it
    assert not (settings().cache_dir / make_dataset().metadata.dataset_id).exists()
