from pathlib import Path
import tempfile
from typing import cast

import pytest

from timenet.config import settings
from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset
from timenet.engine import publish_pipeline, run_pipeline, store_dataset
from timenet.errors import TimeFValidationError
from timenet.manifest import Manifest
from timenet.registry.writable import WritableRegistry
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
    assert version_dir == tmp_path / "timenet/hello-world" / "1.0.0"


def test_store_derives_schema_if_needed(tmp_path):
    connector = _DemoConnector()
    dataset = connector.convert(connector.download(tmp_path))
    assert dataset.schema is None
    store_dataset(dataset, tmp_path)  # should derive schema itself
    assert dataset.schema is not None


def test_run_pipeline_end_to_end(tmp_path):
    version_dir = run_pipeline(_DemoConnector(), tmp_path, cache_dir=tmp_path / "cache")
    manifest = Manifest.from_json((version_dir / "manifest.json").read_text())
    assert manifest.counts.records == 3
    assert manifest.dataset_id == "timenet/hello-world"


def test_connector_defaults_to_the_parquet_values_backend():
    assert _DemoConnector().values_backend == "parquet"


def test_run_pipeline_writes_the_requested_values_backend(tmp_path):
    pytest.importorskip("zarr")

    class _ZarrConnector(_DemoConnector):
        values_backend = "zarr"

    connector = _ZarrConnector()
    version_dir = run_pipeline(
        connector,
        tmp_path,
        cache_dir=tmp_path / "cache",
        values_backend=connector.values_backend,
    )

    manifest = Manifest.from_json((version_dir / "manifest.json").read_text())
    assert manifest.values_backend == "zarr"


def test_publish_pipeline_passes_the_requested_values_backend_to_the_registry(tmp_path):
    class _Registry:
        def __init__(self) -> None:
            self.values_backend: str | None = None

        def exists(self, dataset_id: str, version: str) -> bool:
            return False

        def store(self, dataset, *, force: bool, values_backend: str, progress_cb) -> None:
            self.values_backend = values_backend

    class _ZarrConnector(_DemoConnector):
        values_backend = "zarr"

    connector = _ZarrConnector()
    registry = _Registry()
    publish_pipeline(
        connector,
        cast(WritableRegistry, registry),
        cache_dir=tmp_path / "cache",
        values_backend=connector.values_backend,
    )

    assert registry.values_backend == "zarr"


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


@pytest.mark.parametrize("use_default", [False, True])
def test_invalid_backend_preserves_committed_version(tmp_path, use_default):
    connector = _CountingConnector()
    version_dir = run_pipeline(connector, tmp_path, cache_dir=tmp_path / "cache")
    before = {p.relative_to(version_dir): p.read_bytes() for p in version_dir.rglob("*") if p.is_file()}
    if use_default:
        connector.values_backend = "invalid"
    with pytest.raises(TimeFValidationError, match="backend"):
        run_pipeline(
            connector,
            tmp_path,
            cache_dir=tmp_path / "cache",
            force=True,
            values_backend=None if use_default else "invalid",
        )
    assert {p.relative_to(version_dir): p.read_bytes() for p in version_dir.rglob("*") if p.is_file()} == before
    assert connector.downloads == 1


def test_clean_cache_keeps_caller_supplied_dir(tmp_path):
    cache = tmp_path / "mine"
    run_pipeline(_DemoConnector(), tmp_path / "root", cache_dir=cache, keep_cache=False)
    assert cache.is_dir()  # a caller-owned cache_dir is never deleted, even when asked to clean it


def _isolated_home(tmp_path, monkeypatch) -> None:
    """Point the settings at a scratch home so the default cache lands under tmp_path."""
    for var in ("TIMENET_STORAGE", "TIMENET_CACHE", "TIMENET_REGISTRY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))


def test_clean_cache_removes_the_auto_created_default(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    run_pipeline(_DemoConnector(), tmp_path / "root")  # cache_dir=None -> the engine made it
    assert not (settings().cache_dir / make_dataset().metadata.dataset_id).exists()


def test_run_pipeline_keeps_the_cache_when_asked(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    run_pipeline(_DemoConnector(), tmp_path / "root", keep_cache=True)
    assert (settings().cache_dir / make_dataset().metadata.dataset_id).exists()


def test_run_pipeline_keeps_the_cache_when_download_fails(tmp_path, monkeypatch):
    # Downloads resume: http writes a .part and renames it, so a file that landed is complete and the
    # next run skips it. Deleting the cache here would throw away good bytes.
    _isolated_home(tmp_path, monkeypatch)

    class _HalfDownloadedConnector(_DemoConnector):
        def download(self, cache_dir: Path) -> list[str]:
            (cache_dir / "first.bin").write_bytes(b"complete")
            raise RuntimeError("download blew up")

    with pytest.raises(RuntimeError, match="download blew up"):
        run_pipeline(_HalfDownloadedConnector(), tmp_path / "root")
    cache = settings().cache_dir / make_dataset().metadata.dataset_id
    assert (cache / "first.bin").read_bytes() == b"complete"  # the finished file survives the retry


def test_run_pipeline_keeps_the_cache_when_convert_fails(tmp_path, monkeypatch):
    # The rmtree runs after store_dataset, so a convert that raises keeps what download fetched.
    _isolated_home(tmp_path, monkeypatch)

    class _FailingConnector(_DemoConnector):
        def download(self, cache_dir: Path) -> list[str]:
            (cache_dir / "source.bin").write_bytes(b"raw")
            return ["ref"]

        def convert(self, raw_refs: list[str]) -> TimeFDataset:
            raise RuntimeError("convert blew up")

    with pytest.raises(RuntimeError, match="convert blew up"):
        run_pipeline(_FailingConnector(), tmp_path / "root")
    assert (settings().cache_dir / make_dataset().metadata.dataset_id).exists()
