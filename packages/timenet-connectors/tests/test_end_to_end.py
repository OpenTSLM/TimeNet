from pathlib import Path

import pytest
from typer.testing import CliRunner

from timenet.client import TimeNet
from timenet.config import settings
from timenet.engine import run_pipeline
from timenet.testing import assert_datasets_equal
from timenet_connectors.curate.cli import app as curate_app
from timenet_connectors.datasets.timenet.hello_world import HelloWorldConnector
from timenet_connectors.discovery import available, resolve


runner = CliRunner()


def test_discovery_resolves_and_lists():
    assert resolve("timenet/hello-world") is HelloWorldConnector
    assert set(available()) >= {"timenet/hello-world", "chengsenwang/tsqa", "physionet/ecg-qa-cot"}


def test_curate_build_then_load_round_trips(tmp_path):
    registry = tmp_path / "registry"

    # PRODUCE: curate the demo dataset into a local registry directory.
    result = runner.invoke(curate_app, ["build", "timenet/hello-world", "--out", str(registry)])
    assert result.exit_code == 0, result.output
    assert (registry / "timenet" / "hello-world" / "1.0.0" / "manifest.json").exists()

    # CONSUME: load it back through the SDK and compare to a fresh conversion.
    restored = TimeNet(registry, storage_path=tmp_path / "store").load("timenet/hello-world")
    connector = HelloWorldConnector()
    original = connector.convert(connector.download(Path("cache")))
    assert_datasets_equal(original, restored)


def test_curate_build_unknown_id_fails(tmp_path):
    result = runner.invoke(curate_app, ["build", "acme/not_a_dataset", "--out", str(tmp_path / "registry")])
    assert result.exit_code != 0


# ---- where a bare `build` writes -----------------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """Isolate the TIMENET_* vars so a bare build's output root is fully determined by the test."""
    for var in ("TIMENET_STORAGE", "TIMENET_CACHE", "TIMENET_REGISTRY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))


def test_curate_build_defaults_to_home_registry(clean_env, tmp_path):
    assert runner.invoke(curate_app, ["build", "timenet/hello-world"]).exit_code == 0
    assert (tmp_path / "home" / "registry" / "timenet" / "hello-world" / "1.0.0" / "manifest.json").exists()


def test_curate_build_honors_timenet_registry(clean_env, monkeypatch, tmp_path):
    registry = tmp_path / "elsewhere"
    monkeypatch.setenv("TIMENET_REGISTRY", str(registry))

    assert runner.invoke(curate_app, ["build", "timenet/hello-world"]).exit_code == 0
    assert (registry / "timenet" / "hello-world" / "1.0.0" / "manifest.json").exists()
    # The SDK resolves $TIMENET_REGISTRY the same way, so it reads back what the build just wrote.
    assert TimeNet(storage_path=tmp_path / "store").list()[0].dataset_id == "timenet/hello-world"


def test_curate_build_out_overrides_timenet_registry(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv("TIMENET_REGISTRY", str(tmp_path / "elsewhere"))
    out = tmp_path / "out"

    assert runner.invoke(curate_app, ["build", "timenet/hello-world", "--out", str(out)]).exit_code == 0
    assert (out / "timenet" / "hello-world" / "1.0.0" / "manifest.json").exists()
    assert not (tmp_path / "elsewhere").exists()


def test_curate_build_rejects_remote_timenet_registry(clean_env, monkeypatch):
    monkeypatch.setenv("TIMENET_REGISTRY", "timenet://")
    result = runner.invoke(curate_app, ["build", "timenet/hello-world"])
    assert result.exit_code == 2
    assert "--out" in result.output


def test_run_pipeline_cleans_cache_when_requested(tmp_path, monkeypatch):
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    version_dir = run_pipeline(HelloWorldConnector(), tmp_path / "registry", clean_cache=True)
    assert (version_dir / "manifest.json").exists()  # build succeeded
    assert not (settings().cache_dir / "timenet" / "hello-world").exists()  # raw cache removed


def test_run_pipeline_keeps_cache_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    run_pipeline(HelloWorldConnector(), tmp_path / "registry")
    assert (settings().cache_dir / "timenet" / "hello-world").exists()


def test_curate_build_cleans_cache_but_keep_flag_retains(tmp_path, monkeypatch):
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    cache = settings().cache_dir / "timenet" / "hello-world"

    assert runner.invoke(curate_app, ["build", "timenet/hello-world", "--out", str(tmp_path / "r1")]).exit_code == 0
    assert not cache.exists()  # cleaned by default after a successful build

    keep = runner.invoke(curate_app, ["build", "timenet/hello-world", "--out", str(tmp_path / "r2"), "--keep-cache"])
    assert keep.exit_code == 0
    assert cache.exists()  # retained with --keep-cache
