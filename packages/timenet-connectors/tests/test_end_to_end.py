import json
import platform

import pytest
from typer.testing import CliRunner

from timenet.client import TimeNet
from timenet.config import settings
from timenet.provenance import build_env
from timenet.registry import LocalRegistry
from timenet.sources import InstalledConnectors
from timenet.testing import assert_datasets_equal
from timenet_connectors.builder.cli import app as build_app
from timenet_connectors.datasets.timenet.hello_world import HelloWorldConnector
from timenet_connectors.discovery import available, resolve


runner = CliRunner()


@pytest.fixture
def converted(monkeypatch):
    """Collect every dataset HelloWorldConnector.convert returns, so a test can compare against it."""
    datasets = []
    convert = HelloWorldConnector.convert

    def spy(self, raw_refs):
        dataset = convert(self, raw_refs)
        datasets.append(dataset)
        return dataset

    monkeypatch.setattr(HelloWorldConnector, "convert", spy)
    return datasets


# The CLI tests below pass --no-isolation so a build stays in this process. The isolated path
# shells out to uv and is covered on its own.


def test_discovery_resolves_and_lists():
    assert resolve("timenet/hello-world") is HelloWorldConnector
    assert set(available()) >= {"timenet/hello-world", "chengsenwang/tsqa", "physionet/ecg-qa-cot"}


def test_build_then_load_round_trips(converted, tmp_path):
    registry = tmp_path / "registry"

    # PRODUCE: build the demo dataset into a local registry directory.
    result = runner.invoke(
        build_app, ["build", "timenet/hello-world", "--local-registry", str(registry), "--no-isolation"]
    )
    assert result.exit_code == 0, result.output
    assert (registry / "timenet" / "hello-world" / "1.0.0" / "manifest.json").exists()
    # The flag overrides TIMENET_LOCAL_REGISTRY, which the test setup points at another directory.
    assert not settings().local_registry_dir.exists()

    # CONSUME: load it back through the SDK and compare to the dataset the build wrote.
    restored = TimeNet(sources=[str(registry)]).load("timenet/hello-world")
    [original] = converted
    assert_datasets_equal(original, restored)


# ---- where a bare `build` writes -----------------------------------------------------------------


def test_build_honors_timenet_local_registry(monkeypatch, tmp_path):
    registry = tmp_path / "elsewhere"
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", str(registry))

    assert runner.invoke(build_app, ["build", "timenet/hello-world", "--no-isolation"]).exit_code == 0
    assert (registry / "timenet" / "hello-world" / "1.0.0" / "manifest.json").exists()
    # The store of the SDK is the same directory, so it reads back what the build just wrote. An exact
    # version resolves at the store, without asking the remote registry.
    assert TimeNet().load("timenet/hello-world@1.0.0").metadata.dataset_id == "timenet/hello-world"


def test_build_cleans_cache_but_keep_flag_retains(tmp_path):
    cache = settings().downloads_dir / "timenet" / "hello-world"

    built = runner.invoke(
        build_app, ["build", "timenet/hello-world", "--local-registry", str(tmp_path / "r1"), "--no-isolation"]
    )
    assert built.exit_code == 0
    assert not cache.exists()  # cleaned by default after a successful build

    keep = runner.invoke(
        build_app,
        ["build", "timenet/hello-world", "--local-registry", str(tmp_path / "r2"), "--keep-cache", "--no-isolation"],
    )
    assert keep.exit_code == 0
    assert cache.exists()  # retained with --keep-cache


@pytest.mark.slow
def test_load_builds_a_missing_dataset_in_an_isolated_environment(tmp_path):
    client = TimeNet(sources=[LocalRegistry(tmp_path / "registry", store=True), InstalledConnectors()])

    dataset = client.load("timenet/hello-world")

    assert dataset.metadata.dataset_id == "timenet/hello-world"
    assert (tmp_path / "registry/timenet/hello-world/1.0.0/manifest.json").is_file()


@pytest.mark.slow
def test_an_isolated_build_records_its_environment(tmp_path):
    TimeNet(sources=[LocalRegistry(tmp_path / "registry", store=True), InstalledConnectors()]).load(
        "timenet/hello-world"
    )

    manifest = json.loads((tmp_path / "registry/timenet/hello-world/1.0.0/manifest.json").read_text())
    packages = manifest["build_env"]["packages"]
    assert packages["timenet-connectors"]
    # The child must run on the parent's interpreter, not one uv picked for itself.
    assert manifest["build_env"]["python"] == platform.python_version()
    # pytest runs this process and is a runtime dependency of neither package, so it separates a
    # real isolated build from one that quietly inherited the parent's site-packages. Assert both
    # directions: without the first the second would also pass on a machine that simply lacks it.
    assert "pytest" in build_env()["packages"]
    assert "pytest" not in packages
