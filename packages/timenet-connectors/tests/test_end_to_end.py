import json
import platform

import pytest
from typer.testing import CliRunner

from timenet.client import TimeNet
from timenet.config import settings
from timenet.errors import TimeFValidationError, TimeNetRegistryError
from timenet.provenance import build_env
from timenet.registry import LocalRegistry, RemoteRegistry
from timenet.sources import InstalledConnectors
from timenet.testing import assert_datasets_equal
from timenet_connectors import build, load
from timenet_connectors.builder.cli import _resolve_target, app as build_app
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
    result = runner.invoke(build_app, ["build", "timenet/hello-world", "--out", str(registry), "--no-isolation"])
    assert result.exit_code == 0, result.output
    assert (registry / "timenet" / "hello-world" / "1.0.0" / "manifest.json").exists()

    # CONSUME: load it back through the SDK and compare to the dataset the build wrote.
    restored = TimeNet(sources=[str(registry)]).load("timenet/hello-world")
    [original] = converted
    assert_datasets_equal(original, restored)


def test_build_unknown_id_fails(tmp_path):
    result = runner.invoke(build_app, ["build", "acme/not_a_dataset", "--out", str(tmp_path / "registry")])
    assert result.exit_code != 0


def test_build_rejects_version_mismatch():
    # The guard fires before the pipeline runs, so no registry is written.
    with pytest.raises(TimeFValidationError, match="builds version"):
        build("timenet/hello-world", version="9.9.9")


def test_load_rejects_version_mismatch():
    with pytest.raises(TimeFValidationError, match="builds version"):
        load("timenet/hello-world", "9.9.9")


# ---- where a bare `build` writes -----------------------------------------------------------------


def test_build_honors_timenet_local_registry(monkeypatch, tmp_path):
    registry = tmp_path / "elsewhere"
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", str(registry))

    assert runner.invoke(build_app, ["build", "timenet/hello-world", "--no-isolation"]).exit_code == 0
    assert (registry / "timenet" / "hello-world" / "1.0.0" / "manifest.json").exists()
    # The store of the SDK is the same directory, so it reads back what the build just wrote. An exact
    # version resolves at the store, without asking the remote registry.
    assert TimeNet().load("timenet/hello-world@1.0.0").metadata.dataset_id == "timenet/hello-world"


def test_build_out_overrides_timenet_local_registry(monkeypatch, tmp_path):
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", str(tmp_path / "elsewhere"))
    out = tmp_path / "out"

    result = runner.invoke(build_app, ["build", "timenet/hello-world", "--out", str(out), "--no-isolation"])
    assert result.exit_code == 0
    assert (out / "timenet" / "hello-world" / "1.0.0" / "manifest.json").exists()
    assert not (tmp_path / "elsewhere").exists()


def test_resolve_target_routes_remote_url_to_registry(tmp_path):
    # A remote --out publishes through a writable registry (constructed offline, no network).
    assert isinstance(_resolve_target("timenet://"), RemoteRegistry)
    assert isinstance(_resolve_target("https://registry.timenet.ai"), RemoteRegistry)
    # A local --out writes a directory.
    assert _resolve_target(str(tmp_path / "reg")) == tmp_path / "reg"


def test_resolve_target_rejects_an_empty_out_instead_of_the_current_directory():
    with pytest.raises(TimeNetRegistryError, match=r"^--out is blank"):
        _resolve_target("")


def test_sdk_build_cleans_cache_but_keep_cache_retains(tmp_path, monkeypatch):
    monkeypatch.setenv("TIMENET_ISOLATION", "off")  # the in-process path, the one that kept the cache
    cache = settings().downloads_dir / "timenet" / "hello-world"

    build("timenet/hello-world", out=tmp_path / "r1")
    assert not cache.exists()

    build("timenet/hello-world", out=tmp_path / "r2", keep_cache=True)
    assert cache.exists()


def test_build_cleans_cache_but_keep_flag_retains(tmp_path):
    cache = settings().downloads_dir / "timenet" / "hello-world"

    built = runner.invoke(build_app, ["build", "timenet/hello-world", "--out", str(tmp_path / "r1"), "--no-isolation"])
    assert built.exit_code == 0
    assert not cache.exists()  # cleaned by default after a successful build

    keep = runner.invoke(
        build_app, ["build", "timenet/hello-world", "--out", str(tmp_path / "r2"), "--keep-cache", "--no-isolation"]
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
