from pathlib import Path

from typer.testing import CliRunner

from timenet.client import TimeNet
from timenet.testing import assert_datasets_equal
from timenet_connectors.curate.cli import app as curate_app
from timenet_connectors.datasets.timenet.hello_world import HelloWorldConnector
from timenet_connectors.discovery import available, resolve


runner = CliRunner()


def test_discovery_resolves_and_lists():
    assert resolve("timenet/hello-world") is HelloWorldConnector
    assert set(available()) >= {"timenet/hello-world", "chengsenwang/tsqa"}


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
