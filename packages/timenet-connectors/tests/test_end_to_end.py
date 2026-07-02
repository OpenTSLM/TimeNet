import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from timenet.client import TimeNet
from timenet.testing import assert_datasets_equal
from timenet.types import QATask
from timenet_connectors.curate.cli import app as curate_app
from timenet_connectors.datasets.timenet.hello_world import HelloWorldConnector
from timenet_connectors.discovery import available, resolve


runner = CliRunner()

_TSQA_FIXTURE = Path(__file__).parents[1] / "src/timenet_connectors/datasets/chengsenwang/fixtures/tsqa_sample.json"


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


def test_tsqa_curate_build_then_load_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("TIMENET_TESTING", "1")  # connector serves its offline fixture
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("TIMENET_ROW_LIMIT", raising=False)
    registry = tmp_path / "registry"

    # PRODUCE: curate the (namespaced) TSQA dataset.
    result = runner.invoke(curate_app, ["build", "chengsenwang/tsqa", "--out", str(registry)])
    assert result.exit_code == 0, result.output
    assert (registry / "chengsenwang" / "tsqa" / "1.0.0" / "manifest.json").exists()

    # CONSUME: load it back and verify the Arrow content matches the fixture.
    fixture = json.loads(_TSQA_FIXTURE.read_text())
    dataset = TimeNet(registry, storage_path=tmp_path / "store").load("chengsenwang/tsqa")
    assert len(dataset.samples) == len(fixture)
    qa = [task for task in dataset.tasks if isinstance(task, QATask)]
    assert {task.answer for task in qa} == {row["Answer"] for row in fixture}

    by_id = {sample.sample_id: sample for sample in dataset.samples}
    expected = json.loads(fixture[0]["Series"])
    got = by_id["row-0"].time_series[0].to_numpy()
    assert len(got) == len(expected)
    assert float(got[0]) == pytest.approx(expected[0], rel=1e-5)
