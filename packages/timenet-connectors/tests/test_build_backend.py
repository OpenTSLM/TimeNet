from pathlib import Path
import sys

import pytest

from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet_connectors.builder import backend as backend_module
from timenet_connectors.builder.backend import ConnectorBuilder


def _unexpected_isolation(*args, **kwargs):
    raise AssertionError(f"run_isolated should not have run: {args} {kwargs}")


def test_knows_accepts_a_known_id():
    assert ConnectorBuilder().knows("timenet/hello-world")


def test_knows_rejects_an_unknown_id_without_importing_a_connector():
    leaf = "timenet_connectors.datasets.physionet.ecg_qa_cot"
    sys.modules.pop(leaf, None)

    assert not ConnectorBuilder().knows("nope/nothing")

    assert leaf not in sys.modules


@pytest.mark.parametrize("values_backend", [None, "zarr", "parquet"])
def test_build_runs_in_an_isolated_environment_by_default(tmp_path, monkeypatch, values_backend):
    monkeypatch.delenv("TIMENET_ISOLATION", raising=False)
    calls = []

    def fake_isolated(dataset_id, root, *, force=False, values_backend=None):
        calls.append((dataset_id, root, force, values_backend))
        return Path(root) / "1.0.0"

    monkeypatch.setattr(backend_module, "run_isolated", fake_isolated)

    version_dir = ConnectorBuilder().build("timenet/hello-world", tmp_path, force=True, values_backend=values_backend)

    assert version_dir == tmp_path / "1.0.0"
    assert calls == [("timenet/hello-world", tmp_path, True, values_backend)]


def test_build_runs_in_process_when_isolation_is_off(tmp_path, monkeypatch):
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TIMENET_ISOLATION", "off")
    monkeypatch.setattr(backend_module, "run_isolated", _unexpected_isolation)

    version_dir = ConnectorBuilder().build("timenet/hello-world", tmp_path / "registry")

    assert (version_dir / "manifest.json").is_file()


def test_build_in_process_round_trips_the_connector_default_zarr_backend(tmp_path, monkeypatch):
    pytest.importorskip("zarr")
    connector_cls = backend_module.resolve("timenet/hello-world")
    monkeypatch.setattr(connector_cls, "values_backend", "zarr")
    monkeypatch.setenv("TIMENET_ISOLATION", "off")
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    version_dir = ConnectorBuilder().build("timenet/hello-world", tmp_path / "registry")
    assert Manifest.from_json((version_dir / "manifest.json").read_text()).values_backend == "zarr"
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        dataset = reader.read()
        assert len(dataset.records) == 3
        assert len(dataset.records[0].time_series[0].to_numpy()) == 16


def test_build_in_process_uses_the_connector_default_values_backend(tmp_path, monkeypatch):
    class Connector:
        values_backend = "zarr"

    captured = {}
    monkeypatch.setenv("TIMENET_ISOLATION", "off")
    monkeypatch.setattr(backend_module, "resolve", lambda dataset_id: Connector)

    def pipeline(connector, root, *, force, values_backend):
        captured["backend"] = values_backend
        return Path(root) / "1.0.0"

    monkeypatch.setattr(backend_module, "run_pipeline", pipeline)
    ConnectorBuilder().build("timenet/hello-world", tmp_path)
    assert captured["backend"] == "zarr"
