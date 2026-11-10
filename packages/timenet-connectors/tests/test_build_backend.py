from pathlib import Path
import sys

import pytest

from timenet.errors import TimeNetInvalidCardError
from timenet.manifest import Manifest, PayloadSource
from timenet.manifest.files import FilePart
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.types.references import DatasetRef
from timenet.values_backends import ValuesBackend
from timenet_connectors.builder import backend as backend_module, cli as cli_module
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


def test_a_card_that_cannot_be_read_raises_and_only_a_missing_connector_gives_none(tmp_path, monkeypatch):
    assert ConnectorBuilder().card("nope/nothing") is None
    assert ConnectorBuilder().declared_version("nope/nothing") is None
    (tmp_path / "dataset.yaml").write_text("dataset_id: [unclosed\n")
    monkeypatch.setattr(backend_module, "connector_dir", lambda dataset_id: tmp_path)

    with pytest.raises(TimeNetInvalidCardError, match="could not read dataset card"):
        ConnectorBuilder().card("test/broken")
    with pytest.raises(TimeNetInvalidCardError, match="could not read dataset card"):
        ConnectorBuilder().declared_version("test/broken")


def test_build_runs_in_an_isolated_environment_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("TIMENET_ISOLATION", raising=False)
    calls = []

    def fake_isolated(dataset_id, root, *, force=False, **_):
        calls.append((dataset_id, root, force))
        return Path(root) / "1.0.0"

    monkeypatch.setattr(cli_module, "run_isolated", fake_isolated)

    version_dir = ConnectorBuilder().build("timenet/hello-world", tmp_path, force=True)

    assert version_dir == tmp_path / "1.0.0"
    assert calls == [("timenet/hello-world", tmp_path, True)]


def test_build_hands_its_pins_to_the_build(tmp_path, monkeypatch):
    monkeypatch.delenv("TIMENET_ISOLATION", raising=False)
    received = []

    def fake_isolated(dataset_id, root, *, pinned_by=None, **_):
        received.append(pinned_by)
        return str(Path(root) / "1.0.0")

    monkeypatch.setattr(cli_module, "run_isolated", fake_isolated)
    pinned = FilePart(path="control.duckdb", checksum="sha256:" + "0" * 64, size=1)
    pins = {
        DatasetRef.parse("test/reader@1.0.0"): PayloadSource(
            release=DatasetRef.parse("timenet/hello-world@1.0.0"), files=(pinned,)
        )
    }

    ConnectorBuilder().build("timenet/hello-world", tmp_path, pinned_by=pins)

    assert received == [pins]


def test_build_runs_in_process_when_isolation_is_off(tmp_path, monkeypatch):
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TIMENET_ISOLATION", "off")
    monkeypatch.setattr(cli_module, "run_isolated", _unexpected_isolation)

    version_dir = ConnectorBuilder().build("timenet/hello-world", tmp_path / "registry")

    assert (version_dir / "manifest.json").is_file()


def test_build_in_process_round_trips_the_connector_default_zarr_backend(tmp_path, monkeypatch):
    pytest.importorskip("zarr")
    connector_cls = cli_module.resolve("timenet/hello-world")
    monkeypatch.setattr(connector_cls, "values_backend", ValuesBackend.ZARR)
    monkeypatch.setenv("TIMENET_ISOLATION", "off")
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    version_dir = ConnectorBuilder().build("timenet/hello-world", tmp_path / "registry")
    assert Manifest.model_validate_json((version_dir / "manifest.json").read_text()).values_backend == "zarr"
    with TimeFReader(open_local_version(version_dir)) as reader:
        dataset = reader.read()
        assert len(dataset.records) == 3
        assert len(dataset.records[0].signals[0].to_numpy()) == 16
