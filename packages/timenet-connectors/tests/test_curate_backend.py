from pathlib import Path
import sys

from timenet_connectors.curate import backend as backend_module
from timenet_connectors.curate.backend import ConnectorCurator


def _unexpected_isolation(*args, **kwargs):
    raise AssertionError(f"run_isolated should not have run: {args} {kwargs}")


def test_knows_accepts_a_known_id():
    assert ConnectorCurator().knows("timenet/hello-world")


def test_knows_rejects_an_unknown_id_without_importing_a_connector():
    leaf = "timenet_connectors.datasets.physionet.ecg_qa_cot"
    sys.modules.pop(leaf, None)

    assert not ConnectorCurator().knows("nope/nothing")

    assert leaf not in sys.modules


def test_build_runs_in_an_isolated_environment_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("TIMENET_ISOLATION", raising=False)
    calls = []

    def fake_isolated(dataset_id, root, *, force=False):
        calls.append((dataset_id, root, force))
        return Path(root) / "1.0.0"

    monkeypatch.setattr(backend_module, "run_isolated", fake_isolated)

    version_dir = ConnectorCurator().build("timenet/hello-world", tmp_path, force=True)

    assert version_dir == tmp_path / "1.0.0"
    assert calls == [("timenet/hello-world", tmp_path, True)]


def test_build_runs_in_process_when_isolation_is_off(tmp_path, monkeypatch):
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TIMENET_ISOLATION", "off")
    monkeypatch.setattr(backend_module, "run_isolated", _unexpected_isolation)

    version_dir = ConnectorCurator().build("timenet/hello-world", tmp_path / "registry")

    assert (version_dir / "manifest.json").is_file()
