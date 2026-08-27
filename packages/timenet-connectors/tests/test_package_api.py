from pathlib import Path

import pytest

from timenet.errors import TimeFValidationError
from timenet.types import ClassificationTask
import timenet_connectors
from timenet_connectors import discovery
from timenet_connectors.builder import env as env_module


def _unexpected_isolation(*args, **kwargs):
    raise AssertionError(f"run_isolated should not have run: {args} {kwargs}")


def _unexpected_resolve(*args, **kwargs):
    raise AssertionError(f"resolve should not have run: {args} {kwargs}")


def test_build_then_load_round_trip(monkeypatch, tmp_path):
    # timenet_connectors.build/load anchor on the same registry, so a build here loads back. The
    # consumer default is the hosted service, so point both at a local registry for this round trip.
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path))
    monkeypatch.setenv("TIMENET_REGISTRY", str(tmp_path / "registry"))
    # In-process, so the round trip stays in this interpreter instead of shelling out to uv.
    monkeypatch.setenv("TIMENET_ISOLATION", "off")

    version_dir = timenet_connectors.build("timenet/test-mean")
    assert (version_dir / "manifest.json").exists()

    dataset = timenet_connectors.load("timenet/test-mean")
    assert len(dataset.samples) == 1000

    x, y = dataset.to_features_and_targets(task=ClassificationTask)
    assert len(x) == 1000
    assert set(y.to_pylist()) == {"above_zero", "below_zero"}


def test_build_runs_in_an_isolated_environment_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("TIMENET_ISOLATION", raising=False)
    calls = []

    def fake_isolated(dataset_id, root, *, force=False, keep_cache=False):
        calls.append((dataset_id, root, force, keep_cache))
        return Path(root) / "timenet" / "hello-world" / "1.0.0"

    monkeypatch.setattr(env_module, "run_isolated", fake_isolated)

    version_dir = timenet_connectors.build("timenet/hello-world", out=str(tmp_path), force=True)

    assert calls == [("timenet/hello-world", tmp_path, True, False)]
    assert version_dir == tmp_path / "timenet" / "hello-world" / "1.0.0"


def test_version_guard_does_not_import_the_connector(monkeypatch, tmp_path):
    # The guard runs before the isolated build, so importing the connector here would need the very
    # dependencies that only the child environment has.
    monkeypatch.delenv("TIMENET_ISOLATION", raising=False)
    monkeypatch.setattr(discovery, "resolve", _unexpected_resolve)
    monkeypatch.setattr(env_module, "run_isolated", lambda dataset_id, root, **kwargs: Path(root) / "built")

    version_dir = timenet_connectors.build("timenet/hello-world", version="1.0.0", out=str(tmp_path))

    assert version_dir == tmp_path / "built"


def test_version_mismatch_is_rejected_without_importing_the_connector(monkeypatch, tmp_path):
    monkeypatch.delenv("TIMENET_ISOLATION", raising=False)
    monkeypatch.setattr(discovery, "resolve", _unexpected_resolve)
    monkeypatch.setattr(env_module, "run_isolated", _unexpected_isolation)

    with pytest.raises(TimeFValidationError, match=r"builds version 1\.0\.0"):
        timenet_connectors.build("timenet/hello-world", version="9.9.9", out=str(tmp_path))


def test_build_runs_in_process_when_isolation_is_off(monkeypatch, tmp_path):
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TIMENET_ISOLATION", "off")
    monkeypatch.setattr(env_module, "run_isolated", _unexpected_isolation)

    version_dir = timenet_connectors.build("timenet/hello-world", out=tmp_path / "registry")

    assert version_dir == tmp_path / "registry" / "timenet" / "hello-world" / "1.0.0"
    assert (version_dir / "manifest.json").is_file()
