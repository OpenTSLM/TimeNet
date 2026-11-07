import pytest

from timenet.errors import TimeNetVersionUnavailableError
from timenet.registry import LocalRegistry
from timenet.testing import make_dataset
from timenet.types import ClassificationTask
from timenet.types.references import DatasetRef
import timenet_connectors
from timenet_connectors.builder import backend as backend_module


def _unexpected_isolation(*args, **kwargs):
    raise AssertionError(f"run_isolated_build should not have run: {args} {kwargs}")


def _unexpected_resolve(*args, **kwargs):
    raise AssertionError(f"resolve should not have run: {args} {kwargs}")


def test_build_then_load_round_trip(monkeypatch, tmp_path):
    # timenet_connectors.build/load anchor on the same registry, so a build here loads back. The
    # consumer default is the hosted service, so point both at a local registry for this round trip.
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path))
    monkeypatch.setenv("TIMENET_REGISTRY", str(tmp_path / "registry"))
    # In-process, so the round trip stays in this interpreter instead of shelling out to uv.
    monkeypatch.setenv("TIMENET_ISOLATION", "off")
    monkeypatch.setattr(backend_module, "run_isolated_build", _unexpected_isolation)

    version_dir = timenet_connectors.build("timenet/test-mean")
    assert (version_dir / "manifest.json").exists()

    dataset = timenet_connectors.load("timenet/test-mean")
    assert len(dataset.records) == 1000

    x, y = dataset.to_features_and_targets(task=ClassificationTask)
    assert len(x) == 1000
    assert set(y.to_pylist()) == {"above_zero", "below_zero"}


def test_build_runs_in_an_isolated_environment_by_default(monkeypatch, tmp_path):
    # The parent reads only the card. It does not import the connector, which needs dependencies
    # that only the environment of the child has.
    monkeypatch.delenv("TIMENET_ISOLATION", raising=False)
    monkeypatch.setattr(backend_module, "resolve", _unexpected_resolve)
    calls = []

    def fake_isolated(release, store_root, *, force=False, keep_cache=False, quiet=False, **_):
        calls.append((release, store_root, force, keep_cache, quiet))
        LocalRegistry(store_root).put_dataset(make_dataset())

    monkeypatch.setattr(backend_module, "run_isolated_build", fake_isolated)

    version_dir = timenet_connectors.build("timenet/hello-world", out=str(tmp_path), force=True)

    # The child is not quiet: it shows its downloads and shards.
    assert calls == [(DatasetRef.parse("timenet/hello-world@1.0.0"), tmp_path, True, False, False)]
    assert version_dir == tmp_path / "timenet" / "hello-world" / "1.0.0"


def test_build_rejects_a_version_the_connector_does_not_declare(monkeypatch, tmp_path):
    monkeypatch.delenv("TIMENET_ISOLATION", raising=False)
    monkeypatch.setattr(backend_module, "resolve", _unexpected_resolve)
    monkeypatch.setattr(backend_module, "run_isolated_build", _unexpected_isolation)

    with pytest.raises(
        TimeNetVersionUnavailableError, match=r"builds version 1\.0\.0, not the requested 9\.9\.9"
    ) as caught:
        timenet_connectors.build("timenet/hello-world", version="9.9.9", out=tmp_path)

    assert str(caught.value.declared) == "1.0.0"
    assert not tmp_path.joinpath("timenet").exists()
