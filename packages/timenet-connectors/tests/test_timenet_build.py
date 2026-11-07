import pytest

import timenet
from timenet.client import TimeNet
from timenet.errors import TimeNetVersionUnavailableError
from timenet.registry import LocalRegistry
from timenet.testing import make_dataset
from timenet.types import ClassificationTask
from timenet.types.references import DatasetRef
from timenet_connectors.builder import backend as backend_module


HELLO = DatasetRef.parse("timenet/hello-world@1.0.0")


def _unexpected_isolation(*args, **kwargs):
    raise AssertionError(f"run_isolated_build should not have run: {args} {kwargs}")


def _unexpected_resolve(*args, **kwargs):
    raise AssertionError(f"resolve should not have run: {args} {kwargs}")


def test_build_then_load_round_trip(monkeypatch, tmp_path):
    # In-process, so the round trip stays in this interpreter instead of shelling out to uv.
    monkeypatch.setenv("TIMENET_ISOLATION", "off")
    monkeypatch.setattr(backend_module, "run_isolated_build", _unexpected_isolation)
    out = tmp_path / "out"

    plan = timenet.build("timenet/test-mean", local_registry=out)
    assert [(step.release, step.action) for step in plan.steps] == [
        (DatasetRef.parse("timenet/test-mean@1.0.0"), "build")
    ]

    dataset = TimeNet(sources=[str(out)]).load("timenet/test-mean")
    assert len(dataset.records) == 1000

    x, y = dataset.to_features_and_targets(task=ClassificationTask)
    assert len(x) == 1000
    assert set(y.to_pylist()) == {"above_zero", "below_zero"}


def test_build_runs_in_an_isolated_environment_by_default(monkeypatch, tmp_path):
    # The parent reads only the card. It does not import the connector, which needs dependencies
    # that only the environment of the child has.
    monkeypatch.setattr(backend_module, "resolve", _unexpected_resolve)
    calls = []

    def fake_isolated(release, store_root, *, force=False, keep_cache=False, quiet=False, **_):
        calls.append((release, store_root, force, keep_cache, quiet))
        LocalRegistry(store_root).put_dataset(make_dataset())

    monkeypatch.setattr(backend_module, "run_isolated_build", fake_isolated)

    plan = timenet.build("timenet/hello-world", local_registry=tmp_path / "out", force=True)

    # The child is not quiet: it shows its downloads and shards.
    assert calls == [(HELLO, tmp_path / "out", True, False, False)]
    (step,) = plan.steps
    assert isinstance(step, timenet.BuildStep)
    assert (step.release, step.force) == (HELLO, True)


def test_build_rejects_a_version_the_connector_does_not_declare(monkeypatch, tmp_path):
    monkeypatch.setattr(backend_module, "resolve", _unexpected_resolve)
    monkeypatch.setattr(backend_module, "run_isolated_build", _unexpected_isolation)

    with pytest.raises(
        TimeNetVersionUnavailableError, match=r"builds version 1\.0\.0, not the requested 9\.9\.9"
    ) as caught:
        timenet.build("timenet/hello-world@9.9.9", local_registry=tmp_path / "out")

    assert str(caught.value.declared) == "1.0.0"
    assert not (tmp_path / "out").exists()
