import pytest

from timenet.errors import TimeNetDatasetNotFoundError, TimeNetRegistryError
from timenet.registry import LocalRegistry, default_registry_path


def test_default_registry_path_falls_back_to_home(monkeypatch, tmp_path):
    monkeypatch.delenv("TIMENET_REGISTRY", raising=False)
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path))
    assert default_registry_path() == tmp_path / "registry"


def test_default_registry_path_honors_local_registry(monkeypatch, tmp_path):
    monkeypatch.setenv("TIMENET_REGISTRY", str(tmp_path / "reg"))
    assert default_registry_path() == tmp_path / "reg"


def test_default_registry_path_rejects_remote(monkeypatch):
    monkeypatch.setenv("TIMENET_REGISTRY", "timenet://")
    with pytest.raises(TimeNetRegistryError):
        default_registry_path()


def test_unknown_dataset_error_points_at_a_method_that_exists(tmp_path):
    # The message used to name `timenet-build build <id>`, a console script this repo no longer
    # ships. Anything it names has to be on the object the caller already holds.
    with pytest.raises(TimeNetDatasetNotFoundError) as caught:
        LocalRegistry(tmp_path).get_manifest("nope/missing")
    assert "store()" in str(caught.value)
    assert hasattr(LocalRegistry(tmp_path), "store")
