import pytest

from timenet.errors import DatasetNotFoundError, RegistryError
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
    with pytest.raises(RegistryError):
        default_registry_path()


def test_unknown_dataset_error_is_actionable(tmp_path):
    with pytest.raises(DatasetNotFoundError, match="Curate one with"):
        LocalRegistry(tmp_path).get_manifest("nope/missing")
