from pathlib import Path

import pytest

from timenet.config import TimeNetSettings, settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("TIMENET_HOME", "TIMENET_STORAGE", "TIMENET_CACHE", "TIMENET_REGISTRY"):
        monkeypatch.delenv(var, raising=False)


def test_defaults_under_home():
    cfg = settings(home=Path("/base"))
    assert cfg.home_dir == Path("/base")
    assert cfg.storage_dir == Path("/base/storage")
    assert cfg.cache_dir == Path("/base/cache")
    assert cfg.registry_path == Path("/base/registry")


def test_default_home_is_cache_timenet():
    assert settings().home_dir == Path("~/.cache/timenet").expanduser()


def test_home_env_relocates_everything(monkeypatch):
    monkeypatch.setenv("TIMENET_HOME", "/relocated")
    cfg = settings()
    assert cfg.storage_dir == Path("/relocated/storage")
    assert cfg.cache_dir == Path("/relocated/cache")
    assert cfg.registry_path == Path("/relocated/registry")


def test_specific_env_overrides_only_its_path(monkeypatch):
    monkeypatch.setenv("TIMENET_HOME", "/base")
    monkeypatch.setenv("TIMENET_STORAGE", "/elsewhere/storage")
    cfg = settings()
    assert cfg.storage_dir == Path("/elsewhere/storage")
    assert cfg.cache_dir == Path("/base/cache")  # still under home


def test_explicit_override_beats_env(monkeypatch):
    monkeypatch.setenv("TIMENET_STORAGE", "/from/env")
    cfg = settings(storage=Path("/from/arg"))
    assert cfg.storage_dir == Path("/from/arg")


def test_settings_drops_none_so_env_wins(monkeypatch):
    monkeypatch.setenv("TIMENET_STORAGE", "/from/env")
    cfg = settings(storage=None)  # a missing CLI flag must fall through to env
    assert cfg.storage_dir == Path("/from/env")


def test_registry_selector_defaults_none():
    assert settings(home=Path("/base")).registry is None


def test_registry_selector_from_env(monkeypatch):
    monkeypatch.setenv("TIMENET_REGISTRY", "https://reg.example.io")
    assert settings().registry == "https://reg.example.io"


def test_is_basesettings():
    assert isinstance(settings(), TimeNetSettings)
