from pathlib import Path

import pytest

from timenet.config import TimeNetSettings, settings
from timenet.errors import TimeFValidationError, TimeNetRegistryError


def _unset(monkeypatch):
    """Delete the variables of the defaults, also the temporary paths that the root ``conftest.py`` sets."""
    for name in ("LOCAL_REGISTRY", "REMOTE_REGISTRY", "AUTO_BUILD", "CACHE"):
        monkeypatch.delenv(f"TIMENET_{name}", raising=False)


@pytest.mark.usefixtures("home")
def test_the_defaults_are_the_store_the_hosted_registry_the_connectors_and_the_cache(monkeypatch):
    _unset(monkeypatch)
    cfg = settings()
    assert cfg.local_registry_dir == Path("~/.timenet/registry").expanduser()
    assert (cfg.remote_registry, cfg.auto_build) == ("timenet://", True)
    assert cfg.cache_dir == Path("~/.cache/timenet").expanduser()
    assert cfg.downloads_dir == cfg.cache_dir / "downloads"
    assert cfg.control_cache_dir == cfg.cache_dir / "control"


def test_each_variable_sets_its_value(monkeypatch, tmp_path):
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", str(tmp_path / "store"))
    monkeypatch.setenv("TIMENET_REMOTE_REGISTRY", "https://registry.example.test")
    monkeypatch.setenv("TIMENET_AUTO_BUILD", "false")
    monkeypatch.setenv("TIMENET_CACHE", str(tmp_path / "cache"))
    cfg = settings()
    assert cfg.local_registry_dir == tmp_path / "store"
    assert (cfg.remote_registry, cfg.auto_build) == ("https://registry.example.test", False)
    assert cfg.cache_dir == tmp_path / "cache"
    assert cfg.downloads_dir == tmp_path / "cache" / "downloads"


def test_the_local_registry_rejects_a_remote_url(monkeypatch):
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", "s3://bucket/store")
    with pytest.raises(TimeNetRegistryError, match="the local registry needs a local directory"):
        settings()


def test_an_explicit_override_beats_the_variable_and_none_lets_it_win(monkeypatch, tmp_path):
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", str(tmp_path / "from-env"))
    monkeypatch.setenv("TIMENET_AUTO_BUILD", "true")
    cfg = settings(local_registry=tmp_path / "from-arg", auto_build=False)
    assert (cfg.local_registry_dir, cfg.auto_build) == (tmp_path / "from-arg", False)
    cfg = settings(local_registry=None, auto_build=None)  # a missing CLI flag falls through to the variable
    assert (cfg.local_registry_dir, cfg.auto_build) == (tmp_path / "from-env", True)


def test_a_path_override_works_for_each_path_setting(tmp_path):
    cfg = settings(local_registry=tmp_path / "store", remote_registry=tmp_path / "remote", cache=tmp_path / "cache")
    assert (cfg.local_registry_dir, cfg.remote_registry, cfg.cache_dir) == (
        tmp_path / "store",
        str(tmp_path / "remote"),
        tmp_path / "cache",
    )


def test_an_invalid_timenet_auto_build_raises(monkeypatch):
    monkeypatch.setenv("TIMENET_AUTO_BUILD", "maybe")
    with pytest.raises(TimeFValidationError, match="TIMENET_AUTO_BUILD"):
        settings()


@pytest.mark.parametrize("value", ["", " "])
@pytest.mark.parametrize("name", ["LOCAL_REGISTRY", "REMOTE_REGISTRY", "CACHE"])
def test_an_empty_or_blank_setting_raises_instead_of_naming_a_directory(monkeypatch, name, value):
    monkeypatch.setenv(f"TIMENET_{name}", value)
    with pytest.raises(TimeFValidationError, match=f"TIMENET_{name} must not be empty"):
        settings()


def test_an_empty_argument_raises_with_the_text_of_its_variable():
    with pytest.raises(
        TimeFValidationError,
        match=r"^TIMENET_REMOTE_REGISTRY must not be empty; give it a value, or leave it unset for timenet://\.$",
    ):
        settings(remote_registry="")


def test_an_invalid_variable_fails_even_when_an_argument_overrides_it(monkeypatch, tmp_path):
    monkeypatch.setenv("TIMENET_CACHE", "")
    with pytest.raises(TimeFValidationError, match="TIMENET_CACHE must not be empty"):
        settings(cache=tmp_path / "cache")


@pytest.mark.parametrize("store", [".", "downloads"])
def test_the_local_registry_must_lie_outside_the_cache(monkeypatch, tmp_path, store):
    monkeypatch.setenv("TIMENET_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", str(tmp_path / "cache" / store))
    with pytest.raises(TimeFValidationError, match=r"TIMENET_LOCAL_REGISTRY .* TIMENET_CACHE .* overlap"):
        settings()


def test_the_cache_must_lie_outside_the_local_registry(monkeypatch, tmp_path):
    # Else the downloads folder of the cache can be the directory of a dataset in the store.
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", str(tmp_path / "store"))
    monkeypatch.setenv("TIMENET_CACHE", str(tmp_path / "store" / "acme"))
    with pytest.raises(TimeFValidationError, match=r"TIMENET_LOCAL_REGISTRY .* TIMENET_CACHE .* overlap"):
        settings()


def test_the_overlap_check_follows_a_symlink(monkeypatch, tmp_path):
    (tmp_path / "store" / "acme").mkdir(parents=True)
    (tmp_path / "link").symlink_to(tmp_path / "store" / "acme", target_is_directory=True)
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", str(tmp_path / "store"))
    monkeypatch.setenv("TIMENET_CACHE", str(tmp_path / "link"))
    with pytest.raises(TimeFValidationError, match="overlap"):
        settings()


@pytest.mark.usefixtures("ignores_case")
def test_the_overlap_check_catches_a_case_variant_on_a_volume_that_ignores_case(monkeypatch, tmp_path):
    (tmp_path / "Cache").mkdir()
    monkeypatch.setenv("TIMENET_CACHE", str(tmp_path / "CACHE"))
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", str(tmp_path / "Cache" / "registry"))
    with pytest.raises(TimeFValidationError, match="overlap"):
        settings()


@pytest.mark.parametrize("cache", ["~", "parent"])
def test_the_cache_must_not_be_the_home_folder_or_a_folder_above_it(monkeypatch, tmp_path, home, cache):
    # With TIMENET_CACHE=~, `timenet local cache clear` would delete ~/downloads, which is ~/Downloads on a
    # volume that ignores case.
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", str(tmp_path / "store"))
    monkeypatch.setenv("TIMENET_CACHE", str(home.parent) if cache == "parent" else cache)
    with pytest.raises(TimeFValidationError, match=r"^TIMENET_CACHE \(.*\) is the home folder or a folder above it"):
        settings()


def test_explicit_paths_work_without_a_home_folder(monkeypatch, tmp_path):
    # For example a container whose user has no home folder.
    def no_home():
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "home", staticmethod(no_home))
    monkeypatch.setenv("TIMENET_CACHE", str(tmp_path / "cache"))
    assert settings().cache_dir == tmp_path / "cache"


def test_is_basesettings():
    assert isinstance(settings(), TimeNetSettings)


def test_isolation_defaults_to_on():
    assert settings().isolation == "on"


def test_isolation_reads_the_environment(monkeypatch):
    monkeypatch.setenv("TIMENET_ISOLATION", "off")
    assert settings().isolation == "off"
