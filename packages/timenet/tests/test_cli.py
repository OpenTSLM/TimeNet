import importlib
import re
import sys

import pytest
from typer.testing import CliRunner

from timenet.cli import app
from timenet.errors import DatasetNotFoundError
from timenet.testing import make_dataset
from timenet.writer import TimeFWriter


runner = CliRunner()
# The package re-exports the Typer object as ``timenet.cli.app``, shadowing the submodule; fetch the
# actual module so we can patch its module-global ``app`` when testing main()'s error handling.
_cli_module = importlib.import_module("timenet.cli.app")


@pytest.fixture
def registry_root(tmp_path):
    dataset = make_dataset()
    dataset.derive_schema()
    with TimeFWriter(tmp_path / "reg", dataset) as writer:
        writer.write()
    return tmp_path / "reg"


@pytest.fixture
def home(tmp_path, monkeypatch):
    for var in ("TIMENET_STORAGE", "TIMENET_CACHE", "TIMENET_REGISTRY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    dataset = make_dataset()
    dataset.derive_schema()
    with TimeFWriter(tmp_path / "home" / "registry", dataset) as writer:
        writer.write()
    return tmp_path / "home"


def test_cache_info_lists_datasets(home):
    result = runner.invoke(app, ["cache", "info"])
    assert result.exit_code == 0
    assert "hello_world" in result.stdout
    assert "registry" in result.stdout


def test_cache_info_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "empty"))
    result = runner.invoke(app, ["cache", "info"])
    assert result.exit_code == 0
    assert "No cached datasets" in result.stdout


def test_cache_info_shows_raw_size_without_path(home):
    cache_dir = home / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "blob").write_bytes(b"x" * 10)
    result = runner.invoke(app, ["cache", "info"])
    assert result.exit_code == 0
    assert "Raw download cache:" in result.stdout
    assert str(cache_dir) not in result.stdout  # size only, no noisy absolute path


def test_cache_clear_needs_confirmation(home):
    result = runner.invoke(app, ["cache", "clear"], input="n\n")
    assert result.exit_code != 0  # aborted
    assert (home / "registry").exists()


def test_cache_clear_all_removes_everything(home):
    result = runner.invoke(app, ["cache", "clear", "--all", "--yes"])
    assert result.exit_code == 0
    assert "Freed" in result.stdout
    assert not (home / "registry").exists()


def test_list(registry_root):
    result = runner.invoke(app, ["list", "--registry", str(registry_root)])
    assert result.exit_code == 0
    assert "hello_world" in result.stdout


def test_search_by_domain(registry_root):
    result = runner.invoke(app, ["search", "--registry", str(registry_root), "--domain", "general"])
    assert result.exit_code == 0
    assert "hello_world" in result.stdout


def test_search_no_match(registry_root):
    result = runner.invoke(app, ["search", "--registry", str(registry_root), "--domain", "cardiology"])
    assert result.exit_code == 0
    assert "hello_world" not in result.stdout


def test_info(registry_root):
    result = runner.invoke(app, ["info", "hello_world", "--registry", str(registry_root)])
    assert result.exit_code == 0
    assert "hello_world" in result.stdout
    assert "samples" in result.stdout.lower()


def test_download(registry_root, tmp_path):
    result = runner.invoke(
        app,
        ["download", "hello_world", "--registry", str(registry_root), "--storage", str(tmp_path / "store")],
    )
    assert result.exit_code == 0
    assert (tmp_path / "store" / "hello_world" / "1.0.0" / "manifest.json").exists()


def test_search_rejects_unknown_filter_value(registry_root):
    result = runner.invoke(app, ["search", "--registry", str(registry_root), "--domain", "bogus"])
    assert result.exit_code != 0
    # Strip ANSI: Rich colorizes the ``--domain`` token, so the message isn't a plain substring
    # when color is enabled (e.g. in CI).
    clean = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "invalid --domain" in clean


def test_main_reports_expected_errors_without_traceback(monkeypatch, capsys):
    def raise_expected() -> None:
        raise DatasetNotFoundError("no such dataset")

    monkeypatch.setattr(_cli_module, "app", raise_expected)
    monkeypatch.setattr(sys, "argv", ["timenet"])
    with pytest.raises(SystemExit) as exit_info:
        _cli_module.main()
    assert exit_info.value.code == 1
    assert "no such dataset" in capsys.readouterr().err
