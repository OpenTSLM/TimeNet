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
