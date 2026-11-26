import subprocess

import pytest
from typer.testing import CliRunner

from timenet_connectors.builder import cli
from timenet_connectors.builder.cli import app


runner = CliRunner()


@pytest.fixture(autouse=True)
def _fake_uv_bin(monkeypatch):
    monkeypatch.setattr(cli, "find_uv_bin", lambda: "uv")


def _make_connector(datasets_dir, *parts):
    directory = datasets_dir.joinpath(*parts)
    (directory / "tests").mkdir(parents=True)
    (directory / "dataset.yaml").touch()
    return directory


@pytest.fixture
def subprocess_calls(monkeypatch):
    calls = []

    def run(command, *, check):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", run)
    return calls


def test_check_runs_pytest_and_ty_for_each_connector_with_its_requirements(tmp_path, subprocess_calls):
    flat = _make_connector(tmp_path, "flat")
    pinned = _make_connector(tmp_path, "acme", "pinned")
    (pinned / "requirements.txt").touch()
    (tmp_path / "acme" / "no_tests").mkdir()
    (tmp_path / "acme" / "loose" / "tests").mkdir(parents=True)

    result = runner.invoke(app, ["check", str(tmp_path)])

    assert result.exit_code == 0, result.output
    with_requirements = ["uv", "run", "--with-requirements", str(pinned / "requirements.txt")]
    assert subprocess_calls == [
        [*with_requirements, "pytest", str(pinned / "tests"), "-q"],
        [*with_requirements, "ty", "check", str(pinned)],
        ["uv", "run", "pytest", str(flat / "tests"), "-q"],
        ["uv", "run", "ty", "check", str(flat)],
    ]
    assert "=== acme/pinned ===" in result.output
    assert "all connector checks passed" in result.output


def test_check_lists_each_failed_check_and_exits_1(tmp_path, monkeypatch):
    _make_connector(tmp_path, "acme", "widget")

    def run(command, *, check):
        return subprocess.CompletedProcess(command, 1 if "ty" in command else 0)

    monkeypatch.setattr(cli.subprocess, "run", run)

    result = runner.invoke(app, ["check", str(tmp_path)])

    assert result.exit_code == 1
    assert "  - acme/widget: ty" in result.output
    assert "acme/widget: pytest" not in result.output


@pytest.mark.parametrize("directory", ["", "missing"])
def test_check_exits_2_when_no_connector_has_tests(tmp_path, subprocess_calls, directory):
    _make_connector(tmp_path, "acme", "widget").joinpath("tests").rmdir()

    result = runner.invoke(app, ["check", str(tmp_path / directory)])

    assert result.exit_code == 2
    assert subprocess_calls == []
