from pathlib import Path
import subprocess

import pytest
from typer.testing import CliRunner

from timenet_connectors import datasets
from timenet_connectors.builder import checks
from timenet_connectors.builder.cli import app


runner = CliRunner()


def _connector(root, org, name, *, requirements=False):
    directory = root / org / name
    (directory / "tests").mkdir(parents=True)
    if requirements:
        (directory / "requirements.txt").write_text("numpy\n")
    return directory


@pytest.fixture
def ran(monkeypatch):
    calls = []

    def run(command, *, env, check):
        calls.append((command, env))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(checks.subprocess, "run", run)
    return calls


def test_check_runs_pytest_and_ty_for_each_connector_in_its_overlay(tmp_path, ran, monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/some/venv")
    plain = _connector(tmp_path, "acme", "plain")
    pinned = _connector(tmp_path, "acme", "pinned", requirements=True)
    (tmp_path / "acme" / "no_tests").mkdir()

    result = runner.invoke(app, ["check", str(tmp_path)])

    assert result.exit_code == 0, result.output
    with_requirements = ["uv", "run", "--with-requirements", str(pinned / "requirements.txt")]
    assert [command for command, _ in ran] == [
        [*with_requirements, "pytest", str(pinned / "tests"), "-q"],
        [*with_requirements, "ty", "check", str(pinned)],
        ["uv", "run", "pytest", str(plain / "tests"), "-q"],
        ["uv", "run", "ty", "check", str(plain)],
    ]
    assert all("VIRTUAL_ENV" not in env for _, env in ran)
    assert "=== acme/pinned ===" in result.output
    assert "all connector checks passed" in result.output


def test_check_lists_each_failed_check_and_exits_1(tmp_path, monkeypatch):
    _connector(tmp_path, "acme", "widget")

    def run(command, *, env, check):
        return subprocess.CompletedProcess(command, 1 if "ty" in command else 0)

    monkeypatch.setattr(checks.subprocess, "run", run)

    result = runner.invoke(app, ["check", str(tmp_path)])

    assert result.exit_code == 1
    assert "  - acme/widget: ty" in result.output
    assert "acme/widget: pytest" not in result.output


def test_check_searches_every_given_root(tmp_path, ran):
    first = _connector(tmp_path / "one", "acme", "a")
    second = _connector(tmp_path / "two", "beta", "b")

    result = runner.invoke(app, ["check", str(tmp_path / "one"), str(tmp_path / "two")])

    assert result.exit_code == 0, result.output
    assert [command[-1] for command, _ in ran if command[2] == "ty"] == [str(first), str(second)]


def test_check_without_a_root_searches_the_discovered_roots(tmp_path, ran, monkeypatch):
    connector = _connector(tmp_path, "acme", "widget")
    monkeypatch.setattr(checks, "discovered_roots", lambda: [tmp_path])

    result = runner.invoke(app, ["check"])

    assert result.exit_code == 0, result.output
    assert ["uv", "run", "ty", "check", str(connector)] in [command for command, _ in ran]


def test_the_discovered_roots_start_with_the_built_in_datasets_package():
    assert checks.discovered_roots()[0] == Path(datasets.__path__[0])


def test_check_rejects_a_root_without_connector_tests(tmp_path, ran):
    _connector(tmp_path, "acme", "widget").joinpath("tests").rmdir()

    result = runner.invoke(app, ["check", str(tmp_path)])

    assert result.exit_code == 2
    assert "found no connectors with a tests directory" in result.output
    assert ran == []
