import json
import os
import subprocess
import sys

import pytest
from rich.console import Console
from typer.testing import CliRunner

from timenet.connectors import BaseConnector
from timenet.errors import TimeNetDatasetNotFoundError, TimeNetVersionUnavailableError
from timenet.manifest import Manifest, PayloadSource
from timenet.manifest.files import FilePart
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, open_local_version
from timenet.testing import make_dataset
from timenet.types import DatasetMetadata, Version
from timenet.types.reference_keys import ParentAlias
from timenet.types.references import DatasetRef
from timenet_connectors.builder import backend as backend_module, cli as cli_module
from timenet_connectors.builder.backend import ConnectorBuilder
from timenet_connectors.builder.cli import app


runner = CliRunner()
HELLO = DatasetRef.parse("timenet/hello-world@1.0.0")


def _isolated_child(calls):
    """Return a stand-in for ``run_isolated_build`` that stores the demo dataset and records each call."""

    def run(release, store_root, *, force=False, quiet=False, **_):
        calls.append({"release": release, "store_root": store_root, "quiet": quiet})
        LocalRegistry(store_root).put_dataset(make_dataset(), force=force)

    return run


def _in_process(calls):
    """Return a stand-in for ``run_pipeline`` that stores the demo dataset and records each call."""

    def run(connector, root, **kwargs):
        calls.append({"connector": connector, **kwargs})
        LocalRegistry(root).put_dataset(make_dataset())

    return run


def _unexpected(*args, **kwargs):
    raise AssertionError(f"should not have run: {args} {kwargs}")


def test_build_runs_isolated_by_default(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(backend_module, "run_isolated_build", _isolated_child(calls))

    result = runner.invoke(app, ["build", "timenet/hello-world", "--out", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert [(call["release"], call["store_root"]) for call in calls] == [(HELLO, tmp_path)]
    assert result.stdout == f"{tmp_path / 'timenet/hello-world/1.0.0'}\n"


@pytest.mark.parametrize("how", ["flag", "setting"])
def test_build_runs_in_process_without_isolation(monkeypatch, tmp_path, how):
    calls = []
    monkeypatch.setattr(backend_module, "run_isolated_build", _unexpected)
    monkeypatch.setattr(backend_module, "run_pipeline", _in_process(calls))
    if how == "setting":
        monkeypatch.setenv("TIMENET_ISOLATION", "off")

    result = runner.invoke(
        app, ["build", "timenet/hello-world", "--out", str(tmp_path), *(["--no-isolation"] if how == "flag" else [])]
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1


def test_build_isolates_when_the_flag_overrides_the_setting(monkeypatch, tmp_path):
    monkeypatch.setenv("TIMENET_ISOLATION", "off")
    calls = []
    monkeypatch.setattr(backend_module, "run_isolated_build", _isolated_child(calls))

    result = runner.invoke(app, ["build", "timenet/hello-world", "--out", str(tmp_path), "--isolation"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1


def test_build_without_isolation_shows_each_build_line_once(monkeypatch, tmp_path):
    # In this process, the coordinator sends the build step, and the renderer shows it. Nothing else
    # prints a build line.
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))

    result = runner.invoke(app, ["build", "timenet/hello-world", "--out", str(tmp_path / "out"), "--no-isolation"])

    assert result.exit_code == 0, result.output
    assert result.stderr.count("🔧 Building 'timenet/hello-world'…") == 1
    assert result.stderr.count("✅ Built 'timenet/hello-world' → 1.0.0") == 1


def test_a_build_that_reuses_the_release_says_so_on_stderr(monkeypatch, tmp_path):
    # The line names a long path, so the console must not wrap it.
    monkeypatch.setattr(cli_module.console, "rich", Console(stderr=True, width=1000))
    calls = []
    monkeypatch.setattr(backend_module, "run_isolated_build", _isolated_child(calls))
    command = ["build", "timenet/hello-world", "--out", str(tmp_path)]
    assert runner.invoke(app, command).exit_code == 0

    result = runner.invoke(app, command)

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert result.stdout == f"{tmp_path / 'timenet/hello-world/1.0.0'}\n"
    reused = f"✅ Nothing to do: 'timenet/hello-world' 1.0.0 is already in local registry {tmp_path}"
    assert result.stderr.splitlines() == [reused]


def test_build_takes_an_exact_release_and_refuses_another_version(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(backend_module, "run_isolated_build", _isolated_child(calls))
    result = runner.invoke(app, ["build", "timenet/hello-world@1.0.0", "--out", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert [call["release"] for call in calls] == [HELLO]
    result = runner.invoke(app, ["build", "timenet/hello-world@9.9.9", "--out", str(tmp_path)])
    assert isinstance(result.exception, TimeNetVersionUnavailableError)
    assert len(calls) == 1


def test_build_rejects_an_unknown_id_without_importing_anything(monkeypatch, tmp_path):
    # A connector's dependencies live in its own environment, not this one, so importing one here
    # to answer "which ids exist?" would raise ModuleNotFoundError instead of the unknown-id error.
    prefix = "timenet_connectors.datasets."
    for name in [name for name in sys.modules if name.startswith(prefix)]:
        # delitem, not del: teardown puts the original module objects back, so a test that compares
        # a connector class by identity still sees the same object.
        monkeypatch.delitem(sys.modules, name)

    result = runner.invoke(app, ["build", "nope/nothing", "--out", str(tmp_path)])

    assert result.exit_code != 0
    assert "no connector for" in result.output
    connectors = [n for n in sys.modules if n.startswith(prefix) and n.count(".") >= 3]
    assert connectors == []


def test_build_rejects_an_id_that_no_card_declares_and_names_the_known_ids(tmp_path):
    # The module path of the connector accepts this spelling, but a dataset ID must match its card.
    result = runner.invoke(app, ["build", "timenet/hello_world", "--out", str(tmp_path)])

    assert result.exit_code == 2
    assert "no connector for 'timenet/hello_world'" in result.output
    assert "timenet/hello-world" in result.output


@pytest.mark.parametrize("quiet", [False, True])
def test_build_forwards_quiet_to_the_isolated_child(monkeypatch, tmp_path, quiet):
    # The console is a process-wide singleton; monkeypatch restores it for the tests that follow.
    monkeypatch.setattr(cli_module.console, "quiet", False)
    calls = []
    monkeypatch.setattr(backend_module, "run_isolated_build", _isolated_child(calls))

    result = runner.invoke(
        app, [*(["--quiet"] if quiet else []), "build", "timenet/hello-world", "--out", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert [call["quiet"] for call in calls] == [quiet]


@pytest.mark.slow
def test_isolated_build_prints_the_version_dir_and_narrates_once(tmp_path):
    # A script reads the last non-empty stdout line as the version directory, so the real isolated
    # CLI must keep stdout clean and end on that path. Drive it as a subprocess (not CliRunner) so
    # the child's inherited stderr is captured here too, where a double-narration regression would
    # surface: the parent and the child both see the build step.
    registry = tmp_path / "registry"
    env = {**os.environ, "TIMENET_HOME": str(tmp_path / "home")}
    env.pop("TIMENET_ISOLATION", None)  # default is "on"; make sure nothing forces it off
    run_cli = [sys.executable, "-c", "from timenet_connectors.builder.cli import main; main()"]
    result = subprocess.run(
        [*run_cli, "build", "timenet/hello-world", "--out", str(registry)],
        env=env,
        cwd=tmp_path,  # away from the repo tree so the CLI does not read the repo .env
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    version_dir = registry / "timenet" / "hello-world" / "1.0.0"
    stdout_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert stdout_lines[-1] == str(version_dir)
    assert (version_dir / "manifest.json").is_file()
    assert Manifest.model_validate_json((version_dir / "manifest.json").read_text()).values_backend == "parquet"
    with TimeFReader(open_local_version(version_dir)) as reader:
        dataset = reader.read()
        assert len(dataset.records[0].signals[0].to_numpy()) == 16

    # The isolated child is the sole narrator: each status line appears exactly once.
    assert result.stderr.count("🔧 Building 'timenet/hello-world'…") == 1
    assert result.stderr.count("✅ Built 'timenet/hello-world' → 1.0.0") == 1


_PARENT = DatasetRef.parse("test/parent@1.0.0")


def _demo_card(dataset_id: str, **update) -> DatasetMetadata:
    """Return the card of the demo dataset under ``dataset_id``, at version 1.0.0."""
    return make_dataset().metadata.model_copy(
        update={"dataset_id": dataset_id, "dataset_version": Version(1, 0, 0), **update}
    )


class _Parent(BaseConnector[str]):
    """Builds the demo dataset as test/parent."""

    def metadata(self):
        return _demo_card(_PARENT.dataset_id)

    def download(self, cache_dir):
        return []

    def convert(self, raw_refs):
        dataset = make_dataset()
        dataset._metadata = self.metadata()
        return dataset


class _Child(_Parent):
    """Builds the demo dataset as test/child, which declares test/parent as its parent."""

    def metadata(self):
        return _demo_card("test/child", parents={ParentAlias("parent"): _PARENT})

    def compose(self, raw_refs, context):
        return self.convert(raw_refs)


def test_the_build_prepared_release_command_does_not_plan(monkeypatch, tmp_path):
    # The installed connectors build test/child and its parent test/parent, in this process.
    connectors = {"test/child": _Child, _PARENT.dataset_id: _Parent}
    monkeypatch.setattr(ConnectorBuilder, "dataset_ids", lambda self: list(connectors))
    monkeypatch.setattr(ConnectorBuilder, "card", lambda self, dataset_id: connectors[dataset_id]().metadata())
    monkeypatch.setattr(backend_module, "resolve", connectors.__getitem__)
    monkeypatch.setenv("TIMENET_ISOLATION", "off")
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cli_module.console, "quiet", False)

    # The store lacks the parent. A planner would build it first. The command must fail on the
    # missing parent instead.
    result = runner.invoke(app, ["build-prepared-release", "test/child@1.0.0", "--store", str(tmp_path / "store")])

    assert result.exit_code == 1
    assert isinstance(result.exception, TimeNetDatasetNotFoundError), result.exception
    assert str(result.exception).startswith(f"complete release {_PARENT} is missing")
    assert not (tmp_path / "store" / _PARENT.dataset_id).exists()


def test_the_build_prepared_release_command_builds_a_release_without_parents(monkeypatch, tmp_path):
    monkeypatch.setenv("TIMENET_ISOLATION", "off")
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cli_module.console, "quiet", False)

    result = runner.invoke(
        app, ["build-prepared-release", "timenet/hello-world@1.0.0", "--store", str(tmp_path / "store")]
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "store/timenet/hello-world/1.0.0/manifest.json").is_file()
    assert result.stdout == ""


@pytest.mark.parametrize("quiet", [False, True])
def test_the_build_prepared_release_command_passes_the_pins_and_the_options(monkeypatch, tmp_path, quiet):
    release = DatasetRef.parse("timenet/hello-world@1.0.0")
    pinned = PayloadSource(release=release, files=(FilePart(path="a.bin", checksum="sha256:" + "0" * 64, size=1),))
    pins_file = tmp_path / "pins.json"
    pins_file.write_text(json.dumps({"test/reader@1.0.0": pinned.model_dump(mode="json")}), encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        ConnectorBuilder,
        "build",
        lambda self, release, store, **kwargs: calls.append((release, store, kwargs)),
    )
    monkeypatch.setattr(cli_module.console, "quiet", False)
    options = ["--store", str(tmp_path), "--force", "--keep-cache", "--pinned-by", str(pins_file)]

    result = runner.invoke(app, [*(["--quiet"] if quiet else []), "build-prepared-release", str(release), *options])

    assert result.exit_code == 0, result.output
    ((built, store, kwargs),) = calls
    assert (built, store.root, kwargs["force"]) == (release, tmp_path, True)
    assert kwargs["pinned_by"] == {DatasetRef.parse("test/reader@1.0.0"): pinned}
    assert kwargs["options"].keep_cache is True
    assert (kwargs["options"].progress is None) is quiet
    assert kwargs["options"].quiet is quiet
