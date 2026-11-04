import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from timenet.connectors import BaseConnector
from timenet.errors import TimeNetDatasetNotFoundError, TimeNetVersionUnavailableError
from timenet.manifest import Manifest, PayloadSource
from timenet.manifest.files import FilePart
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.testing import make_dataset
from timenet.types import DatasetMetadata, Version
from timenet.types.reference_keys import ParentAlias
from timenet.types.references import DatasetRef
from timenet_connectors.builder import backend as backend_module, cli as cli_module
from timenet_connectors.builder.backend import ConnectorBuilder
from timenet_connectors.builder.cli import app


runner = CliRunner()


def test_build_runs_isolated_by_default(monkeypatch, tmp_path):
    captured = {}

    def _fake_isolated(  # ruff: ignore[too-many-arguments] (mirrors run_isolated's full keyword surface)
        dataset_id, root, *, force=False, keep_cache=False, quiet=False, pinned_by=None
    ):
        captured["dataset_id"] = dataset_id
        captured["root"] = root
        return Path(root) / "timenet/hello-world/1.0.0"

    monkeypatch.setattr(cli_module, "run_isolated", _fake_isolated)
    result = runner.invoke(app, ["build", "timenet/hello-world", "--out", str(tmp_path)])

    assert result.exit_code == 0
    assert captured["dataset_id"] == "timenet/hello-world"


def test_build_no_isolation_runs_in_process(monkeypatch, tmp_path):
    called = {"isolated": False, "pipeline": False}

    monkeypatch.setattr(cli_module, "run_isolated", lambda *a, **k: called.__setitem__("isolated", True))

    def _fake_pipeline(connector, root, **kwargs):
        called["pipeline"] = True
        return Path(root) / "timenet/hello-world/1.0.0"

    monkeypatch.setattr(cli_module, "run_pipeline", _fake_pipeline)
    result = runner.invoke(app, ["build", "timenet/hello-world", "--out", str(tmp_path), "--no-isolation"])

    assert result.exit_code == 0
    assert called["pipeline"] is True
    assert called["isolated"] is False


def test_build_runs_in_process_when_the_setting_is_off(monkeypatch, tmp_path):
    called = {"isolated": False, "pipeline": False}
    monkeypatch.setenv("TIMENET_ISOLATION", "off")

    monkeypatch.setattr(cli_module, "run_isolated", lambda *a, **k: called.__setitem__("isolated", True))

    def _fake_pipeline(connector, root, **kwargs):
        called["pipeline"] = True
        return Path(root) / "timenet/hello-world/1.0.0"

    monkeypatch.setattr(cli_module, "run_pipeline", _fake_pipeline)
    result = runner.invoke(app, ["build", "timenet/hello-world", "--out", str(tmp_path)])

    assert result.exit_code == 0
    assert called["pipeline"] is True
    assert called["isolated"] is False


def test_build_isolates_when_the_flag_overrides_the_setting(monkeypatch, tmp_path):
    called = {"isolated": False}
    monkeypatch.setenv("TIMENET_ISOLATION", "off")

    def _fake_isolated(  # ruff: ignore[too-many-arguments] (mirrors run_isolated's full keyword surface)
        dataset_id, root, *, force=False, keep_cache=False, quiet=False, pinned_by=None
    ):
        called["isolated"] = True
        return Path(root) / "timenet/hello-world/1.0.0"

    def _unexpected_pipeline(*args, **kwargs):
        raise AssertionError("an explicit --isolation must win over TIMENET_ISOLATION=off")

    monkeypatch.setattr(cli_module, "run_isolated", _fake_isolated)
    monkeypatch.setattr(cli_module, "run_pipeline", _unexpected_pipeline)
    result = runner.invoke(app, ["build", "timenet/hello-world", "--out", str(tmp_path), "--isolation"])

    assert result.exit_code == 0
    assert called["isolated"] is True


def test_build_takes_an_exact_release_and_refuses_another_version(monkeypatch, tmp_path):
    built = []

    def _fake_isolated(dataset_id, root, **kwargs):
        built.append(dataset_id)
        return Path(root) / "timenet/hello-world/1.0.0"

    monkeypatch.setattr(cli_module, "run_isolated", _fake_isolated)
    result = runner.invoke(app, ["build", "timenet/hello-world@1.0.0", "--out", str(tmp_path)])

    assert result.exit_code == 0
    assert built == ["timenet/hello-world"]
    result = runner.invoke(app, ["build", "timenet/hello-world@9.9.9", "--out", str(tmp_path)])
    assert isinstance(result.exception, TimeNetVersionUnavailableError)
    assert built == ["timenet/hello-world"]


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


def test_build_forwards_quiet_to_the_isolated_child(monkeypatch, tmp_path):
    captured = {}
    # The console is a process-wide singleton; monkeypatch restores it for the tests that follow.
    monkeypatch.setattr(cli_module.console, "quiet", False)

    def _fake_isolated(  # ruff: ignore[too-many-arguments] (mirrors run_isolated's full keyword surface)
        dataset_id, root, *, force=False, keep_cache=False, quiet=False, pinned_by=None
    ):
        captured["quiet"] = quiet
        return Path(root) / "timenet/hello-world/1.0.0"

    monkeypatch.setattr(cli_module, "run_isolated", _fake_isolated)
    result = runner.invoke(app, ["--quiet", "build", "timenet/hello-world", "--out", str(tmp_path)])

    assert result.exit_code == 0
    assert captured["quiet"] is True


@pytest.mark.slow
def test_isolated_build_prints_the_version_dir_and_narrates_once(tmp_path):
    # run_isolated reads the child's last non-empty stdout line as the version directory, so the
    # real isolated CLI must keep stdout clean and end on that path. Drive it as a subprocess (not
    # CliRunner) so the child's inherited stderr is captured here too, where a double-narration
    # regression would surface.
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
        "build_prepared",
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
