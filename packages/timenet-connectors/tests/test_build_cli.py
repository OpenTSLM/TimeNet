from importlib.util import find_spec
import os
from pathlib import Path
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from timenet.manifest import Manifest
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet_connectors.builder import cli as cli_module
from timenet_connectors.builder.cli import app


runner = CliRunner()


def test_build_runs_isolated_by_default(monkeypatch, tmp_path):
    captured = {}

    def _fake_isolated(  # noqa: PLR0913 (mirrors run_isolated's full keyword surface)
        dataset_id, root, *, force=False, keep_cache=False, quiet=False, values_backend=None
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


def test_build_without_a_values_backend_uses_the_connector_default(monkeypatch, tmp_path):
    captured = {}
    instances = []

    class _Connector:
        values_backend = "zarr"

        def __init__(self):
            instances.append(self)

    monkeypatch.setenv("TIMENET_ISOLATION", "off")
    monkeypatch.setattr(cli_module, "resolve", lambda dataset_id: _Connector)

    def _fake_pipeline(connector, root, **kwargs):
        assert connector is instances[0]
        captured["values_backend"] = kwargs["values_backend"]
        return Path(root) / "timenet/hello-world/1.0.0"

    monkeypatch.setattr(cli_module, "run_pipeline", _fake_pipeline)
    result = runner.invoke(app, ["build", "timenet/hello-world", "--out", str(tmp_path)])

    assert result.exit_code == 0
    assert captured["values_backend"] == "zarr"
    assert len(instances) == 1


def test_build_forwards_an_explicit_values_backend_in_process(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setenv("TIMENET_ISOLATION", "off")

    def _fake_pipeline(connector, root, **kwargs):
        captured["values_backend"] = kwargs["values_backend"]
        return Path(root) / "timenet/hello-world/1.0.0"

    monkeypatch.setattr(cli_module, "run_pipeline", _fake_pipeline)
    result = runner.invoke(
        app,
        ["build", "timenet/hello-world", "--out", str(tmp_path), "--values-backend", "zarr"],
    )

    assert result.exit_code == 0
    assert captured["values_backend"] == "zarr"


def test_build_forwards_an_explicit_values_backend_to_an_isolated_child(monkeypatch, tmp_path):
    captured = {}

    def _fake_isolated(  # noqa: PLR0913 (mirrors run_isolated's full keyword surface)
        dataset_id, root, *, force=False, keep_cache=False, quiet=False, values_backend=None
    ):
        captured["values_backend"] = values_backend
        return Path(root) / "timenet/hello-world/1.0.0"

    monkeypatch.setattr(cli_module, "run_isolated", _fake_isolated)
    result = runner.invoke(
        app,
        ["build", "timenet/hello-world", "--out", str(tmp_path), "--values-backend", "zarr"],
    )

    assert result.exit_code == 0
    assert captured["values_backend"] == "zarr"


def test_build_rejects_an_unknown_values_backend(tmp_path):
    result = runner.invoke(
        app,
        ["build", "timenet/hello-world", "--out", str(tmp_path), "--values-backend", "unknown"],
    )

    assert result.exit_code != 0
    assert "values backend" in result.output.lower()


def test_build_isolates_when_the_flag_overrides_the_setting(monkeypatch, tmp_path):
    called = {"isolated": False}
    monkeypatch.setenv("TIMENET_ISOLATION", "off")

    def _fake_isolated(  # noqa: PLR0913 (mirrors run_isolated's full keyword surface)
        dataset_id, root, *, force=False, keep_cache=False, quiet=False, values_backend=None
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


def test_build_forwards_quiet_to_the_isolated_child(monkeypatch, tmp_path):
    captured = {}
    # The console is a process-wide singleton; monkeypatch restores it for the tests that follow.
    monkeypatch.setattr(cli_module.console, "quiet", False)

    def _fake_isolated(  # noqa: PLR0913 (mirrors run_isolated's full keyword surface)
        dataset_id, root, *, force=False, keep_cache=False, quiet=False, values_backend=None
    ):
        captured["quiet"] = quiet
        return Path(root) / "timenet/hello-world/1.0.0"

    monkeypatch.setattr(cli_module, "run_isolated", _fake_isolated)
    result = runner.invoke(app, ["--quiet", "build", "timenet/hello-world", "--out", str(tmp_path)])

    assert result.exit_code == 0
    assert captured["quiet"] is True


@pytest.mark.slow
@pytest.mark.parametrize("backend", [None, "zarr", "parquet"])
def test_isolated_build_prints_the_version_dir_and_narrates_once(tmp_path, backend):
    # run_isolated reads the child's last non-empty stdout line as the version directory, so the
    # real isolated CLI must keep stdout clean and end on that path. Drive it as a subprocess (not
    # CliRunner) so the child's inherited stderr is captured here too, where a double-narration
    # regression would surface.
    registry = tmp_path / "registry"
    env = {**os.environ, "TIMENET_HOME": str(tmp_path / "home")}
    env.pop("TIMENET_ISOLATION", None)  # default is "on"; make sure nothing forces it off
    run_cli = [sys.executable, "-c", "from timenet_connectors.builder.cli import main; main()"]
    backend_args = ["--values-backend", backend] if backend else []
    result = subprocess.run(
        [*run_cli, "build", "timenet/hello-world", "--out", str(registry), *backend_args],
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
    assert Manifest.from_json((version_dir / "manifest.json").read_text()).values_backend == (backend or "parquet")
    if backend != "zarr" or find_spec("zarr") is not None:
        with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
            dataset = reader.read()
            assert len(dataset.records[0].time_series[0].to_numpy()) == 16

    # The isolated child is the sole narrator: each status line appears exactly once.
    assert result.stderr.count("🔧 Building 'timenet/hello-world'…") == 1
    assert result.stderr.count("✅ Built 'timenet/hello-world' → 1.0.0") == 1
