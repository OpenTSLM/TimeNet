import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from timenet.errors import CurationError
from timenet_connectors.curate import env as env_module
from timenet_connectors.curate.env import EnvSpec, env_spec, run_isolated, uv_command


def _spec(requirements=None, base=("--with", "timenet==0.1.0")):
    return EnvSpec(base=tuple(base), requirements=requirements, python="/usr/bin/python3")


class _FakeDist:
    """Stand-in for an installed distribution with a chosen ``direct_url.json``."""

    def __init__(self, version="9.9.9", direct_url=None):
        self.version = version
        self._direct_url = direct_url

    def read_text(self, name):
        return self._direct_url if name == "direct_url.json" else None


def _base_args_for(monkeypatch, dist):
    monkeypatch.setattr(env_module, "BASE_DISTRIBUTIONS", ("timenet",))
    monkeypatch.setattr(env_module.Distribution, "from_name", lambda name: dist)
    return env_module._base_args()


def test_uv_command_runs_without_the_project_environment():
    command = uv_command(_spec(), ["timenet-curate", "build", "x/y"])
    assert "--no-project" in command


def test_uv_command_pins_the_parent_interpreter():
    command = uv_command(_spec(), ["timenet-curate"])
    assert command[command.index("--python") + 1] == "/usr/bin/python3"


def test_uv_command_omits_requirements_when_the_connector_declares_none():
    assert "--with-requirements" not in uv_command(_spec(), ["timenet-curate"])


def test_uv_command_passes_the_requirements_file():
    command = uv_command(_spec(requirements=Path("/fake/reqs.txt")), ["timenet-curate"])
    assert command[command.index("--with-requirements") + 1] == "/fake/reqs.txt"


def test_uv_command_puts_the_child_argv_last():
    argv = ["timenet-curate", "build", "x/y", "--out", "/fake/reg"]
    assert uv_command(_spec(), argv)[-len(argv) :] == argv


def test_env_spec_keeps_the_running_python_version():
    probe = "import sys; print('%s.%s' % sys.version_info[:2])"
    result = subprocess.run(
        [env_spec("timenet/hello-world").python, "-c", probe], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == f"{sys.version_info.major}.{sys.version_info.minor}"


def test_env_spec_steps_out_of_a_virtualenv():
    # uv reads a venv --python as the *base* of the ephemeral env and layers --with over it, so a
    # venv interpreter here leaks the parent's site-packages into the child.
    probe = "import sys; print(sys.prefix == sys.base_prefix)"
    result = subprocess.run(
        [env_spec("timenet/hello-world").python, "-c", probe], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "True"


def test_env_spec_uses_editable_paths_in_a_source_checkout():
    # The dev environment installs both packages editable from the workspace.
    base = env_spec("timenet/hello-world").base
    assert "--with-editable" in base
    assert any(arg.endswith("packages/timenet-connectors") for arg in base)


def test_base_args_uses_editable_for_an_editable_install(monkeypatch, tmp_path):
    src = tmp_path / "src"
    direct_url = json.dumps({"url": src.as_uri(), "dir_info": {"editable": True}})

    assert _base_args_for(monkeypatch, _FakeDist(direct_url=direct_url)) == ("--with-editable", str(src))


def test_base_args_uses_editable_for_a_local_path_install(monkeypatch, tmp_path):
    # `pip install ./packages/timenet` records a file:// dir with no editable flag.
    src = tmp_path / "pkg"
    src.mkdir()
    direct_url = json.dumps({"url": src.as_uri(), "dir_info": {}})

    assert _base_args_for(monkeypatch, _FakeDist(direct_url=direct_url)) == ("--with-editable", str(src))


def test_base_args_pins_a_version_for_an_index_install(monkeypatch):
    # An index or wheel install has no direct_url.json, so pin the exact version.
    assert _base_args_for(monkeypatch, _FakeDist(version="9.9.9", direct_url=None)) == ("--with", "timenet==9.9.9")


def test_run_isolated_disables_isolation_in_the_child(monkeypatch, tmp_path):
    captured = {}

    def _fake(command, env):
        captured["env"] = env
        return f"{tmp_path}/timenet/hello-world/1.0.0\n", "", 0

    monkeypatch.setattr(env_module, "_run_curate", _fake)
    run_isolated("timenet/hello-world", tmp_path)

    assert captured["env"]["TIMENET_ISOLATION"] == "off"


def test_run_isolated_returns_the_version_directory(monkeypatch, tmp_path):
    stdout = f"noise\n{tmp_path}/timenet/hello-world/1.0.0\n"
    monkeypatch.setattr(env_module, "_run_curate", lambda command, env: (stdout, "", 0))

    assert run_isolated("timenet/hello-world", tmp_path) == tmp_path / "timenet/hello-world/1.0.0"


def test_run_isolated_raises_on_a_failed_child(monkeypatch, tmp_path):
    monkeypatch.setattr(env_module, "_run_curate", lambda command, env: ("", "", 2))

    with pytest.raises(CurationError, match="exit code 2"):
        run_isolated("timenet/hello-world", tmp_path)


def test_run_isolated_failure_message_includes_the_child_stderr(monkeypatch, tmp_path):
    monkeypatch.setattr(env_module, "_run_curate", lambda command, env: ("", "boom: the disk is full", 1))

    with pytest.raises(CurationError, match="the disk is full"):
        run_isolated("timenet/hello-world", tmp_path)


def test_run_isolated_raises_when_the_child_prints_no_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(env_module, "_run_curate", lambda command, env: ("\n", "", 0))

    with pytest.raises(CurationError, match="no version directory"):
        run_isolated("timenet/hello-world", tmp_path)


def test_run_isolated_forwards_the_build_flags(monkeypatch, tmp_path):
    captured = {}

    def _fake(command, env):
        captured["command"] = command
        return f"{tmp_path}/x\n", "", 0

    monkeypatch.setattr(env_module, "_run_curate", _fake)
    run_isolated("timenet/hello-world", tmp_path, force=True, keep_cache=True)

    assert "--force" in captured["command"]
    assert "--keep-cache" in captured["command"]


def test_run_isolated_forwards_quiet_ahead_of_the_subcommand(monkeypatch, tmp_path):
    captured = {}

    def _fake(command, env):
        captured["command"] = command
        return f"{tmp_path}/x\n", "", 0

    monkeypatch.setattr(env_module, "_run_curate", _fake)
    run_isolated("timenet/hello-world", tmp_path, quiet=True)

    command = captured["command"]
    # --quiet is a root option on timenet-curate, so build never sees it.
    assert command.index("--quiet") == command.index("build") - 1


def test_run_isolated_omits_quiet_when_the_parent_is_not_quiet(monkeypatch, tmp_path):
    captured = {}

    def _fake(command, env):
        captured["command"] = command
        return f"{tmp_path}/x\n", "", 0

    monkeypatch.setattr(env_module, "_run_curate", _fake)
    run_isolated("timenet/hello-world", tmp_path)

    assert "--quiet" not in captured["command"]


def test_run_curate_captures_stdout_and_stderr_and_the_exit_code():
    command = [sys.executable, "-c", "import sys; print('out'); print('an error', file=sys.stderr); sys.exit(3)"]
    stdout, stderr, code = env_module._run_curate(command, dict(os.environ))
    assert stdout.strip() == "out"
    assert "an error" in stderr
    assert code == 3


def test_env_spec_reports_a_base_distribution_that_is_not_installed(monkeypatch):
    monkeypatch.setattr(env_module, "BASE_DISTRIBUTIONS", ("timenet", "timenet-not-installed"))

    with pytest.raises(CurationError, match="timenet-not-installed"):
        env_spec("timenet/hello-world")


def test_uv_command_reports_a_missing_uv_binary(monkeypatch):
    def _missing():
        raise FileNotFoundError("uv")

    monkeypatch.setattr(env_module, "find_uv_bin", _missing)

    with pytest.raises(CurationError, match="uv"):
        uv_command(_spec(), ["timenet-curate"])
