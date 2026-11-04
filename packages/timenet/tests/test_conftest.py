"""Tests of the root ``conftest.py``: each test gets its own state."""

import os
from pathlib import Path
import subprocess
import sys

from timenet.config import settings


pytest_plugins = ["pytester"]

ROOT_CONFTEST = Path(__file__).resolve().parents[3] / "conftest.py"

PATHS = ("registry_path", "storage_dir", "cache_dir", "control_cache_dir")
"""The settings properties that point at local state."""


def test_each_test_starts_without_the_timenet_variables_of_the_shell(pytester, monkeypatch):
    for name, value in {
        "TIMENET_REGISTRY": "timenet://",
        "TIMENET_STORAGE": "/shell/storage",
        "TIMENET_CACHE": "/shell/cache",
        "TIMENET_TOKEN": "secret",
        "TIMENET_ISOLATION": "off",
    }.items():
        monkeypatch.setenv(name, value)
    pytester.makeconftest(ROOT_CONFTEST.read_text())
    pytester.makepyfile(
        """
        import os

        def test_only_the_home_is_set():
            names = sorted(name for name in os.environ if name.startswith("TIMENET_"))
            assert names == ["TIMENET_HOME"]
        """
    )
    pytester.runpytest().assert_outcomes(passed=1)


def test_every_settings_path_lies_in_the_home_of_the_test_also_in_a_child_process(tmp_path):
    home = Path(os.environ["TIMENET_HOME"])
    in_process = [getattr(settings(), name) for name in PATHS]
    code = f"from timenet.config import settings; print(*(getattr(settings(), name) for name in {PATHS}), sep='\\n')"
    child = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, check=True)
    in_child = [Path(line) for line in child.stdout.splitlines()]
    assert len(in_child) == len(in_process) == 4
    assert all(path.is_relative_to(home) for path in [*in_process, *in_child])


def test_a_dotenv_file_in_the_working_directory_does_not_change_the_settings(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("TIMENET_REGISTRY=timenet://\nTIMENET_TOKEN=secret\nTIMENET_ISOLATION=off\n")
    monkeypatch.chdir(tmp_path)
    cfg = settings()
    assert (cfg.registry, cfg.token, cfg.isolation) == (None, None, "on")
