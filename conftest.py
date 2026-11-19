"""Give each test an empty TimeNet store, registry and cache."""

import os
from pathlib import Path
import uuid

import pytest

from timenet.config import TimeNetSettings


@pytest.fixture(autouse=True)
def _timenet_state(tmp_path_factory, monkeypatch):
    """Set up an empty store, registry and cache for each test.

    Clear existing ``TIMENET_*`` variables and ignore the ``.env`` file.
    """
    for name in [name for name in os.environ if name.startswith("TIMENET_")]:
        monkeypatch.delenv(name)
    monkeypatch.setitem(TimeNetSettings.model_config, "env_file", None)
    root = tmp_path_factory.mktemp("timenet-state")
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", str(root / "store"))
    monkeypatch.setenv("TIMENET_REMOTE_REGISTRY", str(root / "remote"))
    monkeypatch.setenv("TIMENET_CACHE", str(root / "cache"))


@pytest.fixture
def tmp_path(tmp_path: Path) -> Path:
    """Give each test a folder that no other test of the run uses.

    pytest deletes the folder of a passing test (``tmp_path_retention_policy``) and then gives its
    name to the next test with the same name prefix. A file that the earlier test left open would
    then be found at the later test's path.

    Returns:
        An empty folder inside pytest's own ``tmp_path``, which pytest deletes after the test.
    """
    path = tmp_path / uuid.uuid4().hex[:8]
    path.mkdir()
    return path
