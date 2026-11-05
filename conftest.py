"""Configure pytest with isolated TimeNet state.

Each test gets its own empty store, remote registry and cache, and no other ``TIMENET_*`` variable
of the shell or of a ``.env`` file.
"""

import os

import pytest

from timenet.config import TimeNetSettings


@pytest.fixture(autouse=True)
def _timenet_state(tmp_path_factory, monkeypatch):
    """Give each test an empty store, remote registry and cache, and delete every other ``TIMENET_*`` variable.

    Ignore settings from ``.env`` files.
    """
    for name in [name for name in os.environ if name.startswith("TIMENET_")]:
        monkeypatch.delenv(name)
    monkeypatch.setitem(TimeNetSettings.model_config, "env_file", None)
    root = tmp_path_factory.mktemp("timenet-state")
    monkeypatch.setenv("TIMENET_LOCAL_REGISTRY", str(root / "store"))
    monkeypatch.setenv("TIMENET_REMOTE_REGISTRY", str(root / "remote"))
    monkeypatch.setenv("TIMENET_CACHE", str(root / "cache"))
