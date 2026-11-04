"""Give each test an empty TimeNet store, registry and cache."""

import os

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
