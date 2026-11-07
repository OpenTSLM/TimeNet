"""Give each test an empty TimeNet home and no other ``TIMENET_*`` setting."""

import os

import pytest

from timenet.config import TimeNetSettings


@pytest.fixture(autouse=True)
def _timenet_state(tmp_path_factory, monkeypatch):
    """Set an empty temporary TimeNet home and clear other ``TIMENET_*`` variables.

    Ignore settings from ``.env`` files.
    """
    for name in [name for name in os.environ if name.startswith("TIMENET_")]:
        monkeypatch.delenv(name)
    monkeypatch.setitem(TimeNetSettings.model_config, "env_file", None)
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path_factory.mktemp("timenet-home")))
