"""Shared pytest configuration for every test suite of the repository.

Each test gets its own empty TimeNet home and no other ``TIMENET_*`` variable of the shell or of a
``.env`` file.
"""

import os

import pytest

from timenet.config import TimeNetSettings


@pytest.fixture(autouse=True)
def _timenet_state(tmp_path_factory, monkeypatch):
    """Give each test an empty ``TIMENET_HOME`` and delete every other ``TIMENET_*`` variable.

    The settings also ignore a ``.env`` file, so no test depends on the shell or the folder it runs in.
    """
    for name in [name for name in os.environ if name.startswith("TIMENET_")]:
        monkeypatch.delenv(name)
    monkeypatch.setitem(TimeNetSettings.model_config, "env_file", None)
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path_factory.mktemp("timenet-home")))
