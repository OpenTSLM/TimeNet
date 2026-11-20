"""Shared pytest configuration for the timenet test suite.

Registers Hypothesis profiles so the example count can be tuned per environment via the
``HYPOTHESIS_PROFILE`` environment variable (``dev`` for a fast edit loop, ``ci`` for the full run).
Also gives the tests a fake home folder, and a way to skip on a volume that tells case apart.
"""

import os
from pathlib import Path
import sys

from hypothesis import HealthCheck, settings
import pytest

from timenet.testing import install_backends


# Make the test helpers of this directory (e.g. ``_demo_child``) importable by name under pytest's
# importlib import mode, which does not add test dirs to ``sys.path``.
sys.path.insert(0, str(Path(__file__).parent))

settings.register_profile("dev", max_examples=10)
settings.register_profile("ci", max_examples=100, print_blob=True)
settings.register_profile(
    "nightly",
    max_examples=1000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))


@pytest.fixture
def home(monkeypatch, tmp_path):
    """Point ``HOME`` and ``Path.home()`` at an empty folder, so that no test looks at the real home."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    assert Path("~").expanduser() == Path.home() == home
    return home


@pytest.fixture(autouse=True)
def _no_installed_connectors(monkeypatch):
    """Install no builder backends, so that no test reads the connector packages of the environment."""
    install_backends(monkeypatch)
