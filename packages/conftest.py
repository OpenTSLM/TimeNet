"""Shared pytest configuration for every workspace package's tests.

Each manifest write asks git for the commit of the TimeNet distributions. Real git calls would make
a test's result depend on the state of this checkout, and they cost seconds across the suite. So
every test sees a clean, committed checkout, unless it carries the ``real_git`` marker.
"""

import pytest

from timenet import provenance


_CLEAN = {"sha": "0" * 40, "local_changes": False}


@pytest.fixture(autouse=True)
def _committed_checkout(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("real_git") is None:
        monkeypatch.setattr(provenance, "_read_git", lambda: dict.fromkeys(provenance.TIMENET_DISTRIBUTIONS, _CLEAN))
