"""Shared pytest configuration for every workspace package's tests.

Each manifest write asks git for the commit of the TimeNet distributions. Real git calls would make
a test's result depend on the state of this checkout, and they cost seconds across the suite. So
every test sees a clean, committed checkout, unless it carries the ``real_git`` marker.
"""

import pytest

from timenet.manifest import GitSource
from timenet.provenance import GitSnapshot, TimeNetDistribution


_CLEAN = GitSnapshot(dict.fromkeys(TimeNetDistribution.NAMES, GitSource(sha="0" * 40, contains_local_changes=False)))


@pytest.fixture(autouse=True)
def _committed_checkout(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("real_git") is None:
        monkeypatch.setattr(GitSnapshot, "read", staticmethod(lambda: _CLEAN))
