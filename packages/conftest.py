"""Shared pytest configuration for every workspace package's tests.

Each manifest write asks git for the commit of the TimeNet distributions. Real git calls would make
a test's result depend on the state of this checkout, and they cost seconds across the suite. So
every installed distribution reports a clean commit, unless the test carries the ``real_git`` marker.
"""

import pytest

from timenet.manifest import GitSource
from timenet.provenance import InstalledDistribution


_CLEAN = GitSource(sha="0" * 40, contains_local_changes=False)


@pytest.fixture(autouse=True)
def _committed_checkout(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("real_git") is None:
        monkeypatch.setattr(InstalledDistribution, "read_git_source", lambda _self: _CLEAN)
