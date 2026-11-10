"""Shared pytest configuration for every workspace package's tests.

Each manifest write reads the commit of the TimeNet distributions. For an editable install this
asks git, so the result would depend on the state of this checkout, and the calls cost seconds
across the suite. So every installed distribution reports a clean commit, whatever its install
type, unless the test carries the ``real_provenance`` marker.
"""

import pytest

from timenet.manifest import GitSource
from timenet.provenance import InstalledDistribution


_CLEAN = GitSource(sha="0" * 40, contains_local_changes=False)


@pytest.fixture(autouse=True)
def _stub_installed_sources(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("real_provenance") is None:
        monkeypatch.setattr(InstalledDistribution, "read_git_source", lambda _self: _CLEAN)
