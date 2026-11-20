"""Shared pytest configuration for every workspace package's tests.

Each manifest write reads the commit of the TimeNet distributions. For an editable install this
asks git, so the result would depend on the state of this checkout, and the calls cost seconds
across the suite. So every installed distribution reports a clean commit, whatever its install
type, unless the test carries the ``real_provenance`` marker.

Each manifest write also lists the installed packages, which reads the metadata of every
distribution in the environment. The environment does not change during a test run, so the list is
read once here and every write reuses it.
"""

import pytest

from timenet.manifest import GitSource, manifest as manifest_module
from timenet.provenance import InstalledDistribution


_CLEAN = GitSource(sha="0" * 40, contains_local_changes=False)
_INSTALLED_PACKAGES = manifest_module.installed_packages()


@pytest.fixture(autouse=True)
def _stub_installed_sources(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("real_provenance") is None:
        monkeypatch.setattr(InstalledDistribution, "read_git_source", lambda _self: _CLEAN)


@pytest.fixture(autouse=True)
def _reuse_installed_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manifest_module, "installed_packages", lambda: dict(_INSTALLED_PACKAGES))
