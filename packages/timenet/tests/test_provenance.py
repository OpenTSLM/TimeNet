from importlib.metadata import Distribution, PackageNotFoundError, PathDistribution
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from timenet import provenance
from timenet.errors import TimeNetBuildError
from timenet.manifest import GitSource
from timenet.provenance import UnknownCommit, build_env, git_snapshot, git_sources, require_traceable, require_unchanged


def test_build_env_records_the_running_python():
    assert build_env()["python"] == ".".join(str(part) for part in sys.version_info[:3])


def test_build_env_records_installed_packages():
    packages = build_env()["packages"]
    assert packages["timenet"]
    assert packages == dict(sorted(packages.items()))


@pytest.fixture
def git(monkeypatch: pytest.MonkeyPatch) -> str:
    """The git executable, with the user's and the system's git configuration switched off."""
    executable = shutil.which("git")
    if executable is None:
        pytest.skip("git is not on PATH")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    return executable


def _run(git: str, directory: Path, *args: str) -> str:
    return subprocess.run([git, "-C", str(directory), *args], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def checkout(tmp_path: Path, git: str) -> Path:
    """A committed repository with one package directory, ``packages/pkg``."""
    repo = tmp_path / "repo"
    package = repo / "packages" / "pkg"
    package.mkdir(parents=True)
    (package / "module.py").write_text("x = 1\n")
    (repo / "README.md").write_text("readme\n")
    _run(git, repo, "init", "-q")
    # Hide untracked files from a plain status, so the lookup must ask for them itself.
    _run(git, repo, "config", "status.showUntrackedFiles", "no")
    _run(git, repo, "add", ".")
    _run(git, repo, "-c", "user.name=t", "-c", "user.email=t@example.org", "commit", "-qm", "init")
    return package


def _install(site: Path, *, direct_url: dict | None = None, build_info: str | None = None) -> PathDistribution:
    """Lay out an installed ``timenet`` with an optional ``direct_url.json`` and build info."""
    dist_info = site / "timenet-1.0.dist-info"
    dist_info.mkdir(parents=True)
    (site / "timenet").mkdir()
    record = ["timenet/__init__.py,,", "timenet-1.0.dist-info/RECORD,,"]
    if build_info is not None:
        (site / "timenet" / "_build_info.json").write_text(build_info)
        record.append("timenet/_build_info.json,,")
    if direct_url is not None:
        (dist_info / "direct_url.json").write_text(json.dumps(direct_url))
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: timenet\nVersion: 1.0\n")
    (dist_info / "RECORD").write_text("\n".join(record) + "\n")
    return PathDistribution(dist_info)


@pytest.fixture
def installed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Make a fake ``timenet`` the only installed TimeNet distribution. Call it to lay one out."""

    def install(**kwargs) -> None:
        dist = _install(tmp_path / "site", **kwargs)

        def from_name(name: str) -> Distribution:
            if name != "timenet":
                raise PackageNotFoundError(name)
            return dist

        monkeypatch.setattr(Distribution, "from_name", staticmethod(from_name))

    return install


def _editable(directory: Path) -> dict:
    return {"url": directory.as_uri(), "dir_info": {"editable": True}}


@pytest.mark.real_git
def test_an_editable_install_records_the_commit_of_its_checkout(git, checkout, installed):
    installed(direct_url=_editable(checkout))
    head = _run(git, checkout, "rev-parse", "HEAD")
    assert build_env()["git"] == {"timenet": {"sha": head, "local_changes": False}}


@pytest.mark.real_git
@pytest.mark.parametrize("change", ["module.py", "untracked.py"])
def test_a_change_in_the_package_directory_is_a_local_change(checkout, installed, change):
    installed(direct_url=_editable(checkout))
    (checkout / change).write_text("x = 2\n")
    assert git_sources()["timenet"]["local_changes"] is True


@pytest.mark.real_git
def test_a_change_outside_the_package_directory_is_not_a_local_change(checkout, installed):
    installed(direct_url=_editable(checkout))
    (checkout.parents[1] / "README.md").write_text("edited\n")
    (checkout.parents[1] / "scratch.ipynb").write_text("{}\n")
    assert git_sources()["timenet"]["local_changes"] is False


@pytest.mark.real_git
def test_an_editable_install_outside_git_has_no_commit(git, tmp_path, installed):
    outside = tmp_path / "outside"
    outside.mkdir()
    installed(direct_url=_editable(outside))
    assert git_sources() == {}


@pytest.mark.real_git
def test_a_path_install_reads_its_build_info_and_not_the_live_checkout(checkout, installed):
    info = {"sha": "a" * 40, "local_changes": False}
    installed(direct_url={"url": checkout.as_uri(), "dir_info": {}}, build_info=json.dumps(info))
    assert git_sources() == {"timenet": info}


@pytest.mark.real_git
def test_a_git_install_records_the_commit_that_pip_installed(installed):
    sha = "b" * 40
    installed(direct_url={"url": "https://example.org/timenet.git", "vcs_info": {"vcs": "git", "commit_id": sha}})
    assert git_sources() == {"timenet": {"sha": sha, "local_changes": False}}


@pytest.mark.real_git
def test_a_wheel_reads_its_commit_from_the_build_info(installed):
    info = {"sha": "a" * 40, "local_changes": True}
    installed(build_info=json.dumps(info))
    assert git_sources() == {"timenet": info}


@pytest.mark.real_git
def test_a_wheel_without_build_info_has_no_commit(installed):
    installed()
    assert git_sources() == {}


@pytest.mark.real_git
@pytest.mark.parametrize(
    "build_info",
    [
        # The manifest forbids extra keys for every nested block, but this standalone check must
        # forbid them itself.
        json.dumps({"sha": "a" * 40, "local_changes": False, "branch": "main"}),
        "not json",
    ],
)
def test_a_wheel_with_invalid_build_info_fails(installed, build_info):
    installed(build_info=build_info)
    with pytest.raises(TimeNetBuildError, match=r"_build_info\.json"):
        git_sources()


_CLEAN: GitSource = {"sha": "a" * 40, "local_changes": False}


@pytest.mark.parametrize(
    ("commit", "message"),
    [
        (
            {"sha": "a" * 40, "local_changes": True},
            r"Uncommitted changes in the package directory of timenet at aaaaaaaaaaaa\. Commit them first",
        ),
        (UnknownCommit("git is not on PATH"), r"No known git commit for timenet \(git is not on PATH\)"),
    ],
)
def test_an_untraceable_build_is_refused_with_the_reason(monkeypatch, commit, message):
    monkeypatch.setattr(provenance, "_read_git", lambda: {"timenet": commit, "timenet-connectors": _CLEAN})
    with pytest.raises(TimeNetBuildError, match=message):
        require_traceable()


@pytest.mark.real_git
def test_a_distribution_that_is_not_installed_needs_no_commit(installed):
    # Only timenet is installed. The build needs no commit for timenet-connectors.
    installed(build_info=json.dumps(_CLEAN))
    require_traceable()


def test_a_snapshot_answers_every_lookup_inside_it_from_one_read(monkeypatch):
    reads = iter([{"timenet": _CLEAN}, {"timenet": {"sha": "b" * 40, "local_changes": False}}])
    monkeypatch.setattr(provenance, "_read_git", lambda: next(reads))

    with git_snapshot():
        with git_snapshot():
            assert git_sources() == {"timenet": _CLEAN}
        assert build_env()["git"] == {"timenet": _CLEAN}
    # Outside the snapshot, a lookup reads git again.
    assert git_sources()["timenet"]["sha"] == "b" * 40


def test_an_upload_is_refused_if_the_code_changed_since_the_manifest_was_written(monkeypatch):
    monkeypatch.setattr(provenance, "_read_git", lambda: {"timenet": {"sha": "b" * 40, "local_changes": False}})
    with pytest.raises(TimeNetBuildError, match="changed while the build ran"):
        require_unchanged({"git": {"timenet": _CLEAN}})
    require_unchanged({"git": {"timenet": {"sha": "b" * 40, "local_changes": False}}})
