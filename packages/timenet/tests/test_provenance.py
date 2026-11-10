from importlib.metadata import Distribution, PackageNotFoundError, PathDistribution
import json
import os
from pathlib import Path
import shutil
import subprocess

from pydantic import ValidationError
import pytest

from timenet.errors import TimeNetBuildError
from timenet.manifest import GitSource
from timenet.provenance import InstalledDistribution, PackageSnapshot, TimeNetSnapshot
from timenet.registry import LocalRegistry
from timenet.testing import make_dataset


_CLEAN = GitSource(sha="a" * 40, contains_local_changes=False)


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


@pytest.mark.real_provenance
def test_an_editable_install_records_the_commit_of_its_checkout(git, checkout, installed):
    installed(direct_url=_editable(checkout))
    head = _run(git, checkout, "rev-parse", "HEAD")
    # timenet-connectors is not installed, so the snapshot has no entry for it.
    assert TimeNetSnapshot.read().to_source_map() == {"timenet": GitSource(sha=head, contains_local_changes=False)}


@pytest.mark.real_provenance
@pytest.mark.parametrize("change", ["module.py", "untracked.py"])
def test_a_change_in_the_package_directory_is_a_local_change(checkout, installed, change):
    installed(direct_url=_editable(checkout))
    (checkout / change).write_text("x = 2\n")
    assert PackageSnapshot.read("timenet").source.contains_local_changes is True


@pytest.mark.real_provenance
def test_a_change_outside_the_package_directory_is_not_a_local_change(checkout, installed):
    installed(direct_url=_editable(checkout))
    (checkout.parents[1] / "README.md").write_text("edited\n")
    (checkout.parents[1] / "scratch.ipynb").write_text("{}\n")
    assert PackageSnapshot.read("timenet").source.contains_local_changes is False


@pytest.mark.real_provenance
def test_an_editable_install_outside_git_fails_with_the_git_error(git, tmp_path, installed):
    outside = tmp_path / "outside"
    outside.mkdir()
    installed(direct_url=_editable(outside))
    with pytest.raises(TimeNetBuildError, match=r"cannot find the git commit of timenet.*not a git repository"):
        PackageSnapshot.read("timenet")


@pytest.mark.real_provenance
def test_an_editable_install_without_git_fails(checkout, installed, monkeypatch):
    installed(direct_url=_editable(checkout))
    monkeypatch.setenv("PATH", "")
    with pytest.raises(TimeNetBuildError, match="git is not on PATH"):
        PackageSnapshot.read("timenet")


@pytest.mark.real_provenance
@pytest.mark.parametrize("editable", [True, False])
def test_an_install_of_a_subdirectory_finds_the_package_directory(checkout, installed, editable):
    # direct_url.json may name the repository and put the package in "subdirectory".
    repository = checkout.parents[1]
    installed(
        direct_url={"url": repository.as_uri(), "dir_info": {"editable": editable}, "subdirectory": "packages/pkg"}
    )
    assert InstalledDistribution.from_name("timenet").path_to_ondisk_project_directory() == checkout
    if editable:
        # A change outside the package directory does not count.
        (repository / "README.md").write_text("edited\n")
        assert PackageSnapshot.read("timenet").source.contains_local_changes is False


@pytest.mark.real_provenance
def test_a_path_install_reads_its_build_info_and_not_the_live_checkout(checkout, installed):
    installed(direct_url={"url": checkout.as_uri(), "dir_info": {}}, build_info=_CLEAN.model_dump_json())
    assert PackageSnapshot.read("timenet").source == _CLEAN


@pytest.mark.real_provenance
def test_a_git_install_records_the_commit_that_pip_installed(installed):
    sha = "b" * 40
    installed(direct_url={"url": "https://example.org/timenet.git", "vcs_info": {"vcs": "git", "commit_id": sha}})
    assert PackageSnapshot.read("timenet").source == GitSource(sha=sha, contains_local_changes=False)


@pytest.mark.real_provenance
def test_a_wheel_reads_its_commit_from_the_build_info(installed):
    info = GitSource(sha="a" * 40, contains_local_changes=True)
    installed(build_info=info.model_dump_json())
    assert PackageSnapshot.read("timenet").source == info


@pytest.mark.real_provenance
def test_a_wheel_without_build_info_fails_and_says_how_to_build_one(installed):
    installed()
    with pytest.raises(TimeNetBuildError, match=r"carries no _build_info\.json\. Build it with `make build`"):
        PackageSnapshot.read("timenet")


@pytest.mark.real_provenance
@pytest.mark.parametrize(
    "build_info",
    [
        # The manifest forbids extra keys for every nested block, but this standalone check must
        # forbid them itself.
        json.dumps({"sha": "a" * 40, "contains_local_changes": False, "branch": "main"}),
        "not json",
    ],
)
def test_a_wheel_with_invalid_build_info_fails(installed, build_info):
    installed(build_info=build_info)
    with pytest.raises(TimeNetBuildError, match=r"_build_info\.json"):
        PackageSnapshot.read("timenet")


@pytest.mark.real_provenance
def test_a_local_build_fails_if_a_commit_is_unknown(installed, tmp_path):
    installed()
    with pytest.raises(TimeNetBuildError, match="cannot find the git commit of timenet"):
        LocalRegistry(tmp_path / "local").put_dataset(make_dataset())


def test_a_snapshot_reads_back_from_the_commits_that_a_manifest_records():
    snapshot = TimeNetSnapshot.read()
    assert TimeNetSnapshot.from_source_map(snapshot.to_source_map()) == snapshot


def test_a_manifest_without_timenet_connectors_reads_as_not_installed():
    snapshot = TimeNetSnapshot.from_source_map({"timenet": _CLEAN})
    assert snapshot.timenet_connectors is None
    assert snapshot.to_source_map() == {"timenet": _CLEAN}


def test_a_manifest_without_the_commit_of_timenet_is_refused():
    with pytest.raises(TimeNetBuildError, match="the manifest records no git commit for timenet"):
        TimeNetSnapshot.from_source_map({"timenet-connectors": _CLEAN})


def test_a_field_must_hold_the_commit_of_its_own_package():
    connectors = PackageSnapshot(package_name="timenet-connectors", source=_CLEAN)
    with pytest.raises(ValidationError, match="the field for timenet holds the commit of timenet-connectors"):
        TimeNetSnapshot(timenet=connectors)
