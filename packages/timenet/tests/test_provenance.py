from importlib.metadata import PathDistribution
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from timenet import provenance
from timenet.errors import TimeNetBuildError
from timenet.provenance import build_env


GIT = shutil.which("git") or pytest.fail("the provenance tests need git on PATH")


def test_build_env_records_the_running_python():
    assert build_env()["python"] == ".".join(str(part) for part in sys.version_info[:3])


def test_build_env_records_installed_packages():
    packages = build_env()["packages"]
    assert packages["timenet"]
    assert packages == dict(sorted(packages.items()))


def test_build_env_records_the_commit_of_this_checkout():
    # The dev environment installs timenet editable from this repository.
    head = subprocess.run([GIT, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert build_env()["git"]["timenet"]["sha"] == head


def _git(directory: Path, *args: str) -> None:
    subprocess.run([GIT, "-C", str(directory), *args], capture_output=True, check=True)


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A committed repository with one package directory, ``packages/pkg``."""
    package = tmp_path / "packages" / "pkg"
    package.mkdir(parents=True)
    (package / "module.py").write_text("x = 1\n")
    (tmp_path / "README.md").write_text("readme\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@example.org", "commit", "-qm", "init")
    return package


def test_a_clean_checkout_has_no_local_changes(checkout):
    source = provenance._checkout_source(checkout)
    assert source is not None
    assert len(source["sha"]) == 40
    assert source["local_changes"] is False


@pytest.mark.parametrize("change", ["module.py", "untracked.py"])
def test_a_change_in_the_package_directory_is_a_local_change(checkout, change):
    (checkout / change).write_text("x = 2\n")
    source = provenance._checkout_source(checkout)
    assert source is not None
    assert source["local_changes"] is True


def test_a_change_outside_the_package_directory_is_not_a_local_change(checkout):
    (checkout.parents[1] / "README.md").write_text("edited\n")
    (checkout.parents[1] / "scratch.ipynb").write_text("{}\n")
    source = provenance._checkout_source(checkout)
    assert source is not None
    assert source["local_changes"] is False


def test_a_directory_outside_git_has_no_commit(tmp_path):
    assert provenance._checkout_source(tmp_path) is None


def _wheel_install(site: Path, build_info: str | None) -> PathDistribution:
    """Lay out an installed wheel of ``pkg`` with an optional ``_build_info.json``."""
    dist_info = site / "pkg-1.0.dist-info"
    dist_info.mkdir(parents=True)
    (site / "pkg").mkdir()
    (site / "pkg" / "__init__.py").write_text("")
    record = ["pkg/__init__.py,,", "pkg-1.0.dist-info/RECORD,,"]
    if build_info is not None:
        (site / "pkg" / "_build_info.json").write_text(build_info)
        record.append("pkg/_build_info.json,,")
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: pkg\nVersion: 1.0\n")
    (dist_info / "RECORD").write_text("\n".join(record) + "\n")
    return PathDistribution(dist_info)


def test_a_wheel_reads_its_commit_from_the_build_info(tmp_path):
    info = {"sha": "a" * 40, "local_changes": False}
    assert provenance._wheel_source(_wheel_install(tmp_path, json.dumps(info))) == info


def test_a_wheel_without_build_info_has_no_commit(tmp_path):
    assert provenance._wheel_source(_wheel_install(tmp_path, None)) is None


def test_a_wheel_with_invalid_build_info_fails(tmp_path):
    with pytest.raises(TimeNetBuildError, match=r"invalid _build_info\.json"):
        provenance._wheel_source(_wheel_install(tmp_path, json.dumps({"sha": "main"})))
