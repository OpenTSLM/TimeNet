"""Describe the environment a build ran in, for the manifest's provenance block."""

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import Distribution, PackageNotFoundError, distributions
from pathlib import Path
import platform
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] - the commands are fixed argument vectors
from typing import Self
from urllib.parse import urlparse
from urllib.request import url2pathname

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr, ValidationError

from timenet.errors import TimeNetBuildError
from timenet.manifest import BuildEnvironment, GitSource, GitSourceMap


_GIT_TIMEOUT_S = 30


def build_env() -> BuildEnvironment:
    """Describe this process's Python version, installed packages, and TimeNet commits.

    Under an isolated build the recorded set is exactly the base, the connector's requirements, and
    their transitives. Under ``--no-isolation`` it is the whole ambient environment, which is noisier
    but still the truth about what produced the dataset.

    Returns:
        A dict with ``python`` (the interpreter version), ``packages`` (name -> version, sorted), and
        ``git`` (see :meth:`GitSnapshot.recorded_sources`).
    """
    packages: dict[str, str] = {}
    for dist in distributions():
        name = dist.metadata["Name"]
        if name:
            packages[name] = dist.version
    return {
        "python": platform.python_version(),
        "packages": dict(sorted(packages.items())),
        "git": GitSnapshot.read().recorded_sources(),
    }


@dataclass(frozen=True)
class UnknownGitSource:
    """Why the commit of an installed TimeNet distribution is unknown."""

    reason: str


class TimeNetDistribution:
    """One installed TimeNet distribution, and the commit of the code it runs.

    Each install type finds its commit in its own place:

    - An editable install asks git. ``contains_local_changes`` covers the package directory only,
      so a scratch file elsewhere in the repository does not set it.
    - A git install (``pip install git+https://...``) has the commit that pip recorded.
    - Any other install, such as a wheel or ``pip install ./packages/timenet``, reads the
      ``_build_info.json`` that ``scripts/build_dists.py`` wrote into the package.
    """

    NAMES = ("timenet", "timenet-connectors")
    """The distributions whose git commit a build records."""

    BUILD_INFO_FILE = "_build_info.json"
    """The commit record in each package of a wheel. ``scripts/build_dists.py`` uses the same name."""

    def __init__(self, distribution: Distribution) -> None:
        self._distribution = distribution

    @classmethod
    def installed(cls) -> list[Self]:
        """Return each TimeNet distribution that is installed in this interpreter."""
        installed: list[Self] = []
        for name in cls.NAMES:
            try:
                installed.append(cls(Distribution.from_name(name)))
            except PackageNotFoundError:
                continue
        return installed

    @property
    def name(self) -> str:
        """The distribution name, for example ``timenet-connectors``."""
        return self._distribution.metadata["Name"]

    def path_to_ondisk_project_directory(self) -> Path | None:
        """Return the project directory of an editable or local path install, else ``None``."""
        direct_url = self._read_direct_url()
        return None if direct_url is None else direct_url.local_directory()

    def read_git_source(self) -> GitSource | UnknownGitSource:
        """Return the commit of the code that this distribution runs, or why it is unknown."""
        direct_url = self._read_direct_url()
        if direct_url is None:
            return self._read_build_info_git_source()
        installed_commit = direct_url.installed_git_commit()
        if installed_commit is not None:
            return self._git_source_of_installed_commit(installed_commit)
        project_directory = direct_url.local_directory()
        if direct_url.is_editable() and project_directory is not None:
            return TimeNetDistribution._read_checkout_git_source(project_directory)
        return self._read_build_info_git_source()

    def _read_direct_url(self) -> "_DirectUrl | None":
        """Read the PEP 610 ``direct_url.json``, or return ``None`` for an install from an index."""
        raw = self._distribution.read_text("direct_url.json")
        if raw is None:
            return None
        try:
            return _DirectUrl.model_validate_json(raw)
        except ValidationError as exc:
            raise TimeNetBuildError(f"{self.name} has a malformed direct_url.json: {exc}") from exc

    def _git_source_of_installed_commit(self, sha: str) -> GitSource:
        """Return the clean commit record of a git install, and raise if pip recorded an invalid sha."""
        try:
            return GitSource(sha=sha, contains_local_changes=False)
        except ValidationError as exc:
            raise TimeNetBuildError(f"{self.name} records an invalid commit in direct_url.json: {exc}") from exc

    @staticmethod
    def _read_checkout_git_source(project_directory: Path) -> GitSource | UnknownGitSource:
        """Ask git for the commit of a checkout and whether its package directory has changes."""
        git = shutil.which("git")
        if git is None:
            return UnknownGitSource("git is not on PATH")
        try:
            sha = _run_git(git, project_directory, "rev-parse", "HEAD")
            # The pathspec "." scopes the status to the package directory, untracked files included.
            status = _run_git(git, project_directory, "status", "--porcelain", "--untracked-files=all", "--", ".")
        except subprocess.CalledProcessError as exc:
            return UnknownGitSource(f"git failed in {project_directory}: {(exc.stderr or '').strip() or exc}")
        except (OSError, subprocess.TimeoutExpired) as exc:
            return UnknownGitSource(f"git failed in {project_directory}: {exc}")
        return GitSource(sha=sha, contains_local_changes=bool(status))

    def _read_build_info_git_source(self) -> GitSource | UnknownGitSource:
        """Read the commit that ``scripts/build_dists.py`` wrote into the installed package."""
        module = self.name.replace("-", "_")
        path = Path(str(self._distribution.locate_file(f"{module}/{self.BUILD_INFO_FILE}")))
        if not path.is_file():
            return UnknownGitSource(f"the installed package carries no {self.BUILD_INFO_FILE}")
        try:
            return GitSource.load_from_build_info(path)
        except ValidationError as exc:
            raise TimeNetBuildError(f"{self.name} records an invalid commit in {path}: {exc}") from exc


@dataclass(frozen=True)
class GitSnapshot:
    """The commit of each installed TimeNet distribution, read at one point in time."""

    sources: Mapping[str, GitSource | UnknownGitSource]
    """The commit, or why it is unknown, keyed by distribution name."""

    @classmethod
    def read(cls) -> Self:
        """Read the commit of each installed TimeNet distribution. This is the only call that runs git."""
        return cls(
            {distribution.name: distribution.read_git_source() for distribution in TimeNetDistribution.installed()}
        )

    def recorded_sources(self) -> GitSourceMap:
        """Return the known commits, in the form that a manifest records."""
        return {name: source for name, source in self.sources.items() if isinstance(source, GitSource)}


class _DirInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    editable: StrictBool = False


class _VcsInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    vcs: StrictStr
    commit_id: StrictStr


class _DirectUrl(BaseModel):
    """The PEP 610 record of how a distribution was installed. Keys this module does not read are ignored."""

    model_config = ConfigDict(frozen=True)

    url: StrictStr
    dir_info: _DirInfo | None = None
    vcs_info: _VcsInfo | None = None

    def installed_git_commit(self) -> str | None:
        """Return the commit that pip installed from a git repository, else ``None``."""
        if self.vcs_info is None or self.vcs_info.vcs != "git":
            return None
        return self.vcs_info.commit_id

    def is_editable(self) -> bool:
        """Whether the distribution was installed editable."""
        return self.dir_info is not None and self.dir_info.editable

    def local_directory(self) -> Path | None:
        """Return the directory of an editable or local path install, else ``None``."""
        if not self._is_file_url():
            return None
        path = Path(url2pathname(urlparse(self.url).path))
        return path if self.is_editable() or path.is_dir() else None

    def _is_file_url(self) -> bool:
        return urlparse(self.url).scheme == "file"


def _run_git(git: str, directory: Path, *args: str) -> str:
    """Run one git command in ``directory`` and return its stripped stdout."""
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        # --no-optional-locks keeps `git status` from refreshing the index, so a build never takes
        # the index lock while the user runs git.
        [git, "--no-optional-locks", "-C", str(directory), *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=_GIT_TIMEOUT_S,
    )
    return result.stdout.strip()
