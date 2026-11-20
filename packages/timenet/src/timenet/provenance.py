"""Read the git commit of the installed TimeNet code, for the manifest's ``build_env.git`` block."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.metadata import Distribution, PackageNotFoundError
from pathlib import Path
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] - the commands are fixed argument vectors
import sys
from typing import ClassVar, Self
from urllib.parse import urlparse
from urllib.request import url2pathname

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, ValidationError, model_validator

from timenet.errors import TimeNetBuildError
from timenet.manifest import GitSource, GitSourceMap, PackageName


_GIT_TIMEOUT_S = 30


@dataclass(frozen=True)
class InstallSource(ABC):
    """Where one package was installed from: a git checkout, a git URL, or a wheel.

    Each kind records its git provenance in a different place.
    """

    package_name: PackageName

    @abstractmethod
    def read_git_source(self) -> GitSource:
        """Return the commit of the code that this install runs. Raises ``TimeNetBuildError`` if it is unknown."""

    def _unknown_commit(self, reason: str) -> TimeNetBuildError:
        return TimeNetBuildError(
            f"cannot find the git commit of {self.package_name}, which each build records in its manifest: {reason}"
        )


@dataclass(frozen=True)
class EditableInstall(InstallSource):
    """An editable install (``pip install -e``, or a uv workspace member).

    Python imports the code from the checkout itself, so git reports its commit and local changes.
    """

    project_directory: Path

    def read_git_source(self) -> GitSource:
        """Ask git for the commit of the checkout and whether the package directory has changes."""
        checkout = _GitCheckout(self.project_directory)
        try:
            # Read the commit after the status. A commit between the two reads then shows as a new
            # commit, not as the old commit without local changes.
            changed = checkout.has_local_changes()
            return GitSource(sha=checkout.head_commit(), contains_local_changes=changed)
        except _GitNotFoundError as exc:
            raise self._unknown_commit(
                f"it is installed editable from {self.project_directory}, and {exc}. Install git, "
                f"or install {self.package_name} from a wheel that scripts/build_dists.py built"
            ) from exc
        except _GitError as exc:
            raise self._unknown_commit(str(exc)) from exc


@dataclass(frozen=True)
class GitInstall(InstallSource):
    """An install from a git URL (``pip install git+https://...``). Pip recorded the commit."""

    commit_id: str

    def read_git_source(self) -> GitSource:
        """Return the commit that pip installed, which has no local changes."""
        try:
            return GitSource(sha=self.commit_id, contains_local_changes=False)
        except ValidationError as exc:
            raise TimeNetBuildError(f"{self.package_name} records an invalid commit in direct_url.json: {exc}") from exc


@dataclass(frozen=True)
class WheelInstall(InstallSource):
    """An install from a wheel, including a path install that is not editable (pip builds a wheel first).

    The installed code is a copy, so a checkout can differ from it. The wheel carries the git
    provenance record baked into it at build time (``_build_info.json``).
    """

    BUILD_INFO_FILE: ClassVar[str] = "_build_info.json"
    """The commit record in each wheel. ``scripts/build_dists.py`` writes it."""

    build_info: Path

    def read_git_source(self) -> GitSource:
        """Read the commit that ``scripts/build_dists.py`` wrote into the package."""
        if not self.build_info.is_file():
            raise self._unknown_commit(
                f"the installed package carries no {self.BUILD_INFO_FILE}. Build it with `make build` or "
                f"scripts/build_dists.py, or upgrade {self.package_name} to a release that records its commit"
            )
        try:
            return GitSource.model_validate_json(self.build_info.read_text(encoding="utf-8"))
        except ValidationError as exc:
            raise TimeNetBuildError(
                f"{self.package_name} records an invalid commit in {self.build_info}: {exc}"
            ) from exc


class InstalledDistribution:
    """One distribution that is installed in this interpreter."""

    def __init__(self, distribution: Distribution) -> None:
        self._distribution = distribution

    @classmethod
    def from_name(cls, name: PackageName) -> Self:
        """Find an installed package by name, such as ``timenet``.

        Note: An extra like ``timenet[zarr]`` is not a name, and ``from_name`` would not find it.

        Returns:
            The installed package.

        Raises:
            PackageNotFoundError: If it is not installed.
        """  # ruff: ignore[docstring-extraneous-exception] - raised by Distribution.from_name
        return cls(Distribution.from_name(name))

    @property
    def name(self) -> PackageName:
        """The package name that pip installs, for example ``timenet-connectors``."""
        return self._distribution.metadata["Name"]

    @property
    def version(self) -> str:
        """The installed version."""
        return self._distribution.version

    def install_source(self) -> InstallSource:
        """Return how this distribution was installed, which decides where its commit is recorded."""
        direct_url = self._read_direct_url()
        if direct_url is not None:
            commit_id = direct_url.installed_git_commit()
            if commit_id is not None:
                return GitInstall(self.name, commit_id)
            project_directory = direct_url.local_directory()
            if direct_url.is_editable() and project_directory is not None:
                return EditableInstall(self.name, project_directory)
        # An install from an index has no direct_url.json. A path install that is not editable
        # copied the package, so the live checkout can differ from the code that runs.
        module = self.name.replace("-", "_")
        build_info = self._distribution.locate_file(f"{module}/{WheelInstall.BUILD_INFO_FILE}")
        return WheelInstall(self.name, Path(str(build_info)))

    def read_git_source(self) -> GitSource:
        """Return the commit of the code that this distribution runs. Raises ``TimeNetBuildError`` if it is unknown."""
        return self.install_source().read_git_source()

    def path_to_ondisk_project_directory(self) -> Path | None:
        """Return the project directory of an editable or local path install, else ``None``."""
        direct_url = self._read_direct_url()
        return None if direct_url is None else direct_url.local_directory()

    def _read_direct_url(self) -> "_DirectUrl | None":
        """Read the PEP 610 ``direct_url.json``, or return ``None`` for an install from an index."""
        raw = self._distribution.read_text("direct_url.json")
        if raw is None:
            return None
        try:
            return _DirectUrl.model_validate_json(raw)
        except ValidationError as exc:
            raise TimeNetBuildError(f"{self.name} has a malformed direct_url.json: {exc}") from exc


class Snapshot(BaseModel, ABC):
    """The commit of some installed code, read at one point in time."""

    model_config = ConfigDict(frozen=True, validate_by_name=True)

    @abstractmethod
    def to_source_map(self) -> GitSourceMap:
        """Return the commits in the form that a manifest records."""


class PackageSnapshot(Snapshot):
    """The commit of one package that pip installed, such as ``timenet`` or ``timenet-connectors``.

    Note: Not a subpackage such as ``timenet.registry``.
    """

    package_name: PackageName
    """The name pip installs, for example ``timenet-connectors``."""
    source: GitSource

    @classmethod
    def read(cls, package_name: PackageName) -> Self:
        """Read the commit of an installed package. Raises ``TimeNetBuildError`` if it is not installed."""
        snapshot = cls.read_if_installed(package_name)
        if snapshot is None:
            raise TimeNetBuildError(
                f"cannot find the git commit of {package_name}: it is not installed in {sys.executable}"
            )
        return snapshot

    @classmethod
    def read_if_installed(cls, package_name: PackageName) -> Self | None:
        """Read the commit of a package, or return ``None`` if it is not installed."""
        try:
            installed = InstalledDistribution.from_name(package_name)
        except PackageNotFoundError:
            return None
        return cls(package_name=package_name, source=installed.read_git_source())

    def to_source_map(self) -> GitSourceMap:
        """Return this package's commit in the form that a manifest records."""
        return {self.package_name: self.source}


class TimeNetSnapshot(Snapshot):
    """The commit of each installed TimeNet package, read at one point in time."""

    timenet: PackageSnapshot
    timenet_connectors: PackageSnapshot | None = Field(default=None, alias="timenet-connectors")
    """``None`` if timenet-connectors is not installed."""

    @model_validator(mode="after")
    def _each_field_holds_its_package(self) -> Self:
        for name, field in type(self).model_fields.items():
            package = getattr(self, name)
            package_name = field.alias or name
            if package is not None and package.package_name != package_name:
                raise ValueError(f"the field for {package_name} holds the commit of {package.package_name}")
        return self

    @classmethod
    def package_names(cls) -> tuple[PackageName, ...]:
        """Return the name of each TimeNet package, in field order."""
        return tuple(field.alias or name for name, field in cls.model_fields.items())

    @classmethod
    def read(cls) -> Self:
        """Read the commit of each installed TimeNet package. Raises ``TimeNetBuildError`` if one is unknown."""
        return cls(
            timenet=PackageSnapshot.read("timenet"),
            timenet_connectors=PackageSnapshot.read_if_installed("timenet-connectors"),
        )

    @classmethod
    def from_source_map(cls, sources: GitSourceMap) -> Self:
        """Return the typed view of the commits that a manifest records."""
        packages = {name: PackageSnapshot(package_name=name, source=source) for name, source in sources.items()}
        try:
            # The field aliases map each package name to its field.
            return cls.model_validate(packages)
        except ValidationError as exc:
            missing = ", ".join(str(error["loc"][0]) for error in exc.errors())
            raise TimeNetBuildError(f"the manifest records no git commit for {missing}") from exc

    def to_source_map(self) -> GitSourceMap:
        """Return the commit of each installed TimeNet package."""
        sources: GitSourceMap = {}
        for package in self._installed():
            sources.update(package.to_source_map())
        return sources

    def packages_with_local_changes(self) -> list[PackageSnapshot]:
        """Return each installed TimeNet package whose code has local changes."""
        return [package for package in self._installed() if package.source.contains_local_changes]

    def _installed(self) -> list[PackageSnapshot]:
        """Return the snapshot of each installed TimeNet package."""
        return [package for package in (self.timenet, self.timenet_connectors) if package is not None]


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
    subdirectory: StrictStr | None = None
    """The project's folder inside ``url``, for a project that does not sit at its root."""

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
        if self.subdirectory is not None:
            path /= self.subdirectory
        return path if self.is_editable() or path.is_dir() else None

    def _is_file_url(self) -> bool:
        return urlparse(self.url).scheme == "file"


class _GitError(Exception):
    """Git could not read a checkout. The message gives the reason."""


class _GitNotFoundError(_GitError):
    """No git executable is on PATH."""


@dataclass(frozen=True)
class _GitCheckout:
    """A directory inside a git checkout, read with the git on PATH."""

    directory: Path

    def head_commit(self) -> str:
        """Return the commit that the checkout is at."""
        return self._run("rev-parse", "HEAD")

    def has_local_changes(self) -> bool:
        """Whether the directory has uncommitted changes, untracked files included."""
        # The pathspec "." scopes the status to the directory. A scratch file elsewhere in the
        # repository does not count as a local change.
        return bool(self._run("status", "--porcelain", "--untracked-files=all", "--", "."))

    def _run(self, *args: str) -> str:
        """Run one git command in the directory and return its stripped stdout."""
        git = shutil.which("git")
        if git is None:
            raise _GitNotFoundError("git is not on PATH")
        try:
            result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
                # --no-optional-locks keeps `git status` from refreshing the index, so a build never
                # takes the index lock while the user runs git.
                [git, "--no-optional-locks", "-C", str(self.directory), *args],
                capture_output=True,
                text=True,
                check=True,
                timeout=_GIT_TIMEOUT_S,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip() or str(exc)
            raise _GitError(f"git failed in {self.directory}: {detail}") from exc
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise _GitError(f"git failed in {self.directory}: {exc}") from exc
        return result.stdout.strip()
