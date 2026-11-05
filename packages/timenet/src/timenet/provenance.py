"""Describe the environment a build ran in, for the manifest's provenance block."""

from dataclasses import dataclass
from importlib.metadata import Distribution, PackageNotFoundError, distributions
import json
from pathlib import Path
import platform
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] - the commands are fixed argument vectors
from urllib.parse import urlparse
from urllib.request import url2pathname

from pydantic import TypeAdapter, ValidationError

from timenet.errors import TimeNetBuildError
from timenet.manifest import BuildEnvironment, GitSource


TIMENET_DISTRIBUTIONS = ("timenet", "timenet-connectors")
"""The distributions whose git commit a build records."""

BUILD_INFO_FILE = "_build_info.json"
"""The file that ``scripts/build_dists.py`` writes into an import package while it builds a wheel.
It holds the :class:`~timenet.manifest.GitSource` of the checkout that built the wheel. The script
keeps the same name in its own constant."""

_GIT_SOURCE = TypeAdapter(GitSource)
_GIT_TIMEOUT_S = 30


@dataclass(frozen=True)
class UnknownCommit:
    """Why the commit of an installed TimeNet distribution is unknown."""

    reason: str


def build_env() -> BuildEnvironment:
    """Describe this process's Python version, installed packages, and TimeNet commits.

    Under an isolated build the recorded set is exactly the base, the connector's requirements, and
    their transitives. Under ``--no-isolation`` it is the whole ambient environment, which is noisier
    but still the truth about what produced the dataset.

    Returns:
        A dict with ``python`` (the interpreter version), ``packages`` (name -> version, sorted), and
        ``git`` (see :func:`git_sources`).
    """
    packages: dict[str, str] = {}
    for dist in distributions():
        name = dist.metadata["Name"]
        if name:
            packages[name] = dist.version
    return {"python": platform.python_version(), "packages": dict(sorted(packages.items())), "git": git_sources()}


def git_sources() -> dict[str, GitSource]:
    """Return the git commit of each installed TimeNet distribution whose commit is known.

    Each distribution reports the commit of the code that runs:

    - An editable install asks git. ``local_changes`` covers the package directory only, so a
      scratch file elsewhere in the repository does not set it.
    - A git install (``pip install git+https://...``) reports the commit that pip recorded.
    - Any other install, such as a wheel or ``pip install ./packages/timenet``, reads the
      ``_build_info.json`` that the build wrote into the package.

    A distribution whose commit is unknown is absent.

    Returns:
        The commits, keyed by distribution name.
    """
    return {name: commit for name, commit in _read_git().items() if not isinstance(commit, UnknownCommit)}


def local_source_path(dist: Distribution) -> Path | None:
    """Return the on-disk project directory an editable or local path install points at.

    ``direct_url.json`` records how a distribution was installed. An editable install carries
    ``dir_info.editable``. A plain path install (``pip install ./packages/timenet``) records a
    ``file://`` URL to the same source tree without that flag. Both return the directory. An index or
    wheel install has no ``file://`` directory and returns ``None``.

    Args:
        dist: The installed distribution.

    Returns:
        The source directory for an editable or local path install, else ``None``.
    """
    info = _direct_url(dist)
    if info is None:
        return None
    parsed = urlparse(info.get("url", ""))
    if parsed.scheme != "file":
        return None
    path = Path(url2pathname(parsed.path))
    if info.get("dir_info", {}).get("editable") or path.is_dir():
        return path
    return None


def _read_git() -> dict[str, GitSource | UnknownCommit]:
    """Look up the commit of each installed TimeNet distribution. This is the only call that runs git.

    Returns:
        The commit or the reason it is unknown, keyed by distribution name. A distribution that is
        not installed is absent.
    """
    commits: dict[str, GitSource | UnknownCommit] = {}
    for name in TIMENET_DISTRIBUTIONS:
        try:
            dist = Distribution.from_name(name)
        except PackageNotFoundError:
            continue
        commits[name] = _commit_of(dist)
    return commits


def _commit_of(dist: Distribution) -> GitSource | UnknownCommit:
    """Return the commit of the code that one installed distribution runs.

    Returns:
        The commit, or the reason it is unknown.

    Raises:
        TimeNetBuildError: If a recorded commit is not a valid commit record.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by _direct_url, _validated, and _wheel_commit
    info = _direct_url(dist)
    if info is not None:
        vcs = info.get("vcs_info", {})
        if vcs.get("vcs") == "git" and "commit_id" in vcs:
            return _validated(dist, {"sha": vcs["commit_id"], "local_changes": False}, "direct_url.json")
        if info.get("dir_info", {}).get("editable"):
            directory = local_source_path(dist)
            if directory is not None:
                return _checkout_commit(directory)
    return _wheel_commit(dist)


def _direct_url(dist: Distribution) -> dict | None:
    """Read the PEP 610 ``direct_url.json`` of a distribution.

    Returns:
        The parsed record, or ``None`` for an install from an index.

    Raises:
        TimeNetBuildError: If the record is not a JSON object.
    """
    raw = dist.read_text("direct_url.json")
    if raw is None:
        return None
    try:
        info = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TimeNetBuildError(f"{dist.metadata['Name']} has a malformed direct_url.json: {exc}") from exc
    if not isinstance(info, dict):
        raise TimeNetBuildError(f"{dist.metadata['Name']} has a direct_url.json that is not a JSON object")
    return info


def _checkout_commit(directory: Path) -> GitSource | UnknownCommit:
    """Ask git for the commit of a checkout and whether its package directory has changes.

    Returns:
        The commit, or the reason git could not give it.
    """
    git = shutil.which("git")
    if git is None:
        return UnknownCommit("git is not on PATH")
    try:
        sha = _git(git, directory, "rev-parse", "HEAD")
        # The pathspec "." scopes the status to the package directory, untracked files included.
        status = _git(git, directory, "status", "--porcelain", "--untracked-files=all", "--", ".")
    except subprocess.CalledProcessError as exc:
        return UnknownCommit(f"git failed in {directory}: {(exc.stderr or '').strip() or exc}")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return UnknownCommit(f"git failed in {directory}: {exc}")
    return {"sha": sha, "local_changes": bool(status)}


def _git(git: str, directory: Path, *args: str) -> str:
    """Run one git command in ``directory`` and return its stripped stdout.

    ``--no-optional-locks`` keeps ``git status`` from refreshing the index, so a build never takes
    the index lock while the user runs git.

    Returns:
        The command's stdout without surrounding whitespace.
    """
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [git, "--no-optional-locks", "-C", str(directory), *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=_GIT_TIMEOUT_S,
    )
    return result.stdout.strip()


def _wheel_commit(dist: Distribution) -> GitSource | UnknownCommit:
    """Read the commit that ``scripts/build_dists.py`` wrote into the installed package.

    Returns:
        The commit, or the reason it is unknown.

    Raises:
        TimeNetBuildError: If the file is not valid JSON or not a valid commit record.
    """
    for file in dist.files or ():
        # The file sits at the top of the import package: ``<package>/_build_info.json``.
        if file.name == BUILD_INFO_FILE and len(file.parent.parts) == 1:
            try:
                record = json.loads(file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise TimeNetBuildError(f"{dist.metadata['Name']} has a malformed {file}: {exc}") from exc
            return _validated(dist, record, str(file))
    return UnknownCommit(f"the installed package carries no {BUILD_INFO_FILE}")


def _validated(dist: Distribution, record: object, where: str) -> GitSource:
    """Validate a recorded commit.

    Returns:
        The commit.

    Raises:
        TimeNetBuildError: If the record is not a valid commit record.
    """
    try:
        return _GIT_SOURCE.validate_python(record)
    except ValidationError as exc:
        raise TimeNetBuildError(f"{dist.metadata['Name']} records an invalid commit in {where}: {exc}") from exc
