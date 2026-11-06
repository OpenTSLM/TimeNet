"""Describe the environment a build ran in, for the manifest's provenance block."""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
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
"""The distributions whose git commit a build records. Each one that is installed must have a known
commit and no local changes before a build publishes to a remote registry."""

BUILD_INFO_FILE = "_build_info.json"
"""The commit record that ``scripts/build_dists.py`` writes into each package of a wheel."""

_GIT_SOURCE = TypeAdapter(GitSource)
_GIT_TIMEOUT_S = 30


@dataclass(frozen=True)
class UnknownCommit:
    """Why the commit of an installed TimeNet distribution is unknown."""

    reason: str


_SNAPSHOT: ContextVar["dict[str, GitSource | UnknownCommit] | None"] = ContextVar("timenet_git_snapshot", default=None)


@contextmanager
def git_snapshot() -> Generator[None]:
    """Read git once, and answer every commit lookup inside the block from that read.

    A build holds one snapshot from its start to its upload. The publish checks and the manifest
    then see the same state, the state of the code that the build imported. An enclosing snapshot
    stays in force. It also works as a decorator: ``@git_snapshot()``.

    Yields:
        Nothing. The snapshot applies to the code inside the block.
    """
    if _SNAPSHOT.get() is not None:
        yield
        return
    token = _SNAPSHOT.set(_read_git())
    try:
        yield
    finally:
        _SNAPSHOT.reset(token)


def require_traceable() -> None:
    """Refuse to publish a build whose TimeNet code cannot be traced to a commit.

    A release in a remote registry outlives the machine that built it, so its manifest
    must name the exact code. Each installed TimeNet distribution needs a known commit and no local
    changes. Inside :func:`git_snapshot`, the check uses the snapshot.

    Raises:
        TimeNetBuildError: If an installed TimeNet distribution has local changes or no known
            commit. The message gives the reason for each.
    """  # ruff: ignore[docstring-extraneous-exception] - raised by _refuse_untraceable
    _refuse_untraceable(_lookup())


def require_unchanged(recorded: BuildEnvironment) -> None:
    """Refuse to upload a manifest that does not name the TimeNet code installed now.

    A remote registry calls this after the write and before the upload. The check reads
    git again, outside any snapshot. If the code changed while the build ran, the manifest names a
    state that no longer exists, and nothing is uploaded.

    Args:
        recorded: The ``build_env`` of the manifest that is about to be uploaded.

    Raises:
        TimeNetBuildError: If the installed code cannot be traced to a commit, or differs from the
            commits that the manifest records.
    """
    current = _read_git()
    _refuse_untraceable(current)
    if recorded.get("git", {}) != current:
        raise TimeNetBuildError(
            f"refusing to publish: the TimeNet code changed while the build ran. The manifest records "
            f"{recorded.get('git', {})}, but the installed code is now at {current}. Build again."
        )


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
    return {name: commit for name, commit in _lookup().items() if not isinstance(commit, UnknownCommit)}


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


def _lookup() -> dict[str, GitSource | UnknownCommit]:
    """Return the active snapshot, or read git if no snapshot is active."""
    snapshot = _SNAPSHOT.get()
    return snapshot if snapshot is not None else _read_git()


def _refuse_untraceable(commits: dict[str, GitSource | UnknownCommit]) -> None:
    """Raise ``TimeNetBuildError`` if a distribution in ``commits`` has no known commit or has local changes."""
    unknown = [f"{name} ({commit.reason})" for name, commit in commits.items() if isinstance(commit, UnknownCommit)]
    changed = [
        f"{name} at {commit['sha'][:12]}"
        for name, commit in commits.items()
        if not isinstance(commit, UnknownCommit) and commit["local_changes"]
    ]
    if not unknown and not changed:
        return
    parts = ["refusing to publish to a remote registry, because its manifest must name the exact code"]
    if changed:
        parts.append(f"Uncommitted changes in the package directory of {', '.join(changed)}. Commit them first")
    if unknown:
        parts.append(
            f"No known git commit for {', '.join(unknown)}. Install TimeNet editable from a git checkout, "
            "from a git URL, or from a wheel that scripts/build_dists.py built"
        )
    parts.append("Or build to a local directory, which accepts any state")
    raise TimeNetBuildError(". ".join(parts) + ".")


def _read_git() -> dict[str, GitSource | UnknownCommit]:
    """Look up the commit of each installed TimeNet distribution. This is the only call that runs git."""
    commits: dict[str, GitSource | UnknownCommit] = {}
    for name in TIMENET_DISTRIBUTIONS:
        try:
            dist = Distribution.from_name(name)
        except PackageNotFoundError:
            continue
        commits[name] = _commit_of(dist)
    return commits


def _commit_of(dist: Distribution) -> GitSource | UnknownCommit:
    """Return the commit of the code that one installed distribution runs, or why it is unknown."""
    info = _direct_url(dist)
    if info is not None:
        vcs = info.get("vcs_info", {})
        if vcs.get("vcs") == "git" and "commit_id" in vcs:
            return _validate_recorded_commit(dist, {"sha": vcs["commit_id"], "local_changes": False}, "direct_url.json")
        if info.get("dir_info", {}).get("editable"):
            directory = local_source_path(dist)
            if directory is not None:
                return _checkout_commit(directory)
    return _wheel_commit(dist)


def _direct_url(dist: Distribution) -> dict | None:
    """Read the PEP 610 ``direct_url.json`` of a distribution, or ``None`` for an index install."""
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
    """Ask git for the commit of a checkout and whether its package directory has changes."""
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


def _wheel_commit(dist: Distribution) -> GitSource | UnknownCommit:
    """Read the commit that ``scripts/build_dists.py`` wrote into the installed package."""
    module = dist.metadata["Name"].replace("-", "_")
    path = Path(str(dist.locate_file(f"{module}/{BUILD_INFO_FILE}")))
    if not path.is_file():
        return UnknownCommit(f"the installed package carries no {BUILD_INFO_FILE}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TimeNetBuildError(f"{dist.metadata['Name']} has a malformed {path}: {exc}") from exc
    return _validate_recorded_commit(dist, record, str(path))


def _validate_recorded_commit(dist: Distribution, record: object, where: str) -> GitSource:
    """Validate a commit record that ``where`` holds, and raise ``TimeNetBuildError`` if it is invalid."""
    try:
        return _GIT_SOURCE.validate_python(record)
    except ValidationError as exc:
        raise TimeNetBuildError(f"{dist.metadata['Name']} records an invalid commit in {where}: {exc}") from exc
