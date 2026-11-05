"""Describe the environment a build ran in, for the manifest's provenance block."""

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
"""The file that ``scripts/write_build_info.py`` writes into an import package before a wheel is
built. It holds the :class:`~timenet.manifest.GitSource` of the checkout that built the wheel."""

_GIT_SOURCE = TypeAdapter(GitSource)


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
    """Return the git commit of each installed TimeNet distribution.

    A local checkout (an editable or path install) asks git directly. Its ``local_changes`` covers
    the package directory only, so a scratch file elsewhere in the repository does not set it. A
    wheel carries the commit in its ``_build_info.json``. A distribution with neither is absent.

    Returns:
        The commits, keyed by distribution name.
    """
    sources: dict[str, GitSource] = {}
    for name in TIMENET_DISTRIBUTIONS:
        try:
            dist = Distribution.from_name(name)
        except PackageNotFoundError:
            continue
        directory = local_source_path(dist)
        source = _checkout_source(directory) if directory is not None else _wheel_source(dist)
        if source is not None:
            sources[name] = source
    return sources


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
    raw = dist.read_text("direct_url.json")
    if raw is None:
        return None
    info = json.loads(raw)
    parsed = urlparse(info.get("url", ""))
    if parsed.scheme != "file":
        return None
    path = Path(url2pathname(parsed.path))
    if info.get("dir_info", {}).get("editable") or path.is_dir():
        return path
    return None


def _checkout_source(directory: Path) -> GitSource | None:
    """Ask git for the commit of a checkout and whether its package directory has changes.

    Returns:
        The commit, or ``None`` if git is missing or the directory is not in a git checkout.
    """
    git = shutil.which("git")
    if git is None:
        return None
    try:
        sha = _git(git, directory, "rev-parse", "HEAD")
        # The pathspec "." scopes the status to the package directory, untracked files included.
        status = _git(git, directory, "status", "--porcelain", "--untracked-files=all", "--", ".")
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"sha": sha, "local_changes": bool(status)}


def _git(git: str, directory: Path, *args: str) -> str:
    """Run one git command in ``directory`` and return its stripped stdout.

    Returns:
        The command's stdout without surrounding whitespace.
    """
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [git, "-C", str(directory), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _wheel_source(dist: Distribution) -> GitSource | None:
    """Read the commit that ``scripts/write_build_info.py`` wrote into a wheel.

    Returns:
        The commit, or ``None`` if the distribution carries no build info.

    Raises:
        TimeNetBuildError: If the build info file is not a valid commit record.
    """
    for file in dist.files or ():
        # The file sits at the top of the import package: ``<package>/_build_info.json``.
        if file.name == BUILD_INFO_FILE and len(file.parent.parts) == 1:
            text = file.read_text(encoding="utf-8")
            try:
                return _GIT_SOURCE.validate_json(text)
            except ValidationError as exc:
                raise TimeNetBuildError(
                    f"{dist.metadata['Name']} carries an invalid {BUILD_INFO_FILE} at {file}: {exc}"
                ) from exc
    return None
