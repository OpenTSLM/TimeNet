"""Run a connector's build in an environment built from its own requirements.

The parent resolves the connector's ``requirements.txt`` without importing the connector (see
:func:`timenet_connectors.discovery.requirements_for`). It layers that over a base pinned to the
parent's own ``timenet`` and ``timenet-connectors``, then runs ``timenet-build`` in that
environment. uv owns resolution and caching. A repeat build with an unchanged requirement set is a
cache hit.
"""

from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import Distribution, PackageNotFoundError
import json
import os
from pathlib import Path
import shlex
import subprocess  # ruff: ignore[suspicious-subprocess-import] - the command is a fixed argument vector
import sys
import tempfile
from urllib.parse import urlparse
from urllib.request import url2pathname

from uv import find_uv_bin

from timenet.errors import TimeNetBuildError, TimeNetIsolationError
from timenet.manifest import PayloadSource
from timenet.types.references import DatasetRef
from timenet_connectors.discovery import requirements_for


BASE_DISTRIBUTIONS = ("timenet", "timenet-connectors")
"""Pinned to the parent's own versions, so the code that writes the TimeF bytes is the code that
reads them back. ``timenet-connectors`` depends on ``timenet[build]``, so the card-YAML
dependencies arrive transitively and the extra never needs naming here."""


@dataclass(frozen=True)
class EnvSpec:
    """What one connector's environment needs."""

    base: tuple[str, ...]
    """uv arguments installing the base layer (``--with`` or ``--with-editable`` pairs)."""
    requirements: Path | None
    """The connector's ``requirements.txt``, or ``None`` when it declares none."""
    python: str
    """The interpreter the environment is built on. Always one outside a virtualenv."""


def env_spec(dataset_id: str) -> EnvSpec:
    """Describe the dependencies and interpreter for a connector's build.

    Args:
        dataset_id: The dataset id.

    Returns:
        The environment specification.

    Raises:
        LookupError: If no connector exists for the id.
        TimeNetIsolationError: If a base distribution is not installed.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by requirements_for and _base_args, not directly here)
    base = _base_args()
    # The parent cannot read the connector's backend before the environment exists, so the
    # environment always includes the optional Zarr backend. Keep the parent's source/version pin;
    # the additional requirement only requests its extra.
    base += ("--with", f"timenet[zarr]=={Distribution.from_name('timenet').version}")
    return EnvSpec(base=base, requirements=requirements_for(dataset_id), python=_base_interpreter())


def uv_command(spec: EnvSpec, argv: Sequence[str]) -> list[str]:
    """Build the uv command that runs ``argv`` inside the described environment.

    Both halves of the isolation are load-bearing. ``--no-project`` stops uv from discovering and
    syncing whatever project the parent runs in. ``spec.python`` points outside any virtualenv,
    so uv has nothing to layer the environment over.

    Args:
        spec: The environment specification.
        argv: The command to run inside the environment.

    Returns:
        The full argument vector.

    Raises:
        TimeNetIsolationError: If the uv executable is missing.
    """
    try:
        uv = find_uv_bin()
    except FileNotFoundError as exc:
        raise TimeNetIsolationError("cannot run an isolated build: no uv executable found.") from exc
    command = [uv, "run", "--no-project", "--python", spec.python, *spec.base]
    if spec.requirements is not None:
        command += ["--with-requirements", str(spec.requirements)]
    return [*command, *argv]


def run_isolated_build(  # ruff: ignore[too-many-arguments]
    release: DatasetRef,
    store_root: Path,
    *,
    force: bool,
    keep_cache: bool,
    quiet: bool,
    pinned_by: Mapping[DatasetRef, PayloadSource],
) -> None:
    """Build one release in its connector's environment.

    The local registry must already contain the release's parents and their payload sources.
    Build errors include the last lines of stderr.

    Args:
        release: The exact dataset ID and version to build.
        store_root: The local registry directory to build into.
        force: Rebuild the release even if the registry already holds it.
        keep_cache: Keep the raw download cache after building.
        quiet: Hide build status messages.
        pinned_by: File records the build must match, grouped by the release that pins them.

    Raises:
        TimeNetBuildError: If the build fails.
        LookupError: If no connector exists for the dataset ID.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by env_spec, not directly here)
    # --quiet belongs to timenet-build, not to build-prepared-release, so it goes before the subcommand.
    argv = ["timenet-build", *(["--quiet"] if quiet else []), "build-prepared-release", str(release)]
    argv += ["--store", str(store_root), *(["--force"] if force else []), *(["--keep-cache"] if keep_cache else [])]
    spec = env_spec(release.dataset_id)
    # TIMENET_ISOLATION=off is the recursion guard: the child is this same CLI.
    child_env = {**os.environ, "TIMENET_ISOLATION": "off"}
    with _pins_file(pinned_by) as pins:
        if pins is not None:
            argv += ["--pinned-by", str(pins)]
        command = uv_command(spec, argv)
        stderr, returncode = _run_build(command, child_env)
    if returncode != 0:
        tail = "\n".join(stderr.splitlines()[-15:]).strip()
        detail = f"\n{tail}" if tail else " The child produced no error output."
        raise TimeNetBuildError(
            f"building {release} failed with exit code {returncode}. Command: {shlex.join(command)}.{detail}"
        )


@contextmanager
def _pins_file(pinned_by: Mapping[DatasetRef, PayloadSource] | None) -> Generator[Path | None]:
    """Write the pins for an isolated child to a temporary JSON file, keyed by the reading release.

    Yields:
        The file while the child runs, or ``None`` when nothing pins the build.
    """
    if not pinned_by:
        yield None
        return
    with tempfile.TemporaryDirectory(prefix="timenet-pins-") as directory:
        path = Path(directory) / "pins.json"
        pins = {str(release): pinned.model_dump(mode="json") for release, pinned in pinned_by.items()}
        path.write_text(json.dumps(pins), encoding="utf-8")
        yield path


def _run_build(command: list[str], env: dict[str, str]) -> tuple[str, int]:
    """Run the build child, streaming its stderr live while also capturing it.

    stderr is echoed to this process's stderr line by line, so a long download still shows progress,
    and captured too, so a failure carries the child's own error.

    Args:
        command: The full argument vector to run.
        env: The child's environment.

    Returns:
        The child's captured stderr and its exit code.
    """
    process = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
        command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    captured: list[str] = []
    for line in process.stderr or []:
        sys.stderr.write(line)
        sys.stderr.flush()
        captured.append(line)
    process.wait()
    return "".join(captured), process.returncode


def _base_interpreter() -> str:
    """Return this process's interpreter, stepping out of a virtualenv if it is in one.

    uv reads a virtualenv passed as ``--python`` as the *base* of the ephemeral environment and
    layers the ``--with`` overlay on top of it. Every parent site-package stays importable in the
    child, and a package the parent already has silently satisfies a connector requirement instead
    of being resolved. The venv's own base interpreter carries none of that.

    Returns:
        The path to the interpreter uv should build the environment on.
    """
    return getattr(sys, "_base_executable", None) or sys.executable


def _base_args() -> tuple[str, ...]:
    """Build the uv arguments pinning the base layer to this process's own distributions.

    Returns:
        Alternating flag/value arguments for uv.

    Raises:
        TimeNetIsolationError: If a base distribution is not installed in this interpreter.
    """
    args: list[str] = []
    for name in BASE_DISTRIBUTIONS:
        try:
            dist = Distribution.from_name(name)
        except PackageNotFoundError as exc:
            raise TimeNetIsolationError(
                f"cannot make a build environment: {name} is not installed in {sys.executable}."
            ) from exc
        source = _local_source_path(dist)
        if source is not None:
            args += ["--with-editable", str(source)]
        else:
            args += ["--with", f"{name}=={dist.version}"]
    return tuple(args)


def _local_source_path(dist: Distribution) -> Path | None:
    """Return the on-disk project directory an editable or local path install points at.

    ``direct_url.json`` records how a distribution was installed. An editable install carries
    ``dir_info.editable``. A plain path install (``pip install ./packages/timenet``) records a
    ``file://`` URL to the same source tree without that flag. Both let the child build against the
    parent's on-disk source, so both return the directory. An index or wheel install has no
    ``file://`` directory and returns ``None``, leaving the caller to pin the exact version.

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
