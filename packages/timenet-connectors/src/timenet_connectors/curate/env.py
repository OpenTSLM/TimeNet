"""Run a connector's curation in an environment built from its own requirements.

The parent resolves the connector's ``requirements.txt`` without importing the connector (see
:func:`timenet_connectors.discovery.requirements_for`). It layers that over a base pinned to the
parent's own ``timenet`` and ``timenet-connectors``, then runs ``timenet-curate build`` in that
environment. uv owns resolution and caching. A repeat build with an unchanged requirement set is a
cache hit.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import Distribution, PackageNotFoundError
import json
import os
from pathlib import Path
import shlex
import subprocess  # noqa: S404 - the command is a fixed argument vector
import sys
from urllib.parse import urlparse
from urllib.request import url2pathname

from uv import find_uv_bin

from timenet.errors import CurationError
from timenet_connectors.discovery import requirements_for


BASE_DISTRIBUTIONS = ("timenet", "timenet-connectors")
"""Pinned to the parent's own versions, so the code that writes the TimeF bytes is the code that
reads them back. ``timenet-connectors`` depends on ``timenet[curation]``, so the card-YAML
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
    """Describe the environment a dataset's connector needs.

    Args:
        dataset_id: The dataset id.

    Returns:
        The environment specification.

    Raises:
        LookupError: If no connector exists for the id.
        CurationError: If a base distribution is not installed.
    """  # noqa: DOC502 (raised by requirements_for and _base_args, not directly here)
    return EnvSpec(base=_base_args(), requirements=requirements_for(dataset_id), python=_base_interpreter())


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
        CurationError: If the uv executable is missing.
    """
    try:
        uv = find_uv_bin()
    except FileNotFoundError as exc:
        raise CurationError(
            "cannot run an isolated build: no uv executable found. Install uv, or build in this "
            "interpreter with TIMENET_ISOLATION=off."
        ) from exc
    command = [uv, "run", "--no-project", "--python", spec.python, *spec.base]
    if spec.requirements is not None:
        command += ["--with-requirements", str(spec.requirements)]
    return [*command, *argv]


def run_isolated(
    dataset_id: str, root: Path, *, force: bool = False, keep_cache: bool = False, quiet: bool = False
) -> Path:
    """Build a dataset in its own environment and return the committed version directory.

    The child's stderr is inherited, so a long download shows its progress live. Its stdout is
    captured, because the last line is the version directory.

    Args:
        dataset_id: The dataset id.
        root: The output registry directory.
        force: Rebuild even if the version is already curated.
        keep_cache: Keep the raw download cache after building.
        quiet: Suppress the child's status output, as ``--quiet`` does in this process.

    Returns:
        The committed version directory.

    Raises:
        CurationError: If the child failed, or printed no version directory.
        LookupError: If no connector exists for the id.
    """  # noqa: DOC502 (raised by env_spec, not directly here)
    # --quiet belongs to timenet-curate, not to build, so it goes before the subcommand.
    argv = ["timenet-curate", *(["--quiet"] if quiet else []), "build", dataset_id, "--out", str(root)]
    if force:
        argv.append("--force")
    if keep_cache:
        argv.append("--keep-cache")
    command = uv_command(env_spec(dataset_id), argv)
    # TIMENET_ISOLATION=off is the recursion guard: the child is this same CLI.
    child_env = {**os.environ, "TIMENET_ISOLATION": "off"}
    result = subprocess.run(  # noqa: S603
        command, env=child_env, stdout=subprocess.PIPE, text=True, check=False
    )
    if result.returncode != 0:
        raise CurationError(
            f"curating {dataset_id!r} failed with exit code {result.returncode}. "
            f"See the output above for the cause. Command: {shlex.join(command)}."
        )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise CurationError(f"curating {dataset_id!r} printed no version directory")
    return Path(lines[-1])


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
        CurationError: If a base distribution is not installed in this interpreter.
    """
    args: list[str] = []
    for name in BASE_DISTRIBUTIONS:
        try:
            dist = Distribution.from_name(name)
        except PackageNotFoundError as exc:
            raise CurationError(
                f"cannot build a curation environment: {name} is not installed in {sys.executable}. "
                "Reinstall it, or build in this interpreter with TIMENET_ISOLATION=off."
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
