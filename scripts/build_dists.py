# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Build distributions with the git commit of each package recorded inside it.

A wheel has no git directory, so ``timenet.provenance.git_sources`` reads the commit from the
``_build_info.json`` that this script writes into each import package. ``local_changes`` covers the
package directory only, the same scope that a local checkout records. The script then runs
``uv build`` with the given arguments, and removes the files again even if the build fails. A
later plain ``uv build`` therefore never ships an old record. ``make build`` and the publish
workflow run this script.

Usage::

    uv run scripts/build_dists.py --all-packages
    uv run scripts/build_dists.py --package timenet --out-dir dist
"""

import json
import os
from pathlib import Path
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] - the commands are fixed argument vectors
import sys


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    ROOT / "packages" / "timenet": "timenet",
    ROOT / "packages" / "timenet-connectors": "timenet_connectors",
}
# Keep in step with timenet.provenance.BUILD_INFO_FILE, which reads the file.
BUILD_INFO_FILE = "_build_info.json"


def executable(name: str) -> str:
    """Return the absolute path of a command on ``PATH``.

    Returns:
        The path of the command.

    Raises:
        SystemExit: If the command is not on ``PATH``.
    """
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"{name} is not on PATH")
    return path


def git(directory: Path, *args: str) -> str:
    """Run one git command in ``directory`` and return its stripped stdout.

    Returns:
        The command's stdout without surrounding whitespace.

    Raises:
        SystemExit: If git fails, for example outside a git checkout.
    """
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [executable("git"), "-C", str(directory), *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed in {directory}: {result.stderr.strip()}")
    return result.stdout.strip()


def main(build_args: list[str]) -> int:
    """Write ``_build_info.json`` into each import package, build, and remove the files.

    Args:
        build_args: The arguments for ``uv build``.

    Returns:
        The exit code of ``uv build``.
    """
    written: list[Path] = []
    try:
        for directory, module in PACKAGES.items():
            sha = git(directory, "rev-parse", "HEAD")
            status = git(directory, "status", "--porcelain", "--untracked-files=all", "--", ".")
            info = {"sha": sha, "local_changes": bool(status)}
            target = directory / "src" / module / BUILD_INFO_FILE
            target.write_text(json.dumps(info) + "\n", encoding="utf-8")
            written.append(target)
            print(f"{target.relative_to(ROOT)}: {info}", file=sys.stderr)
        # uv run sets UV to its own executable.
        uv = os.environ.get("UV") or executable("uv")
        return subprocess.run([uv, "build", *build_args], cwd=ROOT, check=False).returncode  # ruff: ignore[subprocess-without-shell-equals-true]
    finally:
        for path in written:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
