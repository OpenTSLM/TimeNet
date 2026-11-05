# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Write the git commit of each package into its import package before ``uv build``.

A wheel has no git directory, so ``timenet.provenance.git_sources`` reads the commit from the
``_build_info.json`` that this script writes. ``local_changes`` covers the package directory only,
the same scope that a local checkout records. The file is git-ignored, so it never reports itself as
a change. ``make build`` and the publish workflow run this script first.

Usage::

    uv run scripts/write_build_info.py
"""

import json
from pathlib import Path
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] - the commands are fixed argument vectors


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    ROOT / "packages" / "timenet": "timenet",
    ROOT / "packages" / "timenet-connectors": "timenet_connectors",
}
BUILD_INFO_FILE = "_build_info.json"


def git(directory: Path, *args: str) -> str:
    """Run one git command in ``directory`` and return its stripped stdout.

    Returns:
        The command's stdout without surrounding whitespace.

    Raises:
        SystemExit: If git is not on ``PATH``.
    """
    executable = shutil.which("git")
    if executable is None:
        raise SystemExit("git is not on PATH; cannot record the commit of each package")
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [executable, "-C", str(directory), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def main() -> None:
    """Write ``_build_info.json`` into each import package."""
    for directory, module in PACKAGES.items():
        sha = git(directory, "rev-parse", "HEAD")
        status = git(directory, "status", "--porcelain", "--untracked-files=all", "--", ".")
        info = {"sha": sha, "local_changes": bool(status)}
        target = directory / "src" / module / BUILD_INFO_FILE
        target.write_text(json.dumps(info) + "\n", encoding="utf-8")
        print(f"{target.relative_to(ROOT)}: {info}")


if __name__ == "__main__":
    main()
