# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Run ``uv build`` with the git commit of each package recorded in its ``_build_info.json``.

Usage: ``uv run scripts/build_dists.py --all-packages`` (any ``uv build`` arguments work).
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
# Keep in step with timenet.provenance.BuiltPackage.BUILD_INFO_FILE, which reads the file.
BUILD_INFO_FILE = "_build_info.json"


def executable(name: str) -> str:
    """Return the absolute path of a command on ``PATH``, or exit if it is missing."""
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"{name} is not on PATH")
    return path


def git(directory: Path, *args: str) -> str:
    """Run one git command in ``directory`` and return its stripped stdout, or exit if it fails."""
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [executable("git"), "-C", str(directory), *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed in {directory}: {result.stderr.strip()}")
    return result.stdout.strip()


def state(directory: Path) -> tuple[str, str]:
    """Return the commit of a package directory and its ``git status`` output."""
    # Read the commit after the status, as timenet.provenance.EditableCheckout does.
    status = git(directory, "status", "--porcelain", "--untracked-files=all", "--", ".")
    return git(directory, "rev-parse", "HEAD"), status


def main(build_args: list[str]) -> int:
    """Write the commit records, build, and remove the records again, also on failure."""
    written: dict[Path, str] = {}
    before = {directory: state(directory) for directory in PACKAGES}
    try:
        for directory, module in PACKAGES.items():
            sha, status = before[directory]
            # Keep the keys in step with timenet.manifest.GitSource.
            info = {"sha": sha, "contains_local_changes": bool(status)}
            target = directory / "src" / module / BUILD_INFO_FILE
            text = json.dumps(info) + "\n"
            # Register the file before the write, so a write that fails halfway is removed too.
            written[target] = text
            target.write_text(text, encoding="utf-8")
            print(f"{target.relative_to(ROOT)}: {info}", file=sys.stderr)
        # uv run sets UV to its own executable.
        uv = os.environ.get("UV") or executable("uv")
        returncode = subprocess.run([uv, "build", *build_args], cwd=ROOT, check=False).returncode  # ruff: ignore[subprocess-without-shell-equals-true]
        # A file that changed while uv built would ship under a record that does not describe it.
        changed = [str(directory.relative_to(ROOT)) for directory in PACKAGES if state(directory) != before[directory]]
        if changed:
            print(
                f"error: {', '.join(changed)} changed during the build, so the built artifacts do not "
                "match their commit records. Build again.",
                file=sys.stderr,
            )
            return 1
        # git ignores the records, so the status above cannot see one that another build removed.
        lost = [str(path.relative_to(ROOT)) for path, text in written.items() if not _holds(path, text)]
        if lost:
            print(
                f"error: {', '.join(lost)} changed during the build, so the built artifacts can lack their "
                "commit records. Do not run two builds in one checkout at the same time. Build again.",
                file=sys.stderr,
            )
            return 1
        return returncode
    finally:
        for path in written:
            path.unlink(missing_ok=True)


def _holds(path: Path, text: str) -> bool:
    """Return whether ``path`` exists and holds ``text``."""
    try:
        return path.read_text(encoding="utf-8") == text
    except FileNotFoundError:
        return False


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
