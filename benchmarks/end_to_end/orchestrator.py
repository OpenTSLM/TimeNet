"""Isolated-worktree orchestration shared by the no-op and performance gates."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import subprocess  # noqa: S404 - commands use fixed argument vectors
import tempfile
from typing import Any

from benchmarks.end_to_end.core import MatrixCase, case_to_json


def run(command: list[str], *, cwd: Path, capture: bool = False) -> str:
    """Run a checked subprocess and optionally return stdout.

    Returns:
        Captured stdout, or an empty string.
    """
    result = subprocess.run(  # noqa: S603 - no shell; arguments are explicit
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else ""


def resolve_revision(repo: Path, revision: str) -> str:
    """Resolve a revision to a full immutable commit SHA.

    Returns:
        The full commit SHA.
    """
    return run(["git", "rev-parse", "--verify", f"{revision}^{{commit}}"], cwd=repo, capture=True)


@contextmanager
def worktrees(before_revision: str, after_revision: str, *, keep: bool = False) -> Iterator[tuple[Path, Path, Path]]:
    """Create isolated worktrees for two revisions and clean them on exit.

    Yields:
        The before worktree, after worktree, and artifact root.
    """
    repo = Path(run(["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd(), capture=True))
    root = Path(tempfile.mkdtemp(prefix="timenet-end-to-end."))
    before = root / "before-worktree"
    after = root / "after-worktree"
    try:
        run(["git", "worktree", "add", "--detach", str(before), resolve_revision(repo, before_revision)], cwd=repo)
        run(["git", "worktree", "add", "--detach", str(after), resolve_revision(repo, after_revision)], cwd=repo)
        yield before, after, root
    finally:
        for worktree in (before, after):
            if worktree.exists():
                subprocess.run(  # noqa: S603 - fixed cleanup command, no shell
                    ["git", "worktree", "remove", "--force", str(worktree)],  # noqa: S607
                    cwd=repo,
                    check=False,
                )
        if keep:
            print(f"Artifacts kept at {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)


def measure(worktree: Path, output: Path, case: MatrixCase, *, scale: int) -> dict[str, Any]:
    """Run one matrix case with the selected revision's package and benchmark code.

    Returns:
        The worker's decoded fingerprint and metrics.
    """
    stdout = run(
        [
            *_worker_prefix(),
            "--case",
            case_to_json(case),
            "--scale",
            str(scale),
            "--output",
            str(output),
        ],
        cwd=worktree,
        capture=True,
    )
    return json.loads(stdout.splitlines()[-1])


def revision_capabilities(worktree: Path) -> dict[str, bool]:
    """Inspect format capabilities in one isolated revision.

    Returns:
        Flags reported by the revision's benchmark worker.
    """
    stdout = run([*_worker_prefix(), "--capabilities"], cwd=worktree, capture=True)
    return json.loads(stdout.splitlines()[-1])


def _worker_prefix() -> list[str]:
    """Build the common locked-workspace worker command.

    Returns:
        The command prefix through the worker module name.
    """
    return [
        "uv",
        "run",
        "--frozen",
        "--extra",
        "all",
        "--all-packages",
        "python",
        "-m",
        "benchmarks.end_to_end.worker",
    ]
