"""Run each connector's tests and type-check in an environment built from its requirements.

A connector's dependencies live in its own ``requirements.txt``, not in the dev environment. So each
connector's tests and type-check run in an ephemeral uv overlay that adds only that connector's
requirements. Each connector runs in its own overlay, so two connectors never share their
dependencies. A connector with no ``requirements.txt`` runs in the plain dev environment.
"""

from collections.abc import Callable, Iterable
import os
from pathlib import Path
import subprocess  # ruff: ignore[suspicious-subprocess-import] - fixed argument vectors, never a shell string


def connectors(roots: Iterable[Path]) -> list[tuple[str, Path]]:
    """Find every connector directory that has a ``tests`` directory.

    Args:
        roots: The datasets directories to search.

    Returns:
        The name of each connector below its root and its directory, sorted within each root.
    """
    found: list[tuple[str, Path]] = []
    for root in roots:
        directories = sorted({tests.parent for tests in root.glob("*/*/tests")})
        found += [(str(path.relative_to(root)), path) for path in directories]
    return found


def run_checks(found: list[tuple[str, Path]], echo: Callable[[str], None]) -> list[str]:
    """Run pytest and ``ty`` for each connector.

    Args:
        found: The display name and directory of each connector.
        echo: Prints one line of status.

    Returns:
        One ``<name>: <tool>`` entry for each check that failed.
    """
    # Drop VIRTUAL_ENV so uv does not warn that it ignores an active environment.
    env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    failures: list[str] = []
    for name, connector in found:
        echo(f"\n=== {name} ===")
        overlay = ["uv", "run"]
        if (requirements := connector / "requirements.txt").is_file():
            overlay += ["--with-requirements", str(requirements)]
        for command in (["pytest", str(connector / "tests"), "-q"], ["ty", "check", str(connector)]):
            # ruff: ignore[subprocess-without-shell-equals-true]
            result = subprocess.run([*overlay, *command], env=env, check=False)
            if result.returncode != 0:
                failures.append(f"{name}: {command[0]}")
    return failures
