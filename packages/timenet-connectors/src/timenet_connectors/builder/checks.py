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

from timenet_connectors.discovery import _roots


def discovered_roots() -> list[Path]:
    """Return the directories of every datasets root that discovery finds.

    Returns:
        The built-in datasets directory, then the directories of each registered root.
    """
    return [Path(path) for root in _roots() for path in root.paths]


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


def overlay(connector: Path, command: list[str]) -> list[str]:
    """Build the uv command that runs ``command`` in the environment of a connector.

    The overlay adds the requirements of the connector to the current project environment, which
    supplies ``timenet``, pytest and ``ty``. The overlay is ephemeral, so the requirements never
    stay in the project environment and never mix with the requirements of another connector.

    Args:
        connector: The connector directory.
        command: The command to run in the environment.

    Returns:
        The full uv argument vector.
    """
    uv = ["uv", "run"]
    requirements = connector / "requirements.txt"
    if requirements.is_file():
        uv += ["--with-requirements", str(requirements)]
    return [*uv, *command]


def commands(connector: Path) -> list[list[str]]:
    """Return the pytest and ``ty`` commands for one connector.

    Args:
        connector: The connector directory.

    Returns:
        The commands to run in the overlay, pytest first.
    """
    return [["pytest", str(connector / "tests"), "-q"], ["ty", "check", str(connector)]]


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
        for command in commands(connector):
            # ruff: ignore[subprocess-without-shell-equals-true]
            result = subprocess.run(overlay(connector, command), env=env, check=False)
            if result.returncode != 0:
                failures.append(f"{name}: {command[0]}")
    return failures
