# /// script
# requires-python = ">=3.11"
# ///
"""Run each connector's tests and type-check in an environment built from its requirements.

A connector's dependencies live in its own ``requirements.txt``, not in the dev environment. So each
connector's tests and type-check run in an ephemeral uv overlay that adds only that connector's
requirements. Each connector runs in its own overlay, so two connectors never share their
dependencies and a pooled install never happens. A connector with no ``requirements.txt`` runs in the
plain dev environment.
"""

import os
from pathlib import Path
import subprocess  # noqa: S404 - the commands are fixed argument vectors, never a shell string


ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "packages" / "timenet-connectors" / "src" / "timenet_connectors" / "datasets"

# Drop VIRTUAL_ENV so uv does not warn about ignoring an active env: each connector runs in its
# own overlay, not the caller's venv.
_ENV = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}


def _connectors() -> list[Path]:
    """Return every connector directory that has a ``tests`` directory.

    Returns:
        The connector directories, sorted.
    """
    return sorted({tests.parent for tests in DATASETS.glob("*/*/tests")})


def _overlay(connector: Path, command: list[str]) -> list[str]:
    """Build the uv command that runs ``command`` in the connector's environment.

    The overlay layers the connector's requirements over the dev environment (which supplies
    ``timenet``, pytest, and ``ty``). The overlay is ephemeral, so the requirements never persist into
    the dev environment and never mix with another connector's.

    Args:
        connector: The connector directory.
        command: The command to run inside the environment.

    Returns:
        The full uv argument vector.
    """
    uv = ["uv", "run"]
    requirements = connector / "requirements.txt"
    if requirements.is_file():
        uv += ["--with-requirements", str(requirements)]
    return [*uv, *command]


def main() -> int:
    """Run pytest and ``ty`` for every connector.

    Returns:
        ``0`` if every connector passed, ``1`` otherwise.

    Raises:
        RuntimeError: If no connector with a tests directory is found.
    """
    connectors = _connectors()
    if not connectors:
        raise RuntimeError(f"found no connectors with a tests directory under {DATASETS}")
    failures: list[str] = []
    for connector in connectors:
        name = connector.relative_to(DATASETS)
        print(f"\n=== {name} ===", flush=True)
        for command in (["pytest", str(connector / "tests"), "-q"], ["ty", "check", str(connector)]):
            result = subprocess.run(_overlay(connector, command), cwd=ROOT, env=_ENV, check=False)  # noqa: S603
            if result.returncode != 0:
                failures.append(f"{name}: {command[0]}")
    if failures:
        print("\nconnector checks failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nall connector checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
