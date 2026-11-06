"""Static checks of the connector conventions in ``docs/connectors.md``.

The checks parse the code of every connector under ``datasets/``, and the shared code under
``bases/`` and ``sources/`` that connectors call, without tests. Ruff can ban an import, but it cannot
ban ``raise ValueError``, so these checks are tests.
"""

import ast
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pytest

import timenet_connectors


_PACKAGE = Path(timenet_connectors.__file__).parent
_CHECKED = ("datasets", "bases", "sources")
"""The folders whose code reads source data: the connectors, and the bases and readers they share."""

_BANNED_RAISES = frozenset({"ValueError", "TimeFFormatError"})
"""Bad source data raises ``TimeNetConnectorError``. TimeNet raises these errors for its own checks."""


def _connector_files() -> Iterator[Path]:
    """Yield every Python file of the connectors and of their shared code, without tests."""
    for folder in _CHECKED:
        for path in sorted((_PACKAGE / folder).rglob("*.py")):
            if "tests" not in path.relative_to(_PACKAGE).parts:
                yield path


def _raised_name(node: ast.Raise) -> str | None:
    """Return the name of the exception that a ``raise`` statement raises, if the code names it."""
    exception = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
    if isinstance(exception, ast.Name):
        return exception.id
    if isinstance(exception, ast.Attribute):
        return exception.attr
    return None


def _violations(path: Path, root: Path = _PACKAGE) -> tuple[Counter[str], list[str]]:
    """Find the banned raises in one file.

    Returns:
        The count of each banned raise, and one line per raise for the failure message.
    """
    counts: Counter[str] = Counter()
    lines: list[tuple[int, str]] = []
    relative = path.relative_to(root).as_posix()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Raise) and (name := _raised_name(node)) in _BANNED_RAISES:
            counts[name] += 1
            lines.append((node.lineno, f"{relative}:{node.lineno}: raise {name}, raise TimeNetConnectorError instead"))
    return counts, [line for _, line in sorted(lines)]


def test_the_checks_find_connector_files():
    files = [path.relative_to(_PACKAGE).as_posix() for path in _connector_files()]
    assert "datasets/timenet/hello_world/connector.py" in files
    assert "bases/excel.py" in files
    assert "sources/json_stream.py" in files
    assert not any("/tests/" in name for name in files)


@pytest.mark.parametrize("path", list(_connector_files()), ids=lambda path: path.relative_to(_PACKAGE).as_posix())
def test_connectors_raise_connector_errors_for_bad_source_data(path):
    _, lines = _violations(path)
    assert not lines, "\n".join(["This file breaks the error rule in docs/connectors.md:", *lines])


def test_the_checks_catch_each_banned_raise(tmp_path):
    path = tmp_path / "org" / "name" / "connector.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "from timenet import errors\n"
        "from timenet.errors import TimeFFormatError\n"
        "def f(x):\n"
        "    if x: raise ValueError('bad')\n"
        "    if not x: raise errors.TimeFFormatError('bad')\n"
        "    raise TimeFFormatError\n",
        encoding="utf-8",
    )
    counts, lines = _violations(path, root=tmp_path)
    assert counts == Counter({"ValueError": 1, "TimeFFormatError": 2})
    assert lines[0].startswith("org/name/connector.py:4: raise ValueError")
