"""Every ``timenet.``-qualified cross-reference in a docstring must name a symbol that exists.

`scripts/gen_api_docs.py` feeds these docstrings to mkdocstrings, which renders a reference it
cannot resolve as literal text instead of a link. A symbol deleted in a refactor leaves that dead
text on the published page, so check the targets here where the failure is cheap to see.
"""

import importlib
from pathlib import Path
import re

import pytest


_SRC = Path(__file__).resolve().parents[1] / "src" / "timenet"
_REF = re.compile(r":(?:class|meth|func|attr|mod|data|obj|exc):`~?(timenet\.[A-Za-z0-9_.]+)`")


def _targets():
    for path in sorted(_SRC.rglob("*.py")):
        for match in _REF.finditer(path.read_text(encoding="utf-8")):
            yield path.relative_to(_SRC.parents[2]), match.group(1)


def _resolves(target):
    parts = target.split(".")
    module = None
    consumed = 0
    for stop in range(len(parts), 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:stop]))
        except ImportError:
            continue
        consumed = stop
        break
    if module is None:
        return False
    current = module
    for part in parts[consumed:]:
        try:
            current = getattr(current, part)
        except AttributeError:
            return False
    return True


@pytest.mark.parametrize(("source", "target"), list(_targets()), ids=str)
def test_docstring_cross_reference_resolves(source, target):
    assert _resolves(target), f"{source} references {target}, which does not exist"
