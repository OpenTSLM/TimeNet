from pathlib import Path
import re
import warnings

import pytest

from timenet_connectors.discovery import available, connector_dir


TEMPLATE = Path(__file__).resolve().parents[3] / "docs" / "connector-verification-template.md"
STATUSES = {"✅ Done", "☑️ Done once", "⚠️ Open", "⚪ N/A"}
SECTIONS = ("## References", "## Checklist", "## Open items", "## Plan")

# Connectors that predate the verification record. Remove a connector in the commit that adds its
# VERIFICATION.md, and never add one: every new connector ships its record.
_PENDING_VERIFICATION = {
    "bash-lab/sensorcaps",
    "chengsenwang/tsqa",
    "physionet/ecg-qa-cot",
    "physionet/sleep-edfx",
}
# Connectors that generate their data in code. There is no source to verify them against.
_SYNTHETIC = {
    "timenet/hello-world",
    "timenet/test-mean",
}

_VERSION = re.compile(r"^- \*\*Checklist version:\*\* (\d+)\s*$", re.M)
# A checklist row: | number | error | status | evidence |. A pipe inside a cell is escaped as \|.
_ROW = re.compile(r"^\|\s*(\d+)\s*\|.*\|\s*$", re.M)
_NAMED_TEST = re.compile(r"`(test_\w+)`")


def _normalize(status: str) -> str:
    # Emoji may be typed with or without the variation selector, as in ☑️ and ☑.
    return status.replace("️", "")


def _version(text: str, path: Path) -> int:
    match = _VERSION.search(text)
    assert match, f"{path}: no '- **Checklist version:** N' line"
    return int(match.group(1))


def _rows(text: str) -> list[tuple[int, str, str]]:
    # Only the checklist holds rows; Open items or Plan may number their own tables.
    checklist = text.split("## Checklist", 1)[-1].split("\n## ", 1)[0]
    rows = []
    for match in _ROW.finditer(checklist):
        cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", match.group(0))[1:-1]]
        assert len(cells) == 4, f"checklist row needs 4 cells, got {len(cells)}: {match.group(0)}"
        rows.append((int(cells[0]), cells[2], cells[3]))
    return rows


def _template_items() -> tuple[int, list[int]]:
    text = TEMPLATE.read_text()
    return _version(text, TEMPLATE), [number for number, _, _ in _rows(text)]


def _test_names(dataset_id: str) -> set[str]:
    files = [*connector_dir(dataset_id).rglob("test_*.py"), *Path(__file__).parent.glob("test_*.py")]
    return {name for file in files for name in re.findall(r"^\s*def (test_\w+)", file.read_text(), re.M)}


def test_template_lists_every_checklist_item_once_in_order():
    _, items = _template_items()
    assert items == list(range(1, len(items) + 1))


def test_exempt_connectors_exist():
    assert set(available()) >= _PENDING_VERIFICATION | _SYNTHETIC


@pytest.mark.parametrize("dataset_id", available())
def test_every_connector_ships_a_verification_record(dataset_id):
    path = connector_dir(dataset_id) / "VERIFICATION.md"
    if dataset_id in _SYNTHETIC:
        pytest.skip(f"{dataset_id} is synthetic")
    if dataset_id in _PENDING_VERIFICATION:
        assert not path.exists(), f"{path} exists: remove {dataset_id} from _PENDING_VERIFICATION"
        pytest.skip(f"{dataset_id} predates the verification record")
    assert path.is_file(), f"{dataset_id} has no VERIFICATION.md; start from {TEMPLATE.name}"
    text = path.read_text()
    assert "TODO" not in re.sub(r"<!--.*?-->", "", text, flags=re.S), f"{path}: replace every TODO"
    for section in SECTIONS:
        assert re.search(rf"^{section}\s*$", text, re.M), f"{path}: no '{section}' section"

    current, items = _template_items()
    version = _version(text, path)
    assert 1 <= version <= current, f"{path}: checklist version {version} isn't between 1 and {current}"
    if version < current:
        warnings.warn(f"{path} answers checklist version {version}; the template is at {current}", stacklevel=1)
    else:
        assert [number for number, _, _ in _rows(text)] == items, f"{path}: answer every item once, in order"

    tests = _test_names(dataset_id)
    statuses = {_normalize(status) for status in STATUSES}
    for number, status, evidence in _rows(text):
        # Combined statuses are spaced, "✅ Done / ⚠️ Open", which keeps "N/A" whole.
        parts = {_normalize(part) for part in status.split(" / ")}
        assert parts <= statuses, f"{path}: item {number} has status {status!r}; use {sorted(STATUSES)}"
        assert evidence, f"{path}: item {number} needs evidence"
        if _normalize("✅ Done") in parts:
            assert "`" in evidence, f"{path}: item {number} is Done, so its evidence names the check in backticks"
            missing = set(_NAMED_TEST.findall(evidence)) - tests
            assert not missing, f"{path}: item {number} names tests that don't exist: {sorted(missing)}"
