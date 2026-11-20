from collections.abc import Callable
import csv
from pathlib import Path

import pytest

from timenet_connectors.datasets.time_mqa.tsqa.release import ReleasePart


@pytest.fixture
def write_part(tmp_path: Path) -> Callable[[ReleasePart, list[dict[str, str]]], dict[str, Path]]:
    """Write synthetic rows as one CSV file of the release, in the shape of its part."""

    def write(part: ReleasePart, rows: list[dict[str, str]]) -> dict[str, Path]:
        path = tmp_path / Path(part.filename).name
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=part.columns)
            writer.writeheader()
            writer.writerows(rows)
        return {part.value: path}

    return write
