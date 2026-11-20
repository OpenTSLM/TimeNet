from collections.abc import Callable
import csv
from pathlib import Path

import pytest

from timenet_connectors.bases.huggingface import HubFileRef
from timenet_connectors.datasets.time_mqa.tsqa.release import ReleasePart


@pytest.fixture
def write_part(tmp_path: Path) -> Callable[[ReleasePart, list[dict[str, str]]], HubFileRef]:
    """Write synthetic rows as one CSV file of the release, in the shape of its part."""

    def write(part: ReleasePart, rows: list[dict[str, str]]) -> HubFileRef:
        path = tmp_path / Path(part.filename).name
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=part.columns)
            writer.writeheader()
            writer.writerows(rows)
        return HubFileRef(path, part.value)

    return write
