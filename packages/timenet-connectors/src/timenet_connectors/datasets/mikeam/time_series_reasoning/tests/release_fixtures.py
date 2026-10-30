"""The fixture release the tests read: real lines of the pinned release, copied byte for byte, in order.

Each case comes from the row with the shortest series that has it:

- ``MCQ_1_TS.jsonl``: lines 14221 (the answer's text is also a wrong option's) and 80150.
- ``MCQ_2_TS.jsonl``: lines 20351 and 20441, two counterfactuals of ``04e3c8fe``, one of whose changed
  series is longer than the series; and 45584, whose question and options are all blank.
- The description files: the rows of ``1b23301e``, ``37ff629a`` (another metadata key order),
  ``fe2aff54`` (a series in integers) and ``8d269711`` (twice, with the same values) in train,
  ``198e1858`` in val and ``10fbcbdc`` in test.
- ``TS_Dataset.jsonl``: the rows of the 9 series the other files name.
"""

import json
from pathlib import Path

from timenet_connectors.datasets.mikeam.time_series_reasoning.release import SourceFile


FIXTURES = Path(__file__).parent / "fixtures"


def fixture_path(file: SourceFile, root: Path = FIXTURES) -> Path:
    """Return a source file's fixture under ``root``.

    The release's ``TS_Dataset_MCQ/*.json`` files are JSON Lines. Their fixtures end in ``.jsonl``, so
    the check-json hook doesn't parse them as one JSON document.
    """
    return root / Path(file.value).with_suffix(".jsonl")


def fixture_paths(root: Path = FIXTURES) -> dict[SourceFile, Path]:
    """Return every source file's fixture under ``root``."""
    return {file: fixture_path(file, root) for file in SourceFile}


def published_rows(file: SourceFile, root: Path = FIXTURES) -> list[dict]:
    """Return a fixture's rows as published, decoded without the connector."""
    return [json.loads(line) for line in fixture_path(file, root).read_bytes().splitlines()]
