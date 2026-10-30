"""The fixture release: real rows of the pinned release, copied with every column as published.

``fixtures/<split>.parquet`` holds the rows below, in this order, with the release's schema. Each was
chosen as the smallest row of its kind, plus the copies that make a source issue:

- train: one row of each task, published question type and format, and one data transformation
  question per transform. Rows 126163 and 126422 are ``ordering`` rows that ask a true-or-false and
  a multiple-choice question. Row 4591 asks the classification question of val row 2950, under another
  UCR dataset's name, and row 135829 the puzzle of test row 39798, with another answer.
- val: row 2950.
- test: row 39798.

Regenerate them from a downloaded release with ``python -m`` this module and the release's folder.
"""

import sys

import pyarrow.parquet as pq


SOURCE_ROWS: dict[str, tuple[int, ...]] = {
    "train": (
        31492,
        344,
        58715,
        55206,
        97985,
        88586,
        105647,
        124117,
        108026,
        111614,
        68522,
        64476,
        126163,
        126422,
        70268,
        4591,
        135829,
    ),
    "val": (2950,),
    "test": (39798,),
}
"""The release rows each fixture file holds, by split, in fixture order."""


def write_fixtures(release: str, out: str) -> None:
    """Copy :data:`SOURCE_ROWS` from a downloaded release into fixture files.

    Args:
        release: The folder holding the release's ``<split>.parquet`` files.
        out: The folder to write the fixtures to.
    """
    for split, rows in SOURCE_ROWS.items():
        table = pq.read_table(f"{release}/{split}.parquet")
        pq.write_table(table.take(list(rows)).replace_schema_metadata(table.schema.metadata), f"{out}/{split}.parquet")


if __name__ == "__main__":
    write_fixtures(sys.argv[1], sys.argv[2])
