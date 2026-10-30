import json
from pathlib import Path
import shutil

import pytest

from timenet.errors import TimeFValidationError
from timenet_connectors.datasets.mikeam.time_series_reasoning.index import ReleaseIndex
from timenet_connectors.datasets.mikeam.time_series_reasoning.keys import AnnotationKey
from timenet_connectors.datasets.mikeam.time_series_reasoning.records import ReleaseRecords, record_metadata
from timenet_connectors.datasets.mikeam.time_series_reasoning.release import SourceFile
from timenet_connectors.datasets.mikeam.time_series_reasoning.tests.release_fixtures import (
    FIXTURES,
    fixture_path,
    fixture_paths,
    published_rows,
)


def _index(root: Path = FIXTURES) -> ReleaseIndex:
    # The fixtures aren't the pinned release, so their row counts and blank lines aren't checked.
    return ReleaseIndex.from_files(fixture_paths(root), pinned=False)


def _copy_fixtures(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    shutil.copytree(FIXTURES, root)
    return root


def _rewrite(root: Path, file: SourceFile, line: int, change) -> None:
    """Rewrite one fixture line, from 1, with ``change`` applied to its decoded row."""
    path = fixture_path(file, root)
    rows = published_rows(file, root)
    change(rows[line - 1])
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_the_index_holds_the_series_the_description_rows_and_the_changed_series():
    index = _index()
    assert list(index.series) == [row["uuid"] for row in published_rows(SourceFile.SERIES)]
    assert [(row.ref.file.name, row.ref.line) for row in index.descriptions] == [
        ("DESCRIPTION_TRAIN", line) for line in range(1, 6)
    ] + [("DESCRIPTION_VAL", 1), ("DESCRIPTION_TEST", 1)]
    # One changed series per counterfactual: 04e3c8fe has two.
    assert [(uuid[:8], item.ref.line) for (uuid, _), item in index.changed.items()] == [
        ("04e3c8fe", 1),
        ("04e3c8fe", 2),
        ("93e40081", 3),
    ]


def test_every_file_is_required():
    paths = fixture_paths()
    del paths[SourceFile.COUNTERFACTUALS]
    with pytest.raises(TimeFValidationError, match=r"MCQ_2_TS\.jsonl"):
        ReleaseIndex.from_files(paths, pinned=False)


def _set_description(row: dict) -> None:
    row["description"] = "changed"


def _set_description_and_its_option(row: dict) -> None:
    # A description row's first option is its description, so both change together.
    row["description"] = row["options"][0] = "changed"


def _reorder_metadata(row: dict) -> None:
    row["metadata"] = dict(reversed(row["metadata"].items()))


def _change_value(row: dict) -> None:
    row["series"][0] += 1


def _unknown_uuid(row: dict) -> None:
    row["uuid"] = "00000000-0000-0000-0000-000000000000"


@pytest.mark.parametrize(
    ("file", "change", "message"),
    [
        (SourceFile.QUESTIONS, _set_description, r"\['description'\] differ"),
        (SourceFile.QUESTIONS, _reorder_metadata, r"\['metadata'\] differ"),
        (SourceFile.QUESTIONS, _change_value, r"\['series'\] differ"),
        (SourceFile.QUESTIONS, _unknown_uuid, r"is not in TS_Dataset\.jsonl"),
        (SourceFile.DESCRIPTION_VAL, _set_description_and_its_option, r"\['description'\] differ"),
    ],
)
def test_a_row_must_repeat_its_series_row(tmp_path, file, change, message):
    root = _copy_fixtures(tmp_path)
    _rewrite(root, file, 1, change)
    with pytest.raises(TimeFValidationError, match=message):
        _index(root)


def test_a_description_row_may_hold_other_values_than_its_series_row(tmp_path):
    root = _copy_fixtures(tmp_path)
    # Train line 3 holds the same values as its series' row, until they change.
    _rewrite(root, SourceFile.DESCRIPTION_TRAIN, 3, _change_value)
    assert len(ReleaseRecords.build(_index(root)).copies) == 5


def test_a_series_uuid_must_not_repeat(tmp_path):
    root = _copy_fixtures(tmp_path)
    path = fixture_path(SourceFile.SERIES, root)
    first = path.read_bytes().splitlines(keepends=True)[0]
    path.write_bytes(path.read_bytes() + first)
    with pytest.raises(TimeFValidationError, match="repeats line 1"):
        _index(root)


def test_the_rows_of_one_counterfactual_must_hold_one_changed_series(tmp_path):
    root = _copy_fixtures(tmp_path)
    path = fixture_path(SourceFile.COUNTERFACTUALS, root)
    row = published_rows(SourceFile.COUNTERFACTUALS, root)[0]
    row["new_series"][0] += 1
    path.write_bytes(path.read_bytes() + json.dumps(row).encode() + b"\n")
    with pytest.raises(TimeFValidationError, match="differs from line 1's"):
        _index(root)


def test_there_is_one_record_per_series_differing_copy_and_changed_series():
    records = ReleaseRecords.build(_index())
    # Which description rows have a copy: test_a_description_row_uses_its_copy_or_else_its_series_record.
    assert (len(records.series), len(records.copies), len(records.changed), len(records)) == (9, 4, 3, 16)
    assert len(list(records)) == 16


def test_a_series_record_holds_the_published_values_and_texts():
    records = ReleaseRecords.build(_index())
    for line, source in enumerate(published_rows(SourceFile.SERIES), start=1):
        record = records.series[source["uuid"]]
        (signal,) = record.signals
        assert signal.to_numpy().tolist() == source["series"]
        assert signal.spec.dtype == ("int32" if all(type(value) is int for value in source["series"]) else "float64")
        assert record_metadata(record).model_dump(exclude_none=True) == {
            "uuid": source["uuid"],
            "file": SourceFile.SERIES,
            "line": line,
        }
        assert {annotation.key: annotation.value for annotation in record.annotations} == {
            AnnotationKey.DESCRIPTION: source["description"],
            AnnotationKey.DESCRIPTION_SHORT: source["description_short"],
            AnnotationKey.DESCRIPTION_TINY: source["description_tiny"],
            AnnotationKey.CHARACTERISTICS: source["characteristics"],
            AnnotationKey.GENERATOR: source["generator"],
            AnnotationKey.METADATA: str(source["metadata"]),
        }


def test_a_description_row_uses_its_copy_or_else_its_series_record():
    index = _index()
    records = ReleaseRecords.build(index)
    split_files = (SourceFile.DESCRIPTION_TRAIN, SourceFile.DESCRIPTION_VAL, SourceFile.DESCRIPTION_TEST)
    published = {(file, line): row for file in split_files for line, row in enumerate(published_rows(file), start=1)}
    uses = {}
    for row in index.descriptions:
        record = records.of_description(row)
        assert record.signals[0].to_numpy().tolist() == published[row.ref.file, row.ref.line]["series"]
        uses[row.ref.file.name, row.ref.line] = "series" if record is records.series[row.uuid] else "copy"
    assert uses == {
        ("DESCRIPTION_TRAIN", 1): "copy",
        ("DESCRIPTION_TRAIN", 2): "copy",
        ("DESCRIPTION_TRAIN", 3): "series",
        ("DESCRIPTION_TRAIN", 4): "series",
        ("DESCRIPTION_TRAIN", 5): "copy",
        ("DESCRIPTION_VAL", 1): "copy",
        ("DESCRIPTION_TEST", 1): "copy",
    }
    # Train lines 1 and 5 repeat 8d269711 with the same values, so they share one copy.
    first, fifth = index.descriptions[0], index.descriptions[4]
    assert records.of_description(first) is records.of_description(fifth)


def test_a_changed_series_record_holds_its_change_and_generator():
    # Its values are checked by test_tasks.py, which compares every task's series with the authors'.
    index = _index()
    records = ReleaseRecords.build(index)
    published = published_rows(SourceFile.COUNTERFACTUALS)
    for (uuid, counterfactual), record in records.changed.items():
        line = index.changed[uuid, counterfactual].ref.line
        source = published[line - 1]
        assert record_metadata(record).model_dump(exclude_none=True) == {
            "uuid": uuid,
            "file": SourceFile.COUNTERFACTUALS,
            "line": line,
        }
        assert {annotation.key: annotation.value for annotation in record.annotations} == {
            AnnotationKey.COUNTERFACTUAL: source["counterfactual"],
            AnnotationKey.GENERATOR: source["new_generator"],
        }
