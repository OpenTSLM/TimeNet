from pathlib import Path

import pyarrow as pa
import pytest

from timenet.errors import TimeNetConnectorError
from timenet_connectors.datasets.oxwearables.capture24 import Capture24Connector
from timenet_connectors.datasets.oxwearables.capture24.tests.release import (
    SIT,
    SLEEP,
    pid_of,
    release,
    rewrite,
    with_x,
    write_recording,
)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda lines: ["time,x,y,z,label", *lines[1:]], "expected the header"),
        # A first field over the csv module's size limit raises csv.Error.
        (lambda lines: ["t" * 200_000, *lines[1:]], "field larger than field limit"),
        (
            lambda lines: [lines[0], lines[2], lines[1], *lines[3:]],
            "P001.csv.gz:3: the timestamp doesn't follow the one before it",
        ),
        (lambda lines: [*lines[:5], ",0.1,0.2,0.3,", *lines[6:]], "P001.csv.gz:6: no timestamp"),
        (lambda lines: [*lines[:5], with_x(lines[5], ""), *lines[6:]], "P001.csv.gz:6: no x value"),
    ],
)
def test_malformed_recording_fails_with_its_path(tmp_path, edit, message):
    root = release(tmp_path)
    rewrite(root / "P001.csv.gz", edit)
    with pytest.raises(TimeNetConnectorError, match=message) as error:
        Capture24Connector().convert([root])
    assert "P001.csv.gz" in str(error.value)


def _x_of(dataset, pid: str):
    record = next(record for record in dataset.records if pid_of(record) == pid)
    return next(signal for signal in record.signals if signal.name == "x")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda lines: lines[:-1], "holds 4499 rows, 4500 when it was read"),
        # The same number of rows: z of one row rewritten, and two rows swapped. The digest covers the
        # stored float32 values, so the rewrite changes a digit that float32 keeps.
        (lambda lines: [*lines[:5], lines[5].replace("0.98765432", "0.98766432"), *lines[6:]], "values differ"),
        (lambda lines: [*lines[:5], lines[6], lines[5], *lines[7:]], "values differ"),
    ],
)
def test_changed_recording_fails_when_reread(tmp_path, edit, message):
    root = release(tmp_path)
    dataset = Capture24Connector().convert([root])
    rewrite(root / "P001.csv.gz", edit)
    with pytest.raises(TimeNetConnectorError, match=message) as error:
        _x_of(dataset, "P001").to_numpy()
    assert "P001.csv.gz" in str(error.value)
    assert "the file changed" in str(error.value)


def test_a_release_with_no_participants_fails(tmp_path):
    # With no rows and no recordings, the two sets of participants are equal, and both are empty.
    root = release(tmp_path)
    for recording in root.glob("P*.csv.gz"):
        recording.unlink()
    (root / "metadata.csv").write_text("pid,age,sex\n", encoding="utf-8")
    with pytest.raises(TimeNetConnectorError, match="lists no participants"):
        Capture24Connector().convert([root])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda root: (root / "P101.csv.gz").unlink(), r"recordings missing for \['P101'\]"),
        # Reject recordings missing from metadata.csv instead of silently omitting them.
        (
            lambda root: (root / "P152.csv.gz").write_bytes((root / "P100.csv.gz").read_bytes()),
            r"unlisted recordings \['P152'\]",
        ),
    ],
    ids=["missing", "unlisted"],
)
def test_metadata_and_recordings_must_name_the_same_participants(tmp_path, change, message):
    root = release(tmp_path)
    change(root)
    with pytest.raises(TimeNetConnectorError, match=message):
        Capture24Connector().convert([root])


@pytest.mark.parametrize(
    ("table", "old", "new", "message"),
    [
        ("metadata.csv", "pid,age,sex", "pid,age", "expected the columns"),
        # Reject extra columns and fields instead of silently dropping them.
        ("metadata.csv", "pid,age,sex\n", "pid,age,sex,height\n", "expected the columns"),
        ("metadata.csv", "P101,18-29,M", "P101,18-29,M,180", "more fields than the header"),
        ("annotation-label-dictionary.csv", ",light\n", ",\n", "no value for 'label:Walmsley2020'"),
        ("annotation-label-dictionary.csv", "sleep,sleep,sleep,sleep,sleep,sleep\n", "sleep,sleep\n", "no value for"),
        ("metadata.csv", "P101,18-29,M", "P101,18-29,", "no value for 'sex'"),
        ("metadata.csv", "P101,18-29,M", "P101,18-29", "no value for 'sex'"),
        ("annotation-label-dictionary.csv", "label:Walmsley2020", "label:Other", "expected the columns"),
        # Reject an unknown scheme column, so that the connector checks every column it reads.
        (
            "annotation-label-dictionary.csv",
            "label:Walmsley2020\n",
            "label:Walmsley2020,label:Extra\n",
            "expected the columns",
        ),
        # A blank class fails in a scheme that has no tasks too, because the activity annotations keep it.
        (
            "annotation-label-dictionary.csv",
            "0.95,sleep,sleep,",
            "0.95,sleep,,",
            "no value for 'label:WillettsMET2018'",
        ),
        ("metadata.csv", "P101,18-29,M", "P101,18-29,X", "metadata.csv:4: unknown sex 'X'"),
        ("metadata.csv", "P101,18-29,M", "P101,52+,M", r"metadata.csv:4: unknown age '52\+'"),
        (
            "annotation-label-dictionary.csv",
            ",light\n",
            ",strolling\n",
            r"classes missing from label set 'walmsley_2020': \['strolling'\]",
        ),
    ],
)
def test_malformed_tables_fail_with_their_path(tmp_path, table, old, new, message):
    root = release(tmp_path)
    path = root / table
    path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    with pytest.raises(TimeNetConnectorError, match=message) as error:
        Capture24Connector().convert([root])
    assert table in str(error.value)


@pytest.mark.parametrize("table", ["metadata.csv", "annotation-label-dictionary.csv"])
@pytest.mark.parametrize(
    "damage",
    [Path.unlink, lambda path: path.write_bytes(b"\xff\xfe")],
    ids=["missing", "not-utf-8"],
)
def test_unreadable_tables_fail_with_their_path(tmp_path, table, damage):
    root = release(tmp_path)
    damage(root / table)
    with pytest.raises(TimeNetConnectorError) as error:
        Capture24Connector().convert([root])
    assert table in str(error.value)


def test_a_file_in_place_of_the_directory_fails(tmp_path):
    root = release(tmp_path)
    with pytest.raises(TimeNetConnectorError, match="loads from one directory"):
        Capture24Connector().convert([root / "metadata.csv"])


@pytest.mark.parametrize("label", [SIT, ""], ids=["labeled", "unlabeled"])
def test_window_across_a_short_gap_is_rejected_only_when_labeled(tmp_path, label):
    # A 15 ms step is a gap, yet the first 10 s bin still holds 1000 samples, 500 on each side of it.
    root = release(tmp_path)
    rows = [(index * 10_000, label) for index in range(500)]
    rows += [(4_990_000 + 15_000 + index * 10_000, label) for index in range(500)]
    write_recording(root / "P001.csv.gz", rows)
    if not label:
        dataset = Capture24Connector().convert([root])
        record = next(record for record in dataset.records if pid_of(record) == "P001")
        assert len(record.sources) == 2
        tasks = list(dataset.iter_tasks())
        assert tasks  # Other participants still produce tasks.
        assert all(pid_of(task.inputs[0]) != "P001" for task in tasks)
        return
    with pytest.raises(
        TimeNetConnectorError, match=r"P001\.csv\.gz:2: the window that starts here holds 1000 samples across a gap"
    ) as error:
        Capture24Connector().convert([root])
    assert "P001.csv.gz" in str(error.value)


def test_truncated_recording_fails_with_its_path(tmp_path):
    root = release(tmp_path)
    path = root / "P001.csv.gz"
    path.write_bytes(path.read_bytes()[:100])
    with pytest.raises(TimeNetConnectorError) as error:
        Capture24Connector().convert([root])
    assert "P001.csv.gz" in str(error.value)


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ("1e60", "out of float32 range"),
        ("-1e60", "out of float32 range"),
        ("1e400", "out of float32 range"),
        ("nan", "not finite"),
        ("inf", "not finite"),
        ("-inf", "not finite"),
    ],
)
@pytest.mark.parametrize("lazy_read", [False, True])
def test_nonfinite_and_overflow_values_have_distinct_diagnostics(tmp_path, value, reason, lazy_read):
    root = release(tmp_path)
    dataset = Capture24Connector().convert([root]) if lazy_read else None
    path = root / "P001.csv.gz"
    rewrite(path, lambda lines: [*lines[:5], with_x(lines[5], value), *lines[6:]])
    with pytest.raises(TimeNetConnectorError) as error:
        if dataset is None:
            Capture24Connector().convert([root])
        else:
            _x_of(dataset, "P001").to_numpy()
    assert str(error.value) == f"{path}:6: the x value is {reason}"


@pytest.mark.parametrize(
    ("edit", "lazy_read"),
    [
        (lambda lines: [*lines[:5], with_x(lines[5], "abc"), *lines[6:]], False),
        (lambda lines: [*lines[:5], with_x(lines[5], "abc"), *lines[6:]], True),
        (lambda lines: [*lines[:5], "garbage," + lines[5].split(",", 1)[1], *lines[6:]], False),
    ],
    ids=["bad-number", "bad-number-lazy", "bad-timestamp"],
)
def test_arrow_errors_keep_the_path_and_original_diagnostic(tmp_path, edit, lazy_read):
    root = release(tmp_path)
    dataset = Capture24Connector().convert([root]) if lazy_read else None
    path = root / "P001.csv.gz"
    rewrite(path, edit)
    with pytest.raises(TimeNetConnectorError) as error:
        if dataset is None:
            Capture24Connector().convert([root])
        else:
            _x_of(dataset, "P001").to_numpy()
    assert isinstance(error.value.__cause__, pa.ArrowException)
    assert str(error.value) == f"{path}: {error.value.__cause__}"


def test_late_unknown_label_names_its_first_row(tmp_path):
    root = release(tmp_path)
    path = root / "P001.csv.gz"
    rewrite(path, lambda lines: [*lines[:-1], lines[-1].rsplit(",", 1)[0] + ",unknown"])
    with pytest.raises(TimeNetConnectorError) as error:
        Capture24Connector().convert([root])
    assert str(error.value) == f"{path}:4501: raw label missing from the label dictionary: 'unknown'"


@pytest.mark.parametrize("table", ["metadata.csv", "annotation-label-dictionary.csv"])
def test_blank_table_rows_are_rejected_at_their_line(tmp_path, table):
    root = release(tmp_path)
    path = root / table
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join([*lines[:2], "\n", *lines[2:]]))
    with pytest.raises(TimeNetConnectorError) as error:
        Capture24Connector().convert([root])
    assert str(error.value) == f"{path}:3: blank row"


@pytest.mark.parametrize(
    ("table", "original", "replacement", "message"),
    [
        ("metadata.csv", "P001", " \t ", "no value for 'pid'"),
        ("annotation-label-dictionary.csv", SLEEP, " \t ", "no value for 'annotation'"),
        ("metadata.csv", "P001", '"P001"x', "',' expected after '\"'"),
        ("annotation-label-dictionary.csv", SLEEP, '"sleep"x', "',' expected after '\"'"),
    ],
)
def test_blank_values_and_malformed_quotes_name_the_table_line(tmp_path, table, original, replacement, message):
    root = release(tmp_path)
    path = root / table
    path.write_text(path.read_text(encoding="utf-8").replace(original, replacement, 1), encoding="utf-8")
    with pytest.raises(TimeNetConnectorError) as error:
        Capture24Connector().convert([root])
    assert str(error.value) == f"{path}:2: {message}"


@pytest.mark.parametrize(
    ("table", "name", "value"),
    [("metadata.csv", "pid", "P001"), ("annotation-label-dictionary.csv", "raw label", SLEEP)],
)
def test_duplicate_table_values_name_both_lines(tmp_path, table, name, value):
    root = release(tmp_path)
    path = root / table
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join([*lines, lines[1]]))
    with pytest.raises(TimeNetConnectorError) as error:
        Capture24Connector().convert([root])
    assert str(error.value) == (f"{path}:{len(lines) + 1}: a {name} is listed twice: {value!r}; first listed on line 2")


@pytest.mark.parametrize("lazy_read", [False, True])
def test_blank_recording_row_fails_at_its_line(tmp_path, lazy_read):
    root = release(tmp_path)
    dataset = Capture24Connector().convert([root]) if lazy_read else None
    path = root / "P001.csv.gz"
    rewrite(path, lambda lines: [*lines[:5], "", *lines[6:]])
    with pytest.raises(TimeNetConnectorError) as error:
        if dataset is None:
            Capture24Connector().convert([root])
        else:
            _x_of(dataset, "P001").to_numpy()
    missing = "x value" if lazy_read else "timestamp"
    assert str(error.value) == f"{path}:6: no {missing}"
