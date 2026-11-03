from datetime import datetime, timedelta
import gzip
from pathlib import Path

import numpy as np
import pytest

from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types import TimeInterval
from timenet.types.text import read_value
from timenet_connectors.datasets.oxwearables.capture24 import Capture24Connector, connector


_SLEEP = "7030 sleeping;MET 0.95"
_SIT = "sitting;9060 sitting;MET 1.5"
_WALK = "walking;17190 walking;MET 3.5"
# The fixture dictionary, as the release writes it: a raw label, then its class in each of the six schemes.
_DICTIONARY = (
    "annotation,label:WillettsSpecific2018,label:WillettsMET2018,label:DohertySpecific2018,"
    "label:Willetts2018,label:Doherty2018,label:Walmsley2020\n"
    f"{_SLEEP},sleep,sleep,sleep,sleep,sleep,sleep\n"
    f'"{_SIT}",sitting,sitting,sedentary-screen,sit-stand,sedentary,sedentary\n'
    f'"{_WALK}",walking,walking,walking,walking,walking,light\n'
)
_START = datetime(2016, 11, 13, 2, 18)
# P101's second segment starts off the first segment's 10 s grid, after a gap of about an hour.
_SECOND_SEGMENT_US = 3_615_685_000


def _write_recording(path: Path, rows: list[tuple[int, str]]) -> None:
    """Write ``time,x,y,z,annotation`` rows, one per ``(offset_us, label)``, with x, y, z from the row index."""
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        handle.write("time,x,y,z,annotation\n")
        for index, (offset_us, label) in enumerate(rows):
            stamp = (_START + timedelta(microseconds=offset_us)).strftime("%Y-%m-%d %H:%M:%S.%f")
            quoted = f'"{label}"' if label else ""
            handle.write(f"{stamp},{index * 0.001:.8f},{-index * 0.001:.8f},0.98765432,{quoted}\n")


def _p001() -> list[tuple[int, str]]:
    """Four bins and a half, sampled every 10 ms with no gap."""
    labels = (
        [_SLEEP] * 1000  # bin 0: sleep in every scheme
        + [_SIT] * 500
        + [_WALK] * 500  # bin 1: a tie, which goes to the class first in string order
        + [""] * 999
        + [_WALK]  # bin 2: one labelled sample, whose class the window takes
        + [""] * 1000  # bin 3: unlabelled, so dropped
        + [_SLEEP] * 500  # bin 4: 500 samples, so dropped
    )
    return [(index * 10_000, label) for index, label in enumerate(labels)]


def _p101() -> list[tuple[int, str]]:
    """Two segments: 1500 samples, then 1432 after a gap, the last 1000 of them one full bin."""
    first = [(index * 10_000, _SIT) for index in range(1500)]
    second = [(_SECOND_SEGMENT_US + index * 10_000, _WALK) for index in range(1432)]
    return first + second


def _p100() -> list[tuple[int, str]]:
    """One bin of sleep: the last participant of the train split."""
    return [(index * 10_000, _SLEEP) for index in range(1000)]


def _release(tmp_path: Path) -> Path:
    root = tmp_path / "capture24"
    root.mkdir()
    (root / "annotation-label-dictionary.csv").write_text(_DICTIONARY, encoding="utf-8")
    (root / "metadata.csv").write_text("pid,age,sex\nP001,38-52,F\nP100,30-37,F\nP101,18-29,M\n", encoding="utf-8")
    _write_recording(root / "P001.csv.gz", _p001())
    _write_recording(root / "P100.csv.gz", _p100())
    _write_recording(root / "P101.csv.gz", _p101())
    return root


def _output(dataset) -> dict:
    """Everything the dataset holds, as plain values, with records keyed by participant."""
    records = {}
    for record in dataset.records:
        records[record.metadata["subject_id"]] = {
            "file": record.metadata["file"],
            "annotations": {a.key: read_value(a.value) for a in record.annotations},
            "sources": [
                {
                    "name": source.name,
                    "signals": {
                        s.name: (
                            s.n_values,
                            s.time_axis.offset_us,
                            str(s.spec.unit_value),
                            [
                                (a.key, read_value(a.value), a.span.start_us, a.span.exclusive_end)
                                for a in s.annotations
                            ],
                        )
                        for s in source.signals
                    },
                }
                for source in sorted(record.sources, key=lambda source: source.name)
            ],
        }
    tasks = [
        (
            task.inputs[0].metadata["subject_id"],
            task.scope.start_us,
            task.scope.exclusive_end,
            task.target_schema,
            task.targets,
            task.split.value,
        )
        for task in dataset.iter_tasks()
    ]
    annotations = {a.key: read_value(a.value) for a in dataset.annotations}
    return {"annotations": annotations, "records": records, "tasks": tasks}


def _signals(n_values: int, offset_us: int) -> dict:
    # No signal carries an annotation: the raw labels stay out of the dataset.
    return dict.fromkeys(("x", "y", "z"), (n_values, offset_us, "standard_gravity", []))


# The whole converted fixture, written out by hand from the fixture rows above.
_EXPECTED = {
    "annotations": {
        "source_url": "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001/files/rpr76f381b",
        "source_sha256": "69740c22d3e000367988373336dc6486c10fe3f3f929811fd66e4d84861e40e2",
        # Only the classes of the two benchmarked schemes, which the tasks' target_schema name.
        "willetts_specific_2018_vocabulary": ["sitting", "sleep", "walking"],
        "walmsley_2020_vocabulary": ["light", "sedentary", "sleep"],
    },
    "records": {
        "P001": {
            "file": "P001.csv.gz",
            "annotations": {"age": "38-52", "sex": "F", "recording_start_local": "2016-11-13T02:18:00"},
            "sources": [{"name": "Axivity AX3 segment 0", "signals": _signals(4500, 0)}],
        },
        "P100": {
            "file": "P100.csv.gz",
            "annotations": {"age": "30-37", "sex": "F", "recording_start_local": "2016-11-13T02:18:00"},
            "sources": [{"name": "Axivity AX3 segment 0", "signals": _signals(1000, 0)}],
        },
        "P101": {
            "file": "P101.csv.gz",
            "annotations": {"age": "18-29", "sex": "M", "recording_start_local": "2016-11-13T02:18:00"},
            "sources": [
                {"name": "Axivity AX3 segment 0", "signals": _signals(1500, 0)},
                {
                    "name": "Axivity AX3 segment 1",
                    "signals": _signals(1432, _SECOND_SEGMENT_US),
                },
            ],
        },
    },
    # One task per window and scheme: WillettsSpecific2018, then Walmsley2020.
    "tasks": [
        ("P001", 0, 10_000_000, "willetts_specific_2018_vocabulary", ("sleep",), "train"),
        ("P001", 0, 10_000_000, "walmsley_2020_vocabulary", ("sleep",), "train"),
        (
            "P001",
            10_000_000,
            20_000_000,
            "willetts_specific_2018_vocabulary",
            ("sitting",),
            "train",
        ),
        ("P001", 10_000_000, 20_000_000, "walmsley_2020_vocabulary", ("light",), "train"),
        (
            "P001",
            20_000_000,
            30_000_000,
            "willetts_specific_2018_vocabulary",
            ("walking",),
            "train",
        ),
        ("P001", 20_000_000, 30_000_000, "walmsley_2020_vocabulary", ("light",), "train"),
        # P100 is the last train participant and P101 the first test one.
        ("P100", 0, 10_000_000, "willetts_specific_2018_vocabulary", ("sleep",), "train"),
        ("P100", 0, 10_000_000, "walmsley_2020_vocabulary", ("sleep",), "train"),
        ("P101", 0, 10_000_000, "willetts_specific_2018_vocabulary", ("sitting",), "test"),
        ("P101", 0, 10_000_000, "walmsley_2020_vocabulary", ("sedentary",), "test"),
        # The bin [3620 s, 3630 s) of the first segment's grid. Its samples start 5 ms into it, so the
        # scope does too, and ends where the segment does.
        (
            "P101",
            3_620_005_000,
            3_630_005_000,
            "willetts_specific_2018_vocabulary",
            ("walking",),
            "test",
        ),
        ("P101", 3_620_005_000, 3_630_005_000, "walmsley_2020_vocabulary", ("light",), "test"),
    ],
}


# The rows of each recording that make up each segment, by participant and segment name.
_SEGMENT_ROWS = {
    ("P001", "Axivity AX3 segment 0"): range(4500),
    ("P100", "Axivity AX3 segment 0"): range(1000),
    ("P101", "Axivity AX3 segment 0"): range(1500),
    ("P101", "Axivity AX3 segment 1"): range(1500, 2932),
}


def _assert_values(dataset) -> None:
    """Check every segment's x, y and z against the text the fixture wrote, parsed."""
    expected = {}
    for key, rows in _SEGMENT_ROWS.items():
        expected[*key, "x"] = [float(f"{index * 0.001:.8f}") for index in rows]
        expected[*key, "y"] = [float(f"{-index * 0.001:.8f}") for index in rows]
        expected[*key, "z"] = [0.98765432] * len(rows)
    actual = {
        (record.metadata["subject_id"], source.name, signal.name): signal.to_numpy()
        for record in dataset.records
        for source in record.sources
        for signal in source.signals
    }
    assert actual.keys() == expected.keys()
    for key, values in actual.items():
        assert values.dtype == np.float64
        np.testing.assert_array_equal(values, expected[key], err_msg=str(key))


def test_converts_the_fixture(tmp_path):
    dataset = Capture24Connector().convert([_release(tmp_path)])
    assert _output(dataset) == _EXPECTED
    _assert_values(dataset)


def test_storage_round_trip_keeps_the_whole_conversion(tmp_path):
    original = Capture24Connector().convert([_release(tmp_path)])
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(original)
    with TimeFReader(registry.open_version("oxwearables/capture24", "1.0.0")) as reader:
        restored = reader.read()
        assert _output(restored) == _EXPECTED
        _assert_values(restored)


def test_storing_reads_each_recording_once(tmp_path, monkeypatch):
    reads = []
    read_acceleration = connector._read_acceleration

    def spy(participant):
        reads.append(participant.person.path.name)
        return read_acceleration(participant)

    monkeypatch.setattr(connector, "_read_acceleration", spy)
    original = Capture24Connector().convert([_release(tmp_path)])
    LocalRegistry(tmp_path / "registry").store(original)
    assert reads == ["P001.csv.gz", "P100.csv.gz", "P101.csv.gz"]


def test_tasks_mark_only_their_window_as_input(tmp_path):
    # The scope is the window and names no signal, so the whole Record stays available, as the authors'
    # RNN and HMM use it. The window's samples are 1000 of the x, y and z of the one segment that holds
    # it, as make_windows cuts X. No other field gives a model anything.
    dataset = Capture24Connector().convert([_release(tmp_path)])
    for task in dataset.iter_tasks():
        scope = task.scope
        assert isinstance(scope, TimeInterval)
        assert scope.exclusive_end - scope.start_us == 10_000_000
        assert scope.time_series is None
        (holding,) = [
            source
            for source in task.inputs[0].sources
            if (span := source.signals[0].span_us) is not None
            and span[0] <= scope.start_us
            and scope.exclusive_end <= span[1]
        ]
        assert sorted(signal.name for signal in holding.signals) == ["x", "y", "z"]
        for signal in holding.signals:
            axis = signal.time_axis
            assert isinstance(axis, RegularAxis)
            assert axis.index_at_or_after(scope.exclusive_end) - axis.index_at_or_after(scope.start_us) == 1000
        assert task.input_annotations == ()
        assert task.prompt is None
        assert task.rationale is None
        assert dict(task.metadata) == {}


def _rewrite(path: Path, edit) -> None:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(edit(lines)) + "\n")


def _with_x(line: str, value: str) -> str:
    time, _, *rest = line.split(",", 2)
    return ",".join([time, value, *rest])


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda lines: ["time,x,y,z,label", *lines[1:]], "expected the header"),
        (lambda lines: [lines[0], lines[2], lines[1], *lines[3:]], "doesn't follow the one before it"),
        (lambda lines: [*lines[:5], lines[5].replace(_SLEEP, "unknown;MET 1.0"), *lines[6:]], "label dictionary"),
        (lambda lines: [*lines[:5], ",0.1,0.2,0.3,", *lines[6:]], "have no timestamp"),
        (lambda lines: [*lines[:5], _with_x(lines[5], ""), *lines[6:]], "have no x value"),
        (lambda lines: [*lines[:5], _with_x(lines[5], "nan"), *lines[6:]], "a x value is not finite"),
    ],
)
def test_malformed_recording_fails_with_its_path(tmp_path, edit, message):
    root = _release(tmp_path)
    _rewrite(root / "P001.csv.gz", edit)
    with pytest.raises(TimeFFormatError, match=message) as error:
        Capture24Connector().convert([root])
    assert "P001.csv.gz" in str(error.value)


def _x_of(dataset, pid: str):
    record = next(record for record in dataset.records if record.metadata["subject_id"] == pid)
    return next(signal for signal in record.signals if signal.name == "x")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda lines: lines[:-1], "holds 4499 rows, 4500 when it was read"),
        # The same number of rows: z of one row rewritten, and two rows swapped.
        (lambda lines: [*lines[:5], lines[5].replace("0.98765432", "0.98765433"), *lines[6:]], "values differ"),
        (lambda lines: [*lines[:5], lines[6], lines[5], *lines[7:]], "values differ"),
    ],
)
def test_changed_recording_fails_when_reread(tmp_path, edit, message):
    root = _release(tmp_path)
    dataset = Capture24Connector().convert([root])
    _rewrite(root / "P001.csv.gz", edit)
    with pytest.raises(TimeFFormatError, match=message) as error:
        _x_of(dataset, "P001").to_numpy()
    assert "P001.csv.gz" in str(error.value)
    assert "the file changed" in str(error.value)


def test_metadata_and_recordings_must_name_the_same_participants(tmp_path):
    root = _release(tmp_path)
    (root / "P101.csv.gz").unlink()
    with pytest.raises(TimeFFormatError, match=r"recordings missing for \['P101'\]"):
        Capture24Connector().convert([root])


@pytest.mark.parametrize(
    ("table", "old", "new", "message"),
    [
        ("metadata.csv", "P101,18-29,M", "P001,18-29,M", "a pid is listed twice"),
        ("metadata.csv", "pid,age,sex", "pid,age", "expected the columns"),
        ("annotation-label-dictionary.csv", ",light\n", ",\n", "no value for 'label:Walmsley2020'"),
        ("annotation-label-dictionary.csv", "sleep,sleep,sleep,sleep,sleep,sleep\n", "sleep,sleep\n", "no value for"),
        ("metadata.csv", "P101,18-29,M", "P101,18-29,", "no value for 'sex'"),
        ("metadata.csv", "P101,18-29,M", "P101,18-29", "no value for 'sex'"),
        ("annotation-label-dictionary.csv", "label:Walmsley2020", "label:Other", "expected the columns"),
        (
            "annotation-label-dictionary.csv",
            '"walking;17190',
            '"sitting;9060 sitting;MET 1.5",x,x,x,x,x,x\n"walking;17190',
            "a raw label is listed twice",
        ),
    ],
)
def test_malformed_tables_fail_with_their_path(tmp_path, table, old, new, message):
    root = _release(tmp_path)
    path = root / table
    path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    with pytest.raises(TimeFFormatError, match=message) as error:
        Capture24Connector().convert([root])
    assert table in str(error.value)


def test_window_across_a_short_gap_fails(tmp_path):
    # A 15 ms step is a gap, yet the first 10 s bin still holds 1000 samples, 500 on each side of it.
    root = _release(tmp_path)
    rows = [(index * 10_000, _SIT) for index in range(500)]
    rows += [(4_990_000 + 15_000 + index * 10_000, _SIT) for index in range(500)]
    _write_recording(root / "P001.csv.gz", rows)
    with pytest.raises(TimeFFormatError, match="across a gap") as error:
        Capture24Connector().convert([root])
    assert "P001.csv.gz" in str(error.value)


def test_truncated_recording_fails_with_its_path(tmp_path):
    root = _release(tmp_path)
    path = root / "P001.csv.gz"
    path.write_bytes(path.read_bytes()[:100])
    with pytest.raises(TimeFFormatError) as error:
        Capture24Connector().convert([root])
    assert "P001.csv.gz" in str(error.value)
