from datetime import datetime, timedelta
import gzip
from pathlib import Path

import numpy as np
import pytest

from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, open_local_version
from timenet.types.text import read_value
from timenet_connectors.datasets.oxwearables.capture24 import Capture24Connector, source_data


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
    """Write one ``time,x,y,z,annotation`` row per ``(offset_us, label)``.

    The row index determines x and y. Every row uses the same z value."""
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        handle.write("time,x,y,z,annotation\n")
        for index, (offset_us, label) in enumerate(rows):
            stamp = (_START + timedelta(microseconds=offset_us)).strftime("%Y-%m-%d %H:%M:%S.%f")
            quoted = f'"{label}"' if label else ""
            handle.write(f"{stamp},{index * 0.001:.8f},{-index * 0.001:.8f},0.98765432,{quoted}\n")


def _p001() -> list[tuple[int, str]]:
    """Four full bins and one half bin, sampled every 10 ms with no gap."""
    labels = (
        [_SLEEP] * 1000  # bin 0: sleep in every scheme
        + [_SIT] * 500
        + [_WALK] * 500  # bin 1: a tie, which goes to the class first in string order
        + [""] * 999
        + [_WALK]  # bin 2: one labeled sample, whose class the window takes
        + [""] * 1000  # bin 3: unlabeled, so dropped
        + [_SLEEP] * 500  # bin 4: 500 samples, so dropped
    )
    return [(index * 10_000, label) for index, label in enumerate(labels)]


def _p101() -> list[tuple[int, str]]:
    """Two segments separated by a gap: 1500 samples, then 1432.

    The last 1000 samples form one full bin."""
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


def _pid(record) -> str:
    """The participant id that a Record's ``subject_ids`` annotation holds."""
    (pid,) = next(read_value(a.value) for a in record.annotations if a.key == "subject_ids")
    return pid


def _key(record) -> tuple[str, int]:
    """The participant id and the segment index of a Record."""
    return _pid(record), record.metadata["segment"]


def _name(label_set) -> str:
    """The name that a label set's ``name`` annotation holds."""
    return next(read_value(a.value) for a in label_set.annotations if a.key == "name")


def _output(dataset) -> dict:
    """Summarize the dataset structure, annotations, and tasks as plain values, with records keyed by participant and segment."""
    records = {}
    for record in dataset.records:
        records[_key(record)] = {
            "metadata": dict(record.metadata),
            "annotations": {a.key: read_value(a.value) for a in record.annotations},
            "descriptions": {a.key: a.description for a in record.annotations if a.description is not None},
            "sources": [
                {
                    "name": source.name,
                    "annotations": [
                        (a.key, read_value(a.value), a.span.start_us, a.span.exclusive_end, a.description)
                        for a in source.annotations
                    ],
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
            _key(task.inputs[0]),
            task.scope.start_us,
            task.scope.exclusive_end,
            # None: the scope names no signal, so the whole Record stays available, as the authors'
            # RNN and HMM read it. The Record's Sources still carry the activity annotations, which
            # hold the answer, so a consumer must drop them.
            task.scope.time_series,
            _name(task.label_set),
            task.targets,
            task.split.value,
        )
        for task in dataset.iter_tasks()
    ]
    annotations = {a.key: read_value(a.value) for a in dataset.annotations}
    label_sets = {_name(task.label_set): list(task.label_set.names) for task in dataset.iter_tasks()}
    return {"annotations": annotations, "label_sets": label_sets, "records": records, "tasks": tasks}


def _signals(n_values: int, offset_us: int) -> dict:
    # No signal carries an annotation: the raw labels are on the Source.
    return dict.fromkeys(("x", "y", "z"), (n_values, offset_us, "standard_gravity", []))


# Each fixture raw label's class in every scheme of the fixture dictionary.
_SLEEP_CLASSES = dict.fromkeys(
    ("WillettsSpecific2018", "WillettsMET2018", "DohertySpecific2018", "Willetts2018", "Doherty2018", "Walmsley2020"),
    "sleep",
)
_SIT_CLASSES = {
    "WillettsSpecific2018": "sitting",
    "WillettsMET2018": "sitting",
    "DohertySpecific2018": "sedentary-screen",
    "Willetts2018": "sit-stand",
    "Doherty2018": "sedentary",
    "Walmsley2020": "sedentary",
}
_WALK_CLASSES = {**dict.fromkeys(_SLEEP_CLASSES, "walking"), "Walmsley2020": "light"}


_CLASSES = {_SLEEP: _SLEEP_CLASSES, _SIT: _SIT_CLASSES, _WALK: _WALK_CLASSES}


# The published annotation key of each scheme's class.
_SCHEME_KEYS = {
    "WillettsSpecific2018": "activity_willetts_specific_2018",
    "WillettsMET2018": "activity_willetts_met_2018",
    "DohertySpecific2018": "activity_doherty_specific_2018",
    "Willetts2018": "activity_willetts_2018",
    "Doherty2018": "activity_doherty_2018",
    "Walmsley2020": "activity_walmsley_2020",
}


def _run(label: str, start_us: int, end_us: int) -> list[tuple]:
    """The annotations of one run: its raw label, then its class in each scheme."""
    classes = [
        (_SCHEME_KEYS[scheme], name, start_us, end_us, f"The run's activity class in the {scheme} label scheme.")
        for scheme, name in _CLASSES[label].items()
    ]
    return [("activity", label, start_us, end_us, None), *classes]


_WILLETTS = "willetts_specific_2018"
_WALMSLEY = "walmsley_2020"

# Descriptions tell readers that age is a band and time is shifted.
# Other connectors store an exact age under the key age.
_DESCRIPTIONS = {
    "age_band": "The participant's age band in years.",
    "recording_start_local": (
        "The first sample's timestamp as the authors published it, with no zone. For anonymity they "
        "randomized the date and shifted the time by a small random amount."
    ),
}

_P101_ANNOTATIONS = {
    "subject_ids": ["P101"],
    "age_band": "18-29",
    "sex": "male",
    "recording_start_local": "2016-11-13T02:18:00.000000",
}

# The whole converted fixture, written out by hand from the fixture rows above.
_EXPECTED = {
    # The label sets hold the scheme's classes, so the dataset has no annotations of its own.
    "annotations": {},
    "label_sets": {
        # Each benchmarked scheme's whole label set, with the classes that the fixture does not use.
        "willetts_specific_2018": [
            "bicycling",
            "household-chores",
            "manual-work",
            "mixed-activity",
            "sitting",
            "sleep",
            "sports",
            "standing",
            "vehicle",
            "walking",
        ],
        "walmsley_2020": ["light", "moderate-vigorous", "sedentary", "sleep"],
    },
    "records": {
        ("P001", 0): {
            "metadata": {"file": "P001.csv.gz", "segment": 0},
            "descriptions": _DESCRIPTIONS,
            "annotations": {
                "subject_ids": ["P001"],
                "age_band": "38-52",
                "sex": "female",
                "recording_start_local": "2016-11-13T02:18:00.000000",
            },
            "sources": [
                {
                    "name": "Axivity AX3",
                    # One run per change of raw label. The unlabeled rows 2000 to 2998 and 3000 to 3999
                    # belong to no run, and the one labeled row 2999 is a run of its own.
                    "annotations": [
                        *_run(_SLEEP, 0, 10_000_000),
                        *_run(_SIT, 10_000_000, 15_000_000),
                        *_run(_WALK, 15_000_000, 20_000_000),
                        *_run(_WALK, 29_990_000, 30_000_000),
                        *_run(_SLEEP, 40_000_000, 45_000_000),
                    ],
                    "signals": _signals(4500, 0),
                }
            ],
        },
        ("P100", 0): {
            "metadata": {"file": "P100.csv.gz", "segment": 0},
            "descriptions": _DESCRIPTIONS,
            "annotations": {
                "subject_ids": ["P100"],
                "age_band": "30-37",
                "sex": "female",
                "recording_start_local": "2016-11-13T02:18:00.000000",
            },
            "sources": [
                {
                    "name": "Axivity AX3",
                    "annotations": _run(_SLEEP, 0, 10_000_000),
                    "signals": _signals(1000, 0),
                }
            ],
        },
        # P101's recording has a gap, so each of its two segments is a Record of its own.
        ("P101", 0): {
            "metadata": {"file": "P101.csv.gz", "segment": 0},
            "descriptions": _DESCRIPTIONS,
            "annotations": _P101_ANNOTATIONS,
            "sources": [
                {
                    "name": "Axivity AX3",
                    "annotations": _run(_SIT, 0, 15_000_000),
                    "signals": _signals(1500, 0),
                }
            ],
        },
        ("P101", 1): {
            "metadata": {"file": "P101.csv.gz", "segment": 1},
            "descriptions": _DESCRIPTIONS,
            "annotations": _P101_ANNOTATIONS,
            "sources": [
                {
                    "name": "Axivity AX3",
                    # The run stops at the gap: each segment's runs are its own.
                    "annotations": _run(_WALK, _SECOND_SEGMENT_US, _SECOND_SEGMENT_US + 14_320_000),
                    # The offset is from the recording's first sample, on the origin both parts share.
                    "signals": _signals(1432, _SECOND_SEGMENT_US),
                }
            ],
        },
    },
    # One benchmark per scheme, as benchmark.py runs them: every participant's windows in time order.
    "tasks": [
        (("P001", 0), 0, 10_000_000, None, _WILLETTS, ("sleep",), "train"),
        (("P001", 0), 10_000_000, 20_000_000, None, _WILLETTS, ("sitting",), "train"),
        (("P001", 0), 20_000_000, 30_000_000, None, _WILLETTS, ("walking",), "train"),
        # P100 is the last train participant and P101 the first test one.
        (("P100", 0), 0, 10_000_000, None, _WILLETTS, ("sleep",), "train"),
        (("P101", 0), 0, 10_000_000, None, _WILLETTS, ("sitting",), "test"),
        # The bin [3620 s, 3630 s) of the first segment's grid. Its samples start 5 ms into it, so the
        # scope does too, and ends where the segment does.
        (("P101", 1), 3_620_005_000, 3_630_005_000, None, _WILLETTS, ("walking",), "test"),
        (("P001", 0), 0, 10_000_000, None, _WALMSLEY, ("sleep",), "train"),
        (("P001", 0), 10_000_000, 20_000_000, None, _WALMSLEY, ("light",), "train"),
        (("P001", 0), 20_000_000, 30_000_000, None, _WALMSLEY, ("light",), "train"),
        (("P100", 0), 0, 10_000_000, None, _WALMSLEY, ("sleep",), "train"),
        (("P101", 0), 0, 10_000_000, None, _WALMSLEY, ("sedentary",), "test"),
        (("P101", 1), 3_620_005_000, 3_630_005_000, None, _WALMSLEY, ("light",), "test"),
    ],
}


# The rows of each recording that make up each segment, by participant and segment index.
_SEGMENT_ROWS = {
    ("P001", 0): range(4500),
    ("P100", 0): range(1000),
    ("P101", 0): range(1500),
    ("P101", 1): range(1500, 2932),
}


def _assert_values(dataset) -> None:
    """Check every segment's x, y and z against the text the fixture wrote, as float32."""
    expected = {}
    for key, rows in _SEGMENT_ROWS.items():
        expected[*key, "x"] = np.array([float(f"{index * 0.001:.8f}") for index in rows], dtype=np.float32)
        expected[*key, "y"] = np.array([float(f"{-index * 0.001:.8f}") for index in rows], dtype=np.float32)
        expected[*key, "z"] = np.full(len(rows), 0.98765432, dtype=np.float32)
    actual = {
        (*_key(record), signal.name): signal.to_numpy()
        for record in dataset.records
        for source in record.sources
        for signal in source.signals
    }
    assert actual.keys() == expected.keys()
    for key, values in actual.items():
        assert values.dtype == np.float32
        np.testing.assert_array_equal(values, expected[key], err_msg=str(key))


def _assert_origins(dataset) -> None:
    """Check that only the parts of one recording share a time origin."""
    origins = {_key(record): record.start_time for record in dataset.records}
    assert origins["P101", 0] is origins["P101", 1]
    assert len({id(origin) for origin in origins.values()}) == 3


def test_converts_the_fixture(tmp_path):
    dataset = Capture24Connector().convert(_release(tmp_path))
    assert _output(dataset) == _EXPECTED
    _assert_values(dataset)
    _assert_origins(dataset)


def test_recording_start_keeps_one_format_when_it_has_microseconds(tmp_path):
    # isoformat() alone drops a zero fraction, which can give Records different timestamp formats.
    root = _release(tmp_path)
    _write_recording(root / "P001.csv.gz", [(10_000 + index * 10_000, _SLEEP) for index in range(1000)])
    records = _output(Capture24Connector().convert(root))["records"]
    assert records["P001", 0]["annotations"]["recording_start_local"] == "2016-11-13T02:18:00.010000"
    assert records["P100", 0]["annotations"]["recording_start_local"] == "2016-11-13T02:18:00.000000"


def test_storage_round_trip_keeps_the_whole_conversion_and_reads_each_recording_once(tmp_path, monkeypatch):
    reads = []
    read_acceleration = source_data.read_acceleration

    def spy(recording):
        reads.append(recording.participant.path.name)
        return read_acceleration(recording)

    monkeypatch.setattr(source_data, "read_acceleration", spy)
    original = Capture24Connector().convert(_release(tmp_path))
    # A signal read before storing, as describe() does, still costs only one read of its recording.
    _x_of(original, "P001").to_numpy()
    registry = LocalRegistry(tmp_path / "registry")
    registry.put_dataset(original)
    assert reads == ["P001.csv.gz", "P100.csv.gz", "P101.csv.gz"]
    with TimeFReader(open_local_version(registry.root / "oxwearables/capture24/1.0.0")) as reader:
        restored = reader.read()
        assert _output(restored) == _EXPECTED
        _assert_values(restored)
        _assert_origins(restored)


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
        (lambda lines: [f"{line},extra" for line in lines], "expected the header"),
        (lambda lines: [lines[0], lines[2], lines[1], *lines[3:]], "doesn't follow the one before it"),
        (lambda lines: [*lines[:5], _with_x(lines[5], "nan"), *lines[6:]], "a value of x is not finite"),
    ],
)
def test_malformed_recording_fails_with_its_path(tmp_path, edit, message):
    root = _release(tmp_path)
    _rewrite(root / "P001.csv.gz", edit)
    with pytest.raises(TimeNetConnectorError, match=message) as error:
        Capture24Connector().convert(root)
    assert "P001.csv.gz" in str(error.value)


def _x_of(dataset, pid: str):
    record = next(record for record in dataset.records if _pid(record) == pid)
    return next(signal for signal in record.signals if signal.name == "x")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda lines: lines[:-1], "holds 4499 rows, 4500 when it was read"),
        # The same number of rows, with z of one row rewritten. The digest covers the stored float32
        # values, so the rewrite changes a digit that float32 keeps.
        (lambda lines: [*lines[:5], lines[5].replace("0.98765432", "0.98766432"), *lines[6:]], "values differ"),
    ],
)
def test_changed_recording_fails_when_reread(tmp_path, edit, message):
    root = _release(tmp_path)
    dataset = Capture24Connector().convert(root)
    _rewrite(root / "P001.csv.gz", edit)
    with pytest.raises(TimeNetConnectorError, match=message) as error:
        _x_of(dataset, "P001").to_numpy()
    assert "P001.csv.gz" in str(error.value)
    assert "the file changed" in str(error.value)


def test_a_recording_that_metadata_does_not_list_fails(tmp_path):
    # Reject the recording instead of silently omitting it.
    root = _release(tmp_path)
    (root / "P152.csv.gz").write_bytes((root / "P100.csv.gz").read_bytes())
    with pytest.raises(TimeNetConnectorError, match=r"unlisted recordings \['P152'\]"):
        Capture24Connector().convert(root)


@pytest.mark.parametrize(
    ("table", "old", "new", "message"),
    [
        ("metadata.csv", "P101,18-29,M", "P001,18-29,M", "a pid is listed twice"),
        # A blank or unknown class fails in a scheme that has no tasks too, because the activity
        # annotations keep it.
        ("annotation-label-dictionary.csv", "0.95,sleep,sleep,", "0.95,sleep,,", "label:WillettsMET2018"),
        (
            "annotation-label-dictionary.csv",
            ",sit-stand,",
            ",sit-and-stand,",
            r"classes missing from the class list of Willetts2018: \['sit-and-stand'\]",
        ),
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
    with pytest.raises(TimeNetConnectorError, match=message) as error:
        Capture24Connector().convert(root)
    assert table in str(error.value)


def test_window_across_a_short_gap_fails(tmp_path):
    # A 15 ms step is a gap, yet the first 10 s bin still holds 1000 samples, 500 on each side of it.
    root = _release(tmp_path)
    rows = [(index * 10_000, _SIT) for index in range(500)]
    rows += [(4_990_000 + 15_000 + index * 10_000, _SIT) for index in range(500)]
    _write_recording(root / "P001.csv.gz", rows)
    with pytest.raises(TimeNetConnectorError, match="across a gap") as error:
        Capture24Connector().convert(root)
    assert "P001.csv.gz" in str(error.value)
