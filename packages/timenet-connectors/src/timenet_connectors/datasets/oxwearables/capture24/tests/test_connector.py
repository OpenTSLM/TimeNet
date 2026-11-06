import numpy as np

from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types.text import read_value
from timenet_connectors.datasets.oxwearables.capture24 import Capture24Connector, connector, schemes, source
from timenet_connectors.datasets.oxwearables.capture24.tests.release import (
    SECOND_SEGMENT_US,
    SIT,
    SLEEP,
    WALK,
    pid_of,
    release,
    write_recording,
)


def _output(dataset) -> dict:
    """Summarize the dataset structure, annotations, and tasks as plain values, with records keyed by participant."""
    records = {}
    for record in dataset.records:
        records[pid_of(record)] = {
            "metadata": dict(record.metadata),
            "annotations": {a.key: read_value(a.value) for a in record.annotations},
            "descriptions": {a.key: a.description for a in record.annotations if a.description is not None},
            "sources": [
                {
                    "name": source.name,
                    "annotations": [
                        (a.key, read_value(a.value), a.span.start_us, a.span.exclusive_end, dict(a.metadata))
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
            pid_of(task.inputs[0]),
            task.scope.start_us,
            task.scope.exclusive_end,
            # None: the scope names no signal, so the whole Record stays available, as the authors'
            # RNN and HMM read it. The Record's Sources still carry the activity annotations, which
            # hold the answer, so a consumer must drop them.
            task.scope.time_series,
            task.label_set.name,
            task.targets,
            task.split.value,
        )
        for task in dataset.iter_tasks()
    ]
    annotations = {a.key: read_value(a.value) for a in dataset.annotations}
    label_sets = {task.label_set.name: list(task.label_set.names) for task in dataset.iter_tasks()}
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


def _sleep(start_us: int, end_us: int) -> tuple:
    return ("activity", SLEEP, start_us, end_us, _SLEEP_CLASSES)


def _sit(start_us: int, end_us: int) -> tuple:
    return ("activity", SIT, start_us, end_us, _SIT_CLASSES)


def _walk(start_us: int, end_us: int) -> tuple:
    return ("activity", WALK, start_us, end_us, _WALK_CLASSES)


_WILLETTS = schemes.WILLETTS_SPECIFIC_2018.name
_WALMSLEY = schemes.WALMSLEY_2020.name

# Descriptions tell readers that age is a band and time is shifted.
# Other connectors store an exact age under the key age.
_DESCRIPTIONS = {
    "age_band": "The participant's age band in years, as metadata.csv writes it. The authors give no exact age.",
    "recording_start_local": (
        "The first sample's timestamp as the authors published it, with no zone. For anonymity they "
        "randomized the date and shifted the time by a small random amount. The time of day is as "
        "published, not an exact local time."
    ),
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
        "P001": {
            "metadata": {"file": "P001.csv.gz"},
            "descriptions": _DESCRIPTIONS,
            "annotations": {
                "subject_ids": ["P001"],
                "age_band": "38-52",
                "sex": "female",
                "recording_start_local": "2016-11-13T02:18:00.000000",
            },
            "sources": [
                {
                    "name": "Axivity AX3 segment 0",
                    # One run per change of raw label. The unlabeled rows 2000 to 2998 and 3000 to 3999
                    # belong to no run, and the one labeled row 2999 is a run of its own.
                    "annotations": [
                        _sleep(0, 10_000_000),
                        _sit(10_000_000, 15_000_000),
                        _walk(15_000_000, 20_000_000),
                        _walk(29_990_000, 30_000_000),
                        _sleep(40_000_000, 45_000_000),
                    ],
                    "signals": _signals(4500, 0),
                }
            ],
        },
        "P100": {
            "metadata": {"file": "P100.csv.gz"},
            "descriptions": _DESCRIPTIONS,
            "annotations": {
                "subject_ids": ["P100"],
                "age_band": "30-37",
                "sex": "female",
                "recording_start_local": "2016-11-13T02:18:00.000000",
            },
            "sources": [
                {"name": "Axivity AX3 segment 0", "annotations": [_sleep(0, 10_000_000)], "signals": _signals(1000, 0)}
            ],
        },
        "P101": {
            "metadata": {"file": "P101.csv.gz"},
            "descriptions": _DESCRIPTIONS,
            "annotations": {
                "subject_ids": ["P101"],
                "age_band": "18-29",
                "sex": "male",
                "recording_start_local": "2016-11-13T02:18:00.000000",
            },
            "sources": [
                {"name": "Axivity AX3 segment 0", "annotations": [_sit(0, 15_000_000)], "signals": _signals(1500, 0)},
                {
                    "name": "Axivity AX3 segment 1",
                    # The run stops at the gap: each segment's runs are its own.
                    "annotations": [_walk(SECOND_SEGMENT_US, SECOND_SEGMENT_US + 14_320_000)],
                    "signals": _signals(1432, SECOND_SEGMENT_US),
                },
            ],
        },
    },
    # One benchmark per scheme, as benchmark.py runs them: every participant's windows in time order.
    "tasks": [
        ("P001", 0, 10_000_000, None, _WILLETTS, ("sleep",), "train"),
        ("P001", 10_000_000, 20_000_000, None, _WILLETTS, ("sitting",), "train"),
        ("P001", 20_000_000, 30_000_000, None, _WILLETTS, ("walking",), "train"),
        # P100 is the last train participant and P101 the first test one.
        ("P100", 0, 10_000_000, None, _WILLETTS, ("sleep",), "train"),
        ("P101", 0, 10_000_000, None, _WILLETTS, ("sitting",), "test"),
        # The bin [3620 s, 3630 s) of the first segment's grid. Its samples start 5 ms into it, so the
        # scope does too, and ends where the segment does.
        ("P101", 3_620_005_000, 3_630_005_000, None, _WILLETTS, ("walking",), "test"),
        ("P001", 0, 10_000_000, None, _WALMSLEY, ("sleep",), "train"),
        ("P001", 10_000_000, 20_000_000, None, _WALMSLEY, ("light",), "train"),
        ("P001", 20_000_000, 30_000_000, None, _WALMSLEY, ("light",), "train"),
        ("P100", 0, 10_000_000, None, _WALMSLEY, ("sleep",), "train"),
        ("P101", 0, 10_000_000, None, _WALMSLEY, ("sedentary",), "test"),
        ("P101", 3_620_005_000, 3_630_005_000, None, _WALMSLEY, ("light",), "test"),
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
    """Check every segment's x, y and z against the text the fixture wrote, as float32."""
    expected = {}
    for key, rows in _SEGMENT_ROWS.items():
        expected[*key, "x"] = np.array([float(f"{index * 0.001:.8f}") for index in rows], dtype=np.float32)
        expected[*key, "y"] = np.array([float(f"{-index * 0.001:.8f}") for index in rows], dtype=np.float32)
        expected[*key, "z"] = np.full(len(rows), 0.98765432, dtype=np.float32)
    actual = {
        (pid_of(record), source.name, signal.name): signal.to_numpy()
        for record in dataset.records
        for source in record.sources
        for signal in source.signals
    }
    assert actual.keys() == expected.keys()
    for key, values in actual.items():
        assert values.dtype == np.float32
        np.testing.assert_array_equal(values, expected[key], err_msg=str(key))


def test_converts_the_fixture(tmp_path):
    dataset = Capture24Connector().convert([release(tmp_path)])
    assert _output(dataset) == _EXPECTED
    assert all(not task.input_annotations for task in dataset.iter_tasks())
    _assert_values(dataset)


def test_recording_start_keeps_one_format_when_it_has_microseconds(tmp_path):
    # isoformat() alone drops a zero fraction, which can give Records different timestamp formats.
    root = release(tmp_path)
    write_recording(root / "P001.csv.gz", [(10_000 + index * 10_000, SLEEP) for index in range(1000)])
    records = _output(Capture24Connector().convert([root]))["records"]
    assert records["P001"]["annotations"]["recording_start_local"] == "2016-11-13T02:18:00.010000"
    assert records["P100"]["annotations"]["recording_start_local"] == "2016-11-13T02:18:00.000000"


def test_card_pins_the_archive_the_connector_downloads():
    assert Capture24Connector().metadata().source_revision == f"sha256:{connector.ARCHIVE_SHA256}"


def test_storage_round_trip_keeps_the_whole_conversion(tmp_path):
    original = Capture24Connector().convert([release(tmp_path)])
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(original)
    with TimeFReader(registry.open_version("oxwearables/capture24", "1.0.0")) as reader:
        reader.verify()
        restored = reader.read()
        assert _output(restored) == _EXPECTED
        assert all(not task.input_annotations for task in restored.iter_tasks())
        _assert_values(restored)


def test_storing_reads_each_recording_once(tmp_path, monkeypatch):
    reads = []
    read_acceleration = source.read_acceleration

    def spy(summary):
        reads.append(summary.participant.path.name)
        return read_acceleration(summary)

    monkeypatch.setattr(source, "read_acceleration", spy)
    original = Capture24Connector().convert([release(tmp_path)])
    LocalRegistry(tmp_path / "registry").store(original)
    assert reads == ["P001.csv.gz", "P100.csv.gz", "P101.csv.gz"]


def test_segments_share_a_read_and_evicted_recordings_are_reread(tmp_path, monkeypatch):
    root = release(tmp_path)
    reads = []
    read_acceleration = source.read_acceleration

    def spy(summary):
        reads.append(summary.participant.pid)
        return read_acceleration(summary)

    monkeypatch.setattr(source, "read_acceleration", spy)
    dataset = Capture24Connector().convert([root])
    records = {pid_of(record): record for record in dataset.records}
    assert reads == []  # Conversion does not invoke the lazy reader.
    x_signals = [signal for signal in records["P101"].signals if signal.name == "x"]
    for signal, first_row, length in zip(x_signals, (0, 1500), (1500, 1432), strict=True):
        expected = np.array(
            [float(f"{index * 0.001:.8f}") for index in range(first_row, first_row + length)], dtype=np.float32
        )
        np.testing.assert_array_equal(signal.to_numpy(), expected)
    assert reads == ["P101"]

    records["P001"].signals[0].to_numpy()
    x_signals[1].to_numpy()
    assert reads == ["P101", "P001", "P101"]

    # A separate conversion must not reuse this conversion's cached arrays.
    other = Capture24Connector().convert([root])
    next(record for record in other.records if pid_of(record) == "P101").signals[0].to_numpy()
    assert reads == ["P101", "P001", "P101", "P101"]
