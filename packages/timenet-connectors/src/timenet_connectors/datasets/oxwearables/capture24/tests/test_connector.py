from datetime import datetime, timedelta
import gzip
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types import ClassificationTask, Split
from timenet.types.text import read_value
from timenet_connectors.datasets.oxwearables.capture24 import Capture24Connector
from timenet_connectors.datasets.oxwearables.capture24.loader import Capture24DatasetLoader
from timenet_connectors.datasets.oxwearables.capture24.records import record_metadata
from timenet_connectors.datasets.oxwearables.capture24.release import PINNED_COUNTS, PinnedCounts, split_of
from timenet_connectors.datasets.oxwearables.capture24.tasks import Capture24WindowTask


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
_FIXTURE_COUNTS = PinnedCounts(participants=2, segments=3, windows=5)


class _FixtureConnector(Capture24Connector):
    """The connector, pinned to the fixtures' counts instead of the release's."""

    CARD = Path(__file__).parents[1] / "dataset.yaml"
    SOURCE_COUNTS: ClassVar[PinnedCounts] = _FIXTURE_COUNTS


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


def _release(tmp_path: Path) -> Path:
    root = tmp_path / "capture24"
    root.mkdir()
    (root / "annotation-label-dictionary.csv").write_text(_DICTIONARY, encoding="utf-8")
    (root / "metadata.csv").write_text("pid,age,sex\nP001,38-52,F\nP101,18-29,M\n", encoding="utf-8")
    _write_recording(root / "P001.csv.gz", _p001())
    _write_recording(root / "P101.csv.gz", _p101())
    return root


def _output(dataset) -> dict:
    """Everything the dataset holds, as plain values, with records keyed by participant."""
    records = {}
    for record in dataset.records:
        core, extension = record_metadata(record)
        records[core.subject_id] = {
            "file": extension.file,
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
            record_metadata(task.inputs[0])[0].subject_id,
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


def _runs(*runs: tuple[str, int, int]) -> list[tuple[str, str, int, int]]:
    return [("activity_label", label, start, end) for label, start, end in runs]


_P001_RUNS = _runs((_SLEEP, 0, 10_000_000), (_SIT, 10_000_000, 15_000_000), (_WALK, 15_000_000, 20_000_000))
_P001_RUNS += _runs((_WALK, 29_990_000, 30_000_000), (_SLEEP, 40_000_000, 45_000_000))
_P101_FIRST_RUNS = _runs((_SIT, 0, 15_000_000))
_P101_SECOND_RUNS = _runs((_WALK, _SECOND_SEGMENT_US, _SECOND_SEGMENT_US + 14_320_000))


def _signals(n_values: int, offset_us: int, runs: list) -> dict:
    return dict.fromkeys(("x", "y", "z"), (n_values, offset_us, "standard_gravity", runs))


# The whole converted fixture, written out by hand from the fixture rows above.
_EXPECTED = {
    "annotations": {
        "source_url": "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001/files/rpr76f381b",
        "source_sha256": "69740c22d3e000367988373336dc6486c10fe3f3f929811fd66e4d84861e40e2",
        "activity_label_vocabulary": [_SLEEP, _SIT, _WALK],
        "willetts_specific_2018_vocabulary": ["sitting", "sleep", "walking"],
        "willetts_specific_2018_mapping": ["sleep", "sitting", "walking"],
        "willetts_met_2018_vocabulary": ["sitting", "sleep", "walking"],
        "willetts_met_2018_mapping": ["sleep", "sitting", "walking"],
        "doherty_specific_2018_vocabulary": ["sedentary-screen", "sleep", "walking"],
        "doherty_specific_2018_mapping": ["sleep", "sedentary-screen", "walking"],
        "willetts_2018_vocabulary": ["sit-stand", "sleep", "walking"],
        "willetts_2018_mapping": ["sleep", "sit-stand", "walking"],
        "doherty_2018_vocabulary": ["sedentary", "sleep", "walking"],
        "doherty_2018_mapping": ["sleep", "sedentary", "walking"],
        "walmsley_2020_vocabulary": ["light", "sedentary", "sleep"],
        "walmsley_2020_mapping": ["sleep", "sedentary", "light"],
    },
    "records": {
        "P001": {
            "file": "P001.csv.gz",
            "annotations": {"age": "38-52", "sex": "F", "recording_start_local": "2016-11-13T02:18:00"},
            "sources": [{"name": "Axivity AX3 segment 0", "signals": _signals(4500, 0, _P001_RUNS)}],
        },
        "P101": {
            "file": "P101.csv.gz",
            "annotations": {"age": "18-29", "sex": "M", "recording_start_local": "2016-11-13T02:18:00"},
            "sources": [
                {"name": "Axivity AX3 segment 0", "signals": _signals(1500, 0, _P101_FIRST_RUNS)},
                {
                    "name": "Axivity AX3 segment 1",
                    "signals": _signals(1432, _SECOND_SEGMENT_US, _P101_SECOND_RUNS),
                },
            ],
        },
    },
    # One task per window and scheme: WillettsSpecific2018, then Walmsley2020.
    "tasks": [
        ("P001", 0, 10_000_000, "willetts_specific_2018_vocabulary", ("sleep",), "train"),
        ("P001", 0, 10_000_000, "walmsley_2020_vocabulary", ("sleep",), "train"),
        ("P001", 10_000_000, 20_000_000, "willetts_specific_2018_vocabulary", ("sitting",), "train"),
        ("P001", 10_000_000, 20_000_000, "walmsley_2020_vocabulary", ("light",), "train"),
        ("P001", 20_000_000, 30_000_000, "willetts_specific_2018_vocabulary", ("walking",), "train"),
        ("P001", 20_000_000, 30_000_000, "walmsley_2020_vocabulary", ("light",), "train"),
        ("P101", 0, 10_000_000, "willetts_specific_2018_vocabulary", ("sitting",), "test"),
        ("P101", 0, 10_000_000, "walmsley_2020_vocabulary", ("sedentary",), "test"),
        # The bin [3620 s, 3630 s) of the first segment's grid. Its samples start 5 ms into it, so the
        # scope does too, and ends where the segment does.
        ("P101", 3_620_005_000, 3_630_005_000, "willetts_specific_2018_vocabulary", ("walking",), "test"),
        ("P101", 3_620_005_000, 3_630_005_000, "walmsley_2020_vocabulary", ("light",), "test"),
    ],
}


def test_converts_the_fixture(tmp_path):
    dataset = _FixtureConnector().convert([_release(tmp_path)])
    assert _output(dataset) == _EXPECTED
    assert all(isinstance(task, ClassificationTask) for task in dataset.iter_tasks())


def test_signals_reread_their_segment(tmp_path):
    dataset = _FixtureConnector().convert([_release(tmp_path)])
    p101 = next(record for record in dataset.records if record_metadata(record)[0].subject_id == "P101")
    second = next(source for source in p101.sources if source.name.endswith("segment 1"))
    x = next(signal for signal in second.signals if signal.name == "x").to_numpy()
    assert x.dtype == np.float64
    # Row 1500 is the second segment's first sample. The values are the written text, parsed.
    np.testing.assert_array_equal(x, [float(f"{index * 0.001:.8f}") for index in range(1500, 2932)])


def test_storage_round_trip_keeps_the_whole_conversion(tmp_path):
    original = _FixtureConnector().convert([_release(tmp_path)])
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(original)
    with TimeFReader(registry.open_version("oxwearables/capture24", "1.0.0")) as reader:
        restored = reader.read()
    assert _output(original) == _EXPECTED
    assert _output(restored) == _EXPECTED


def test_tasks_of_selects_the_benchmark_split(tmp_path):
    loader = Capture24DatasetLoader.parse_from([_release(tmp_path)], counts=_FIXTURE_COUNTS)
    test_tasks = list(loader.tasks_of(Capture24WindowTask, split=Split.TEST))
    assert {record_metadata(task.record)[0].subject_id for task in test_tasks} == {"P101"}
    assert len(test_tasks) == 4


@pytest.mark.parametrize(
    ("number", "split"), [(1, Split.TRAIN), (100, Split.TRAIN), (101, Split.TEST), (151, Split.TEST)]
)
def test_split_follows_benchmark_py(number, split):
    assert split_of(number) is split


def test_pinned_counts_mismatch_fails(tmp_path):
    with pytest.raises(TimeFValidationError, match="the pinned archive holds"):
        Capture24DatasetLoader.parse_from([_release(tmp_path)], counts=PINNED_COUNTS)


def _rewrite(path: Path, edit) -> None:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(edit(lines)) + "\n")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda lines: ["time,x,y,z,label", *lines[1:]], "expected the header"),
        (lambda lines: [lines[0], lines[2], lines[1], *lines[3:]], "doesn't follow the one before it"),
        (lambda lines: [*lines[:5], lines[5].replace(_SLEEP, "unknown;MET 1.0"), *lines[6:]], "label dictionary"),
        (lambda lines: [*lines[:5], ",0.1,0.2,0.3,", *lines[6:]], "have no timestamp"),
    ],
)
def test_malformed_recording_fails_with_its_path(tmp_path, edit, message):
    root = _release(tmp_path)
    _rewrite(root / "P001.csv.gz", edit)
    with pytest.raises(TimeFFormatError, match=message):
        Capture24DatasetLoader.parse_from([root], counts=_FIXTURE_COUNTS)


@pytest.mark.parametrize(("value", "message"), [("", "have no x value"), ("nan", "is not finite")])
def test_bad_acceleration_fails_when_read(tmp_path, value, message):
    root = _release(tmp_path)
    _rewrite(root / "P001.csv.gz", lambda lines: [*lines[:5], _with_x(lines[5], value), *lines[6:]])
    dataset = _FixtureConnector().convert([root])
    p001 = next(record for record in dataset.records if record_metadata(record)[0].subject_id == "P001")
    x = next(signal for signal in p001.signals if signal.name == "x")
    with pytest.raises(TimeFFormatError, match=message):
        x.to_numpy()


def _with_x(line: str, value: str) -> str:
    time, _, *rest = line.split(",", 2)
    return ",".join([time, value, *rest])


def test_metadata_and_recordings_must_name_the_same_participants(tmp_path):
    root = _release(tmp_path)
    (root / "P101.csv.gz").unlink()
    with pytest.raises(TimeFFormatError, match=r"recordings missing for \['P101'\]"):
        Capture24DatasetLoader.parse_from([root], counts=_FIXTURE_COUNTS)
