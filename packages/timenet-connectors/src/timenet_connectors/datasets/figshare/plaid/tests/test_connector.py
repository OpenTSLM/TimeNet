import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeFFormatError, TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types import ClassificationTask, LocalizationMode, TemporalLocalizationTask, TimePoint
from timenet.types.text import read_value
from timenet_connectors.datasets.figshare.plaid import PlaidConnector
from timenet_connectors.datasets.figshare.plaid.loader import PlaidDatasetLoader
from timenet_connectors.datasets.figshare.plaid.records import record_metadata
from timenet_connectors.datasets.figshare.plaid.release import (
    KNOWN_DEFECTS,
    PINNED_COUNTS,
    KnownDefects,
    PinnedCounts,
    Subset,
)


_ROWS = 300
"""Every fixture file's rows: 0.01 s at 30 kHz, which the entries state as ``0.01s``."""
_FIXTURE_COUNTS = PinnedCounts(submetered=3, aggregated=2, events=8, households=3)
_FIXTURE_DEFECTS = KnownDefects(
    unlisted_files={Subset.SUBMETERED: frozenset(), Subset.AGGREGATED: frozenset({"0.csv"})},
    misstated_lengths={Subset.SUBMETERED: frozenset(), Subset.AGGREGATED: frozenset({2})},
)


class _FixtureConnector(PlaidConnector):
    """The connector, pinned to the fixtures' counts and defects instead of the release's."""

    CARD = Path(__file__).parents[1] / "dataset.yaml"
    SOURCE_COUNTS: ClassVar[PinnedCounts] = _FIXTURE_COUNTS
    SOURCE_DEFECTS: ClassVar[KnownDefects] = _FIXTURE_DEFECTS


def _appliance(appliance_type: str, load: str, **facts: str) -> dict[str, str]:
    """An appliance's fields, as the metadata writes them: every fact blank unless given."""
    blank = dict.fromkeys(("brand", "current", "manufacture_year", "model_number", "notes", "voltage", "wattage"), "")
    return {**blank, **facts, "type": appliance_type, "load": load}


def _header(collection_time: str, notes: str = "") -> dict[str, str]:
    return {"collection_time": collection_time, "notes": notes, "sampling_frequency": "30000Hz"}


def _submetered_entries() -> dict:
    # The 2017 layout writes the appliance's notes as appliance_notes, always blank.
    kettle = _appliance("Water kettle", "R")
    kettle["appliance_notes"] = kettle.pop("notes")
    return {
        "1": {
            "appliance": {**_appliance("Fan", "I", brand="Lasko"), "status": "high"},
            "header": _header("July, 2013", "some instances are not well calibrated"),
            "instances": {"length": "0.01s"},
            "location": "house1",
        },
        # The same location in another campaign is another household.
        "2": {
            "appliance": {**_appliance("Laptop", "NL"), "status": "off-on"},
            "header": _header("December, 2014"),
            "instances": {"length": "0.01s"},
            "location": "house1",
        },
        "3": {
            "appliance": {**kettle, "status": "off-on"},
            "header": _header("August, 2017"),
            "instances": {"length": "0.01s"},
            "location": "CMU lab",
        },
    }


def _aggregated_entries() -> dict:
    return {
        # The soldering iron switches on in two steps, both labelled; the status counts them once.
        "1": {
            "appliances": [
                {**_appliance("Fan", "I"), "on": "[10]", "off": "[250]"},
                {**_appliance("Iron solder", "NL"), "on": "[ 20 200]", "off": "[280]"},
            ],
            "header": _header("August, 2017"),
            "instances": {"length": "0.01s", "status": "on-on-off-off"},
            "location": "CMU lab",
        },
        # The air conditioner is never labelled off, and the stated length is wrong, as pinned.
        "2": {
            "appliances": [
                {**_appliance("Air Conditioner", "NL", wattage="740W"), "on": "[5]", "off": "[]"},
                {**_appliance("Vacuum", "I"), "on": "[31]", "off": "[100]"},
            ],
            "header": _header("August, 2017"),
            "instances": {"length": "0.02s", "status": "on-on-off-off"},
            "location": "CMU lab",
        },
    }


def _value(row: int) -> tuple[float, float]:
    """The current and voltage written in a fixture row."""
    return round(row * 0.001, 3), round(120.0 - row * 0.25, 2)


def _write_csv(path: Path, rows: int = _ROWS) -> None:
    path.write_text("".join("{},{}\n".format(*_value(row)) for row in range(rows)), encoding="utf-8")


def _release(tmp_path: Path) -> Path:
    root = tmp_path / "plaid"
    for subset, entries in ((Subset.SUBMETERED, _submetered_entries()), (Subset.AGGREGATED, _aggregated_entries())):
        (root / subset.directory).mkdir(parents=True)
        (root / subset.metadata.name).write_text(json.dumps(entries), encoding="utf-8")
        for number in entries:
            _write_csv(root / subset.directory / f"{number}.csv")
    _write_csv(root / Subset.AGGREGATED.directory / "0.csv")
    return root


_AXIS = RegularAxis.from_rate_hz(30_000)


def _output(dataset) -> dict:
    """Everything the dataset holds, as plain values, with records and tasks keyed by file."""
    records = {}
    for record in dataset.records:
        metadata = record_metadata(record)
        records[metadata.file] = {
            "subject_id": metadata.subject_id,
            "status": metadata.status,
            "length": metadata.length,
            "appliances": [appliance.model_dump(mode="json", exclude_none=True) for appliance in metadata.appliances],
            "sources": [
                (source.name, sorted((s.name, s.n_values, str(s.spec.unit_value)) for s in source.signals))
                for source in record.sources
            ],
            "events": [
                (a.key, read_value(a.value), _AXIS.index_at_or_after(a.span.start_us)) for a in record.annotations
            ],
        }
    tasks = []
    for task in dataset.iter_tasks():
        file = record_metadata(task.inputs[0]).file
        if isinstance(task, ClassificationTask):
            tasks.append((file, [read_value(t) for t in task.targets], task.target_schema, task.split))
        else:
            assert isinstance(task, TemporalLocalizationTask)
            events = [
                (a.key, read_value(a.value), _AXIS.index_at_or_after(a.span.start_us)) for a in task.target_annotations
            ]
            tasks.append((file, events, task.mode, task.split))
    annotations = {a.key: read_value(a.value) for a in dataset.annotations}
    return {"annotations": annotations, "records": records, "tasks": tasks}


_SIGNALS = [("NI-9215", [("current", _ROWS, "ampere"), ("voltage", _ROWS, "volt")])]
_EVENTS_1 = [
    ("appliance_on", "Fan", 10),
    ("appliance_on", "Iron solder", 20),
    ("appliance_on", "Iron solder", 200),
    ("appliance_off", "Fan", 250),
    ("appliance_off", "Iron solder", 280),
]
_EVENTS_2 = [
    ("appliance_on", "Air Conditioner", 5),
    ("appliance_on", "Vacuum", 31),
    ("appliance_off", "Vacuum", 100),
]

# The whole converted fixture, written out by hand from the fixture entries above.
_EXPECTED = {
    "annotations": {
        "source_url": "https://doi.org/10.6084/m9.figshare.10084619.v2",
        "source_files": [
            f"{file.name} sha256:{file.sha256}" for subset in Subset for file in (subset.metadata, subset.archive)
        ],
        # Only the submetered types, sorted, which the appliance task names.
        "appliance_type_vocabulary": ["Fan", "Laptop", "Water kettle"],
    },
    "records": {
        "submetered_new/1.csv": {
            "subject_id": "July, 2013_house1",
            "status": "high",
            "length": "0.01s",
            "appliances": [{"type": "Fan", "load": "I", "brand": "Lasko"}],
            "sources": _SIGNALS,
            "events": [],
        },
        "submetered_new/2.csv": {
            "subject_id": "December, 2014_house1",
            "status": "off-on",
            "length": "0.01s",
            "appliances": [{"type": "Laptop", "load": "NL"}],
            "sources": _SIGNALS,
            "events": [],
        },
        "submetered_new/3.csv": {
            "subject_id": "August, 2017_CMU lab",
            "status": "off-on",
            "length": "0.01s",
            "appliances": [{"type": "Water kettle", "load": "R"}],
            "sources": _SIGNALS,
            "events": [],
        },
        "aggregated/1.csv": {
            "subject_id": "August, 2017_CMU lab",
            "status": "on-on-off-off",
            "length": "0.01s",
            "appliances": [{"type": "Fan", "load": "I"}, {"type": "Iron solder", "load": "NL"}],
            "sources": _SIGNALS,
            "events": _EVENTS_1,
        },
        "aggregated/2.csv": {
            "subject_id": "August, 2017_CMU lab",
            "status": "on-on-off-off",
            # As written, though the file lasts 0.01 s.
            "length": "0.02s",
            "appliances": [
                {"type": "Air Conditioner", "load": "NL", "wattage": "740W"},
                {"type": "Vacuum", "load": "I"},
            ],
            "sources": _SIGNALS,
            "events": _EVENTS_2,
        },
    },
    "tasks": [
        ("submetered_new/1.csv", ["Fan"], "appliance_type_vocabulary", None),
        ("submetered_new/2.csv", ["Laptop"], "appliance_type_vocabulary", None),
        ("submetered_new/3.csv", ["Water kettle"], "appliance_type_vocabulary", None),
        ("aggregated/1.csv", _EVENTS_1, LocalizationMode.SPARSE, None),
        ("aggregated/2.csv", _EVENTS_2, LocalizationMode.SPARSE, None),
    ],
}


def test_converts_the_fixture(tmp_path):
    dataset = _FixtureConnector().convert([_release(tmp_path)])
    assert _output(dataset) == _EXPECTED


def test_storage_round_trip_keeps_the_whole_conversion(tmp_path):
    original = _FixtureConnector().convert([_release(tmp_path)])
    registry = LocalRegistry(tmp_path / "registry")
    registry.store(original)
    with TimeFReader(registry.open_version("figshare/plaid", "1.0.0")) as reader:
        restored = reader.read()
    assert _output(original) == _EXPECTED
    assert _output(restored) == _EXPECTED


def test_signals_reread_their_file(tmp_path):
    dataset = _FixtureConnector().convert([_release(tmp_path)])
    record = next(r for r in dataset.records if record_metadata(r).file == "aggregated/2.csv")
    signals = {signal.name: signal for signal in record.walk_signals()}
    expected = np.array([_value(row) for row in range(_ROWS)])
    for column, name in enumerate(("current", "voltage")):
        values = signals[name].to_numpy()
        assert values.dtype == np.float64
        np.testing.assert_array_equal(values, expected[:, column])
        assert signals[name].time_axis == _AXIS


def test_events_sit_at_their_sample_floored_to_microseconds(tmp_path):
    # Sample 10 is at 333.33 us, so its point is 333 us; sample 200 is at 6666.67 us, so 6666 us.
    dataset = _FixtureConnector().convert([_release(tmp_path)])
    record = next(r for r in dataset.records if record_metadata(r).file == "aggregated/1.csv")
    starts = []
    for annotation in record.annotations:
        span = annotation.span
        assert isinstance(span, TimePoint)
        assert span.time_series is None
        starts.append(span.start_us)
    assert starts == [333, 666, 6666, 8333, 9333]


def test_tasks_hold_nothing_beyond_their_record(tmp_path):
    # The appliance type is a target only: no prompt, input annotation or task metadata repeats it.
    dataset = _FixtureConnector().convert([_release(tmp_path)])
    for task in dataset.iter_tasks():
        assert len(task.inputs) == 1
        assert task.prompt is None
        assert task.scope is None
        assert task.input_annotations == ()
        assert dict(task.metadata) == {}
    submetered = [r for r in dataset.records if record_metadata(r).subset is Subset.SUBMETERED]
    assert all(record.annotations == () for record in submetered)


def test_each_task_has_its_method(tmp_path):
    loader = PlaidDatasetLoader.parse_from([_release(tmp_path)], counts=_FIXTURE_COUNTS, defects=_FIXTURE_DEFECTS)
    appliance = list(loader.appliance_tasks())
    events = list(loader.switching_events_tasks())
    assert [task.targets for task in appliance] == [("Fan",), ("Laptop",), ("Water kettle",)]
    assert [len(task.target_annotations) for task in events] == [5, 3]
    assert all(isinstance(task, ClassificationTask) for task in appliance)
    assert all(isinstance(task, TemporalLocalizationTask) for task in events)
    # tasks() yields from each method in turn.
    assert [type(task) for task in loader.tasks()] == [type(task) for task in (*appliance, *events)]


def test_pinned_counts_mismatch_fails(tmp_path):
    with pytest.raises(TimeFValidationError, match="the pinned article holds"):
        PlaidDatasetLoader.parse_from([_release(tmp_path)], counts=PINNED_COUNTS, defects=_FIXTURE_DEFECTS)


def test_unpinned_misstated_length_fails(tmp_path):
    with pytest.raises(TimeFFormatError, match="don't last their stated length"):
        PlaidDatasetLoader.parse_from([_release(tmp_path)], counts=_FIXTURE_COUNTS, defects=KNOWN_DEFECTS)


def test_truncated_file_fails(tmp_path):
    root = _release(tmp_path)
    _write_csv(root / "submetered_new" / "2.csv", rows=_ROWS - 200)
    with pytest.raises(TimeFFormatError, match=r"are \[2\], the pinned release's are \[\]"):
        PlaidDatasetLoader.parse_from([root], counts=_FIXTURE_COUNTS, defects=_FIXTURE_DEFECTS)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing", "CSV files missing for entries"),
        ("extra", "files with no entry"),
        ("unlisted", "expected the unlisted files"),
    ],
)
def test_files_and_entries_must_match(tmp_path, change, message):
    root = _release(tmp_path)
    if change == "missing":
        (root / "aggregated" / "1.csv").unlink()
    elif change == "extra":
        _write_csv(root / "submetered_new" / "4.csv")
    else:
        (root / "aggregated" / "0.csv").unlink()
    with pytest.raises(TimeFFormatError, match=message):
        PlaidDatasetLoader.parse_from([root], counts=_FIXTURE_COUNTS, defects=_FIXTURE_DEFECTS)


@pytest.mark.parametrize(
    ("subset", "edit", "message"),
    [
        (Subset.AGGREGATED, lambda e: e["1"]["appliances"][0].update(on="37043"), "expected sample indices"),
        (Subset.AGGREGATED, lambda e: e["1"]["appliances"][0].update(off="[300]"), "fall outside its 300 rows"),
        (Subset.SUBMETERED, lambda e: e["1"]["appliance"].update(type=""), "does not fit the SubmeteredEntry"),
        (Subset.SUBMETERED, lambda e: e["1"]["appliance"].update(load="X"), "does not fit the SubmeteredEntry"),
        (Subset.SUBMETERED, lambda e: e["1"]["appliance"].update(colour="red"), "does not fit the SubmeteredEntry"),
        (Subset.SUBMETERED, lambda e: e["1"]["header"].update(sampling_frequency="1000Hz"), "SubmeteredEntry"),
        (Subset.SUBMETERED, lambda e: e["1"]["instances"].update(length="2s"), "does not fit the SubmeteredEntry"),
        (Subset.AGGREGATED, lambda e: e["1"].pop("location"), "does not fit the AggregatedEntry"),
        (Subset.AGGREGATED, lambda e: e.update(x={}), "expected a file number"),
    ],
)
def test_malformed_metadata_fails_with_its_entry(tmp_path, subset, edit, message):
    root = _release(tmp_path)
    path = root / subset.metadata.name
    entries = json.loads(path.read_text(encoding="utf-8"))
    edit(entries)
    path.write_text(json.dumps(entries), encoding="utf-8")
    with pytest.raises(TimeFFormatError, match=message):
        PlaidDatasetLoader.parse_from([root], counts=_FIXTURE_COUNTS, defects=_FIXTURE_DEFECTS)


@pytest.mark.parametrize(
    ("row", "message"),
    [("nan,120.0", "is not finite"), (",120.0", "have no current value"), ("1,2,3", "Expected 2 columns, got 3")],
)
def test_bad_values_fail_when_read(tmp_path, row, message):
    root = _release(tmp_path)
    dataset = _FixtureConnector().convert([root])
    path = root / "submetered_new" / "1.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[5] = row
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    record = next(r for r in dataset.records if record_metadata(r).file == "submetered_new/1.csv")
    current = next(signal for signal in record.walk_signals() if signal.name == "current")
    with pytest.raises(TimeFFormatError, match=message):
        current.to_numpy()


def test_changed_row_count_fails_when_read(tmp_path):
    root = _release(tmp_path)
    dataset = _FixtureConnector().convert([root])
    _write_csv(root / "submetered_new" / "3.csv", rows=_ROWS + 1)
    record = next(r for r in dataset.records if record_metadata(r).file == "submetered_new/3.csv")
    with pytest.raises(TimeFFormatError, match="when it was parsed"):
        next(iter(record.walk_signals())).to_numpy()
