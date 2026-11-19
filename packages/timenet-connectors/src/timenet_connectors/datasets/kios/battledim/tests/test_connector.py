from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, open_local_version
from timenet.types import AnswerTask, ClassificationTask, ScalarPredictionTask, TemporalLocalizationTask
from timenet.types.text import read_value
from timenet_connectors.datasets.kios.battledim import BattLeDIMConnector
from timenet_connectors.datasets.kios.battledim.connector import BattLeDIMSource
from timenet_connectors.datasets.kios.battledim.source import Release


# A synthetic release shaped exactly like the Zenodo files: semicolon-separated CSVs with a decimal
# comma on a 5 minute grid, the configuration YAML, the repair report and an EPANET network. The
# values and the network are hand-made for the tests, so the repo ships no dataset bytes. Two full
# years are needed because the split groups windows into calendar blocks and asks for 10 folds.
STEPS_PER_YEAR = 365 * 288
_TIMES = (np.datetime64("2018-01-01T00:00") + np.arange(2 * STEPS_PER_YEAR) * np.timedelta64(5, "m")).astype(str)
TIMES = [time.replace("T", " ") + ":00" for time in _TIMES]

CONFIG = """\
Comments:
  - Synthetic configuration for tests
Network:
  filename: L-TOWN_Real.inp
times:
  StartTime: 2018-01-01 00:00
  EndTime: 2019-12-31 23:55
leakages:
- # linkID, startTime, endTime, leakDiameter (m), leakType, peakTime
- p3, 2018-03-05 15:45, 2018-03-23 10:25, 0.022916, abrupt, 2018-03-05 15:45
- p5, 2018-11-01 00:00, 2019-12-31 23:55, 0.010000, incipient, 2018-11-15 00:00
- p6, 2019-05-01 12:00, 2019-12-31 23:55, 0.015000, abrupt, 2019-05-01 12:00
pressure_sensors:
- n1
- n4
flow_sensors:
- PUMP_1
- p227
level_sensors:
- T1
amrs:
- n2
- n3
"""
REPORT = "# FIXED LEAKAGES\n# linkID, endTime\n- p3, 2018-03-23 10:25\n"
NETWORK = """\
[JUNCTIONS]
 n1 10 0 ;
[PIPES]
;ID Node1 Node2 Length Diameter Roughness MinorLoss Status
 p227 R1 n1 10 200 140 0 Open ;
 p2   n1 n2 10 150 140 0 Open ;
 p3   n2 n3 10 150 140 0 Open ;
 p4   n3 n4 10 150 140 0 Open ;
 p5   n4 n5 10 150 140 0 Open ;
 p6   n5 n6 10 150 140 0 Open ;
[PUMPS]
 PUMP_1 n2 T1 HEAD 1 ;FLOW SENSOR
[VALVES]
[END]
"""


def _step(text: str) -> int:
    return TIMES.index(text + ":00")


def _outflows() -> dict[str, np.ndarray]:
    outflow = {pipe: np.zeros(2 * STEPS_PER_YEAR) for pipe in ("p3", "p5", "p6")}
    outflow["p3"][_step("2018-03-05 15:45") : _step("2018-03-23 10:25") + 1] = 28.4
    start, peak = _step("2018-11-01 00:00"), _step("2018-11-15 00:00")
    outflow["p5"][start:peak] = np.linspace(0.0, 5.0, peak - start)
    outflow["p5"][peak:] = 5.0
    outflow["p6"][_step("2019-05-01 12:00") :] = 10.5
    return outflow


def _write_csv(path: Path, columns: dict[str, np.ndarray], rows: slice) -> None:
    lines = ["Timestamp;" + ";".join(columns)]
    formatted = [[f"{value:.2f}".replace(".", ",") for value in values[rows]] for values in columns.values()]
    for index, time in enumerate(TIMES[rows]):
        lines.append(";".join([time, *(column[index] for column in formatted)]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_release(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    clock = np.arange(2 * STEPS_PER_YEAR)
    daily = np.sin(2 * np.pi * clock / 288)
    scada = {
        "SCADA_Pressures": {"n1": 50.0 + daily, "n4": 40.0 + 0.5 * daily},
        "SCADA_Flows": {"p227": 80.0 + 10 * daily, "PUMP_1": np.full(clock.size, 44.04)},
        "SCADA_Levels": {"T1": 3.0 + 0.5 * daily},
        "SCADA_Demands": {"n2": 100.0 + 20 * daily, "n3": np.full(clock.size, 12.5)},
    }
    outflow = _outflows()
    years = {2018: slice(0, STEPS_PER_YEAR), 2019: slice(STEPS_PER_YEAR, 2 * STEPS_PER_YEAR)}
    for year, rows in years.items():
        for name, columns in scada.items():
            _write_csv(root / f"{year}_{name}.csv", columns, rows)
        active = {pipe: values for pipe, values in outflow.items() if values[rows].any()}
        _write_csv(root / f"{year}_Leakages.csv", active, rows)
    (root / "dataset_configuration.yaml").write_text(CONFIG, encoding="utf-8")
    (root / "2018_Fixed_Leakages_Report.txt").write_text(REPORT, encoding="utf-8")
    (root / "L-TOWN.inp").write_text(NETWORK, encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def release_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_release(tmp_path_factory.mktemp("battledim"))


@pytest.fixture(scope="module")
def dataset(release_dir: Path):
    return BattLeDIMConnector().convert([BattLeDIMSource(release_dir)])


def test_record_holds_every_sensor_on_one_grid(dataset) -> None:
    (record,) = dataset.records
    sources = {source.name: [signal.name for signal in source.signals] for source in record.sources}
    assert sources == {
        "Pressure sensors": ["n1", "n4"],
        "Flow sensors": ["PUMP_1", "p227"],
        "Tank level sensor": ["T1"],
        "Automated meter readings": ["n2", "n3"],
    }
    pressure = record.sources[0].signals[0]
    assert pressure.n_values == 2 * STEPS_PER_YEAR
    assert str(pressure.spec.unit_value) == "meter"
    assert pressure.time_axis.period_us == 300_000_000
    values = pressure.to_numpy()
    # The decimal comma parses, and the second year continues the first.
    assert values[0] == pytest.approx(50.0)
    assert values[STEPS_PER_YEAR + 72] == pytest.approx(51.0, abs=0.01)
    flow = record.sources[1].signals[0]
    assert str(flow.spec.unit_value) == "meter ** 3 / hour"
    assert flow.to_numpy()[5] == pytest.approx(44.04)


def test_leaks_become_annotations_over_their_intervals(dataset) -> None:
    (record,) = dataset.records
    pipes = [annotation for annotation in record.annotations if annotation.key == "leak_pipe"]
    assert [annotation.value for annotation in pipes] == ["p3", "p5", "p6"]
    p3 = pipes[0].span
    assert p3.start_us == _step("2018-03-05 15:45") * 300_000_000
    # The interval ends after the repair step, the last step with outflow.
    assert p3.end_us == (_step("2018-03-23 10:25") + 1) * 300_000_000
    repaired = {a.span.start_us: a.value for a in record.annotations if a.key == "leak_repaired"}
    assert sorted(repaired.values()) == [False, False, True]
    peak = [a for a in record.annotations if a.key == "leak_peak" and a.value == "p5"]
    assert peak[0].span.start_us == _step("2018-11-15 00:00") * 300_000_000


def test_windows_give_four_tasks_each(dataset) -> None:
    tasks = list(dataset.get_all())
    # 730 days hold 103 two-week windows that start every 7 days.
    assert Counter(type(task) for task in tasks) == {
        ClassificationTask: 103,
        TemporalLocalizationTask: 103,
        ScalarPredictionTask: 103,
        AnswerTask: 103,
    }
    first = tasks[0]
    assert first.scope.start_us == 0
    assert first.scope.end_us == 14 * 24 * 3600 * 1_000_000


def _tasks_of_week(dataset, target_week_start: str) -> list:
    return [task for task in dataset.get_all() if task.metadata["target_week_start"] == target_week_start]


def test_a_new_abrupt_leak_is_classified_located_and_explained(dataset) -> None:
    classify, locate, outflow, answer = _tasks_of_week(dataset, "2018-03-05 00:00")
    assert classify.targets == ("abrupt",)
    assert [point.start_us for point in locate.targets] == [_step("2018-03-05 15:45") * 300_000_000]
    assert outflow.target_name == "mean_leak_outflow"
    assert outflow.targets[0] == pytest.approx(28.4 * (7 * 288 - _step("2018-03-05 15:45") % 2016) / 2016, rel=1e-3)
    text = read_value(answer.targets[0])
    assert text.startswith("A sudden leak started on pipe p3 on 2018-03-05 15:45, with a hole of 22.9 mm")
    # p3 joins n2 and n3. n1 and n4 are both one link away, so the first listed sensor wins.
    assert "The nearest pressure sensor is n1, 1 link away." in text
    assert text.endswith("At the end of the window 1 leak is active, with a total outflow of 28.4 m3/h.")


def test_a_repair_and_a_quiet_week_are_explained(dataset) -> None:
    (*_, answer) = _tasks_of_week(dataset, "2018-03-19 00:00")
    assert "The leak on pipe p3 was repaired on 2018-03-23 10:25; its outflow before the repair was 28.4 m3/h." in (
        read_value(answer.targets[0])
    )
    classify, locate, _, answer = _tasks_of_week(dataset, "2018-07-02 00:00")
    assert classify.targets == ()
    assert locate.targets == ()
    assert read_value(answer.targets[0]).startswith("No leak started or was repaired in the last 7 days.")


def test_split_keeps_each_calendar_block_in_one_partition(dataset) -> None:
    tasks = list(dataset.get_all())
    partitions: dict[int, set] = {}
    for task in tasks:
        partitions.setdefault(task.metadata["calendar_block"], set()).add(task.split)
    assert all(len(splits) == 1 for splits in partitions.values())
    counts = Counter(task.split for task in tasks)
    assert sum(counts.values()) == len(tasks)
    assert len(counts) == 3


def _answers(dataset) -> list[str]:
    answers = []
    for task in dataset.get_all():
        if isinstance(task, AnswerTask):
            assert task.targets is not None
            answers.append(read_value(task.targets[0]))
    return answers


def test_dataset_survives_storage_round_trip(dataset, tmp_path: Path) -> None:
    registry = LocalRegistry(tmp_path / "registry")
    registry.put_dataset(dataset)
    with TimeFReader(open_local_version(registry.root / "kios/battledim/1.0.0")) as reader:
        restored = reader.read()
    (record,) = restored.records
    assert record.sources[3].signals[1].to_numpy()[0] == pytest.approx(12.5)
    # The reader returns the tasks in the order the connector added them.
    assert [task.split for task in restored.get_all()] == [task.split for task in dataset.get_all()]
    assert _answers(restored) == _answers(dataset)


def test_nearest_sensor_prefers_the_nearer_pipe_end(release_dir: Path) -> None:
    network = Release(release_dir).network
    assert network.nearest("p5", ["n1", "n4"]) == ("n4", 0)
    assert network.nearest("p6", ["n1", "n4"]) == ("n4", 1)


def _corrupt(release_dir: Path, tmp_path: Path, name: str, edit) -> Path:
    copy = tmp_path / "release"
    copy.mkdir()
    for path in release_dir.iterdir():
        (copy / path.name).write_bytes(path.read_bytes())
    target = copy / name
    target.write_text(edit(target.read_text(encoding="utf-8")), encoding="utf-8")
    return copy


@pytest.mark.parametrize(
    ("name", "edit", "message"),
    [
        (
            "2018_SCADA_Pressures.csv",
            lambda text: text.replace("2018-01-01 00:05:00;", "2018-01-01 00:06:00;"),
            "5 minutes",
        ),
        ("2019_SCADA_Levels.csv", lambda text: text.replace(";3,00\n", ";\n", 1), "missing values"),
        ("dataset_configuration.yaml", lambda text: text.replace("- p6,", "- p4,"), "different pipes"),
        ("2018_Fixed_Leakages_Report.txt", lambda text: text.replace("10:25", "10:30"), "disagree"),
        ("L-TOWN.inp", lambda text: text.replace(" p5   n4 n5", " p9   n4 n5"), "no pipe for the leaks"),
    ],
)
def test_inconsistent_release_raises(release_dir: Path, tmp_path: Path, name: str, edit, message: str) -> None:
    with pytest.raises(TimeNetConnectorError, match=message):
        Release(_corrupt(release_dir, tmp_path, name, edit))
