import pytest

from timenet.errors import TimeNetConnectorError
from timenet.reader import TimeFReader
from timenet.registry import open_local_version
from timenet.testing import assert_datasets_equal
from timenet.types import ClassificationTask, TemporalLocalizationTask, TimeInterval
from timenet.types.text import read_value
from timenet_connectors.datasets.fraunhofer_iee.care_to_compare.connector import (
    EVENT_LABEL,
    FAULT_COMPONENT,
    CareToCompareConnector,
    CareWindFarm,
)
from timenet_connectors.datasets.fraunhofer_iee.care_to_compare.tests import fixtures


def _convert(root):
    fixtures.write_wind_farm(root)
    return CareToCompareConnector().convert(CareWindFarm.discover(root))


def _records(dataset):
    # The detection task of each event, in event order, names its Record.
    return [task.inputs[0] for task in dataset.tasks_of(ClassificationTask) if task.label_set == EVENT_LABEL]


def _annotation(owner, key):
    (annotation,) = [annotation for annotation in owner.annotations if annotation.key == key]
    return annotation


def test_the_wind_farm_directories_are_found_under_the_extraction_root(tmp_path):
    fixtures.write_wind_farm(tmp_path / "CARE_To_Compare", "B")
    fixtures.write_wind_farm(tmp_path / "CARE_To_Compare", "A")
    farms = CareWindFarm.discover(tmp_path)
    assert [farm.name for farm in farms] == ["A", "B"]
    with pytest.raises(TimeNetConnectorError, match="no 'Wind Farm"):
        CareWindFarm.discover(tmp_path / "CARE_To_Compare" / "Wind Farm A" / "datasets")


def test_each_event_becomes_a_record_with_one_signal_per_column(tmp_path):
    records = _records(_convert(tmp_path))
    assert len(records) == 2
    record = records[0]
    signals = {signal.name: signal for signal in record.signals}
    assert sorted(signals) == sorted([*fixtures.FEATURE_COLUMNS, "status_type_id"])
    n_steps = fixtures.N_TRAIN + fixtures.N_PREDICTION
    wind_speed = signals["wind_speed_3_avg"].to_arrow()
    assert len(wind_speed) == n_steps
    # The skipped step and the empty cell are nulls; the other values keep the release's digits.
    assert not wind_speed.is_valid()[fixtures.SKIPPED_STEP].as_py()
    assert not wind_speed.is_valid()[fixtures.EMPTY_CELL[0]].as_py()
    assert wind_speed[0].as_py() == pytest.approx(fixtures.value(0, 1), abs=1e-5)
    # The counter without a statistic suffix reads as its sensor's one statistic.
    assert [read_value(a.value) for a in signals["sensor_44"].annotations if a.key == "statistic"] == ["average"]
    status = signals["status_type_id"].to_arrow()
    assert status[fixtures.ANOMALY_END_STEP].as_py() == 4
    assert not status.is_valid()[fixtures.SKIPPED_STEP].as_py()


def test_the_event_table_becomes_annotations(tmp_path):
    record = _records(_convert(tmp_path))[0]
    assert read_value(_annotation(record, "wind_farm").value) == "A"
    assert read_value(_annotation(record, "event_id").value) == fixtures.ANOMALY_EVENT
    assert read_value(_annotation(record, "asset_id").value) == 21
    assert read_value(_annotation(record, "event_label").value) == "anomaly"
    assert read_value(_annotation(record, "fault_component").value) == "hydraulic"
    period_s = 600
    assert _annotation(record, "event_window").span == TimeInterval.seconds(
        fixtures.ANOMALY_START_STEP * period_s, (fixtures.ANOMALY_END_STEP + 1) * period_s
    )


def test_each_event_gets_its_tasks(tmp_path):
    dataset = _convert(tmp_path)
    classifications = dataset.tasks_of(ClassificationTask)
    localizations = dataset.tasks_of(TemporalLocalizationTask)
    # Detection for both events, and a component label for the anomaly.
    assert [read_value(task.targets[0]) for task in classifications] == ["anomaly", "hydraulic", "normal"]
    assert [task.label_set for task in classifications] == [EVENT_LABEL, FAULT_COMPONENT, EVENT_LABEL]
    (localization,) = localizations
    assert [annotation.key for annotation in localization.target_annotations] == ["event_window"]


def test_an_event_file_without_an_event_info_row_raises(tmp_path):
    farm = fixtures.write_wind_farm(tmp_path)
    fixtures.write_event_file(farm / "datasets" / "99.csv", asset=3)
    with pytest.raises(TimeNetConnectorError, match=r"event files \[99\] do not match"):
        CareToCompareConnector().convert(CareWindFarm.discover(tmp_path))


def test_the_dataset_round_trips_through_storage(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    dataset = _convert(source)
    version_dir = dataset.write(path=tmp_path / "registry")
    with TimeFReader(open_local_version(version_dir)) as reader:
        reader.verify()
        assert_datasets_equal(dataset, reader.read())
