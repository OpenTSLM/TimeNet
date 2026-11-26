import numpy as np
import pytest

from timenet.errors import TimeNetConnectorError
from timenet_connectors.datasets.fraunhofer_iee.care_to_compare.source import (
    PERIOD_US,
    STATUS_MISSING,
    parse_timestamp_us,
    read_event_info,
    read_event_table,
    read_sensor_descriptions,
    split_feature_column,
)
from timenet_connectors.datasets.fraunhofer_iee.care_to_compare.tests import fixtures


def test_the_sensor_descriptions_keep_the_release_units(tmp_path):
    sensors = read_sensor_descriptions(fixtures.write_wind_farm(tmp_path) / "feature_description.csv")
    assert list(sensors) == ["sensor_0", "wind_speed_3", "sensor_5", "sensor_31", "sensor_44"]
    assert sensors["sensor_0"].unit_expression == "degree_Celsius"
    assert sensors["wind_speed_3"].statistics == ("maximum", "minimum", "average", "std_dev")
    assert sensors["sensor_5"].is_angle
    assert sensors["sensor_5"].unit_expression == "degree"
    # Reactive power has no safe pint form, so the connector keeps the source string.
    assert sensors["sensor_31"].unit == "kVAr"
    assert sensors["sensor_31"].unit_expression is None


def test_a_column_splits_into_its_sensor_and_statistic():
    assert split_feature_column("wind_speed_3_avg") == ("wind_speed_3", "average")
    assert split_feature_column("sensor_52_std") == ("sensor_52", "std_dev")
    with pytest.raises(TimeNetConnectorError, match="does not end in"):
        split_feature_column("sensor_52")


def test_the_event_info_reads_both_asset_column_names(tmp_path):
    events = read_event_info(fixtures.write_wind_farm(tmp_path) / "event_info.csv")
    anomaly = events[fixtures.ANOMALY_EVENT]
    assert anomaly.asset_id == 21
    assert anomaly.label == "anomaly"
    assert anomaly.description == "Hydraulic group"
    assert anomaly.fault_component("A") == "hydraulic"
    assert anomaly.fault_component("C") is None
    normal = events[fixtures.NORMAL_EVENT]
    assert normal.description is None
    assert normal.fault_component("A") is None

    renamed = tmp_path / "renamed.csv"
    renamed.write_text(
        "asset_id;event_id;event_label;event_start;event_start_id;event_end;event_end_id;event_description\r\n"
        '53;35;anomaly;2023-09-24 00:00:00;1;2023-09-30 09:00:00;2;"Several stops; failure ""Level 1"""\r\n',
        encoding="utf-8",
    )
    (event,) = read_event_info(renamed).values()
    assert event.asset_id == 53
    # A quoted description keeps its semicolon and its doubled quotes.
    assert event.description == 'Several stops; failure "Level 1"'


def test_the_event_table_lies_on_the_grid_with_nulls_for_gaps(tmp_path):
    farm = fixtures.write_wind_farm(tmp_path)
    table = read_event_table(farm / "datasets" / f"{fixtures.ANOMALY_EVENT}.csv", block_rows=4)
    n_steps = fixtures.N_TRAIN + fixtures.N_PREDICTION
    assert table.asset_id == 21
    assert table.feature_columns == fixtures.FEATURE_COLUMNS
    assert table.values.shape == (len(fixtures.FEATURE_COLUMNS), n_steps)
    assert table.values.dtype == np.float32
    assert table.start_us == parse_timestamp_us(fixtures.timestamp(0))
    # The skipped step is a null in every signal and in the status.
    assert np.isnan(table.values[:, fixtures.SKIPPED_STEP]).all()
    assert table.status[fixtures.SKIPPED_STEP] == STATUS_MISSING
    # The empty cell is a null in its own signal only. Row 3 is grid step 3, before the gap.
    row, feature = fixtures.EMPTY_CELL
    assert np.isnan(table.values[feature, row])
    assert table.values[feature + 1, row] == np.float32(fixtures.value(row, feature + 1))
    # Every other cell keeps its value, and rows after the gap land on their own step.
    assert table.values[2, 15] == np.float32(fixtures.value(15, 2))
    assert table.status[fixtures.ANOMALY_END_STEP] == 4
    assert table.train_steps == (0, fixtures.N_TRAIN - 1)
    assert table.prediction_steps == (fixtures.N_TRAIN, n_steps - 1)
    assert table.step_of(table.start_us + 5 * PERIOD_US) == 5
    with pytest.raises(TimeNetConnectorError, match="off the grid"):
        table.step_of(table.start_us + PERIOD_US // 2)


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        ([0, 1, 1, 2, 25], "not rising"),
        ([0, 25, 3], "not rising"),
    ],
    ids=["a repeated timestamp", "time goes back"],
)
def test_an_event_table_off_the_grid_raises(tmp_path, steps, message):
    path = tmp_path / "1.csv"
    fixtures.write_event_file(path, asset=1, steps=steps)
    with pytest.raises(TimeNetConnectorError, match=message):
        read_event_table(path)


def test_a_wrong_row_id_raises_with_its_row(tmp_path):
    path = tmp_path / "1.csv"
    fixtures.write_event_file(path, asset=1)
    lines = path.read_text(encoding="utf-8").splitlines()
    cells = lines[3].split(";")
    cells[2] = "9"
    lines[3] = ";".join(cells)
    path.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(TimeNetConnectorError, match=r"1\.csv row 2: id '9'"):
        read_event_table(path)
