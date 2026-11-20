"""Write a small synthetic wind farm directory in the shape of the CARE to Compare release."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path


FEATURE_DESCRIPTION = (
    "sensor_name;statistics_type;description;unit;is_angle;is_counter\r\n"
    "sensor_0;average;Ambient temperature;�C;False;False\r\n"
    "wind_speed_3;maximum,minimum,average,std_dev;Windspeed;m/s;False;False\r\n"
    "sensor_5;average;Pitch angle;�;True;False\r\n"
    "sensor_31;average;Grid reactive power;kVAr;False;False\r\n"
    "sensor_44;average;Active power - generator disconnected;Wh;False;False\r\n"
)
FEATURE_COLUMNS = (
    "sensor_0_avg",
    "wind_speed_3_avg",
    "wind_speed_3_max",
    "wind_speed_3_min",
    "wind_speed_3_std",
    "sensor_5_avg",
    "sensor_31_avg",
    "sensor_44",
)
"""The release's column layout, including a counter that wind farm A names without a statistic suffix."""
START = datetime(2022, 8, 12, 9, 50)
STEP = timedelta(minutes=10)
N_TRAIN = 20
N_PREDICTION = 10
SKIPPED_STEP = 7
"""A grid step that the event files leave out, like the gaps of the release."""
EMPTY_CELL = (3, 1)
"""A (row, feature) cell that the event files leave empty."""
ANOMALY_EVENT = 22
NORMAL_EVENT = 24
ANOMALY_START_STEP = N_TRAIN + 2
ANOMALY_END_STEP = N_TRAIN + 6


def timestamp(step: int) -> str:
    return (START + step * STEP).strftime("%Y-%m-%d %H:%M:%S")


def value(step: int, feature: int) -> float:
    return round(step + feature / 10, 1)


def write_event_file(path: Path, *, asset: int, steps: list[int] | None = None) -> None:
    """Write one event file. The rows follow the grid except at :data:`SKIPPED_STEP`."""
    if steps is None:
        steps = [step for step in range(N_TRAIN + N_PREDICTION) if step != SKIPPED_STEP]
    lines = [";".join(("time_stamp", "asset_id", "id", "train_test", "status_type_id", *FEATURE_COLUMNS))]
    for row, step in enumerate(steps):
        cells = [
            "" if (row, feature) == EMPTY_CELL else str(value(step, feature)) for feature in range(len(FEATURE_COLUMNS))
        ]
        phase = "train" if step < N_TRAIN else "prediction"
        status = "4" if step == ANOMALY_END_STEP else "0"
        lines.append(";".join((timestamp(step), str(asset), str(row), phase, status, *cells)))
    path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")


def write_wind_farm(root: Path, name: str = "A") -> Path:
    """Write a wind farm with one anomaly event and one normal event.

    Returns:
        The wind farm directory.
    """
    directory = root / f"Wind Farm {name}"
    (directory / "datasets").mkdir(parents=True)
    (directory / "feature_description.csv").write_text(FEATURE_DESCRIPTION, encoding="utf-8")
    (directory / "event_info.csv").write_text(
        "asset;event_id;event_label;event_start;event_start_id;event_end;event_end_id;event_description\r\n"
        f"21;{ANOMALY_EVENT};anomaly;{timestamp(ANOMALY_START_STEP)};0;{timestamp(ANOMALY_END_STEP)};0;Hydraulic group\r\n"
        f"0;{NORMAL_EVENT};normal;{timestamp(N_TRAIN)};0;{timestamp(N_TRAIN + 9)};0;\r\n",
        encoding="utf-8",
    )
    write_event_file(directory / "datasets" / f"{ANOMALY_EVENT}.csv", asset=21)
    write_event_file(directory / "datasets" / f"{NORMAL_EVENT}.csv", asset=0)
    return directory
