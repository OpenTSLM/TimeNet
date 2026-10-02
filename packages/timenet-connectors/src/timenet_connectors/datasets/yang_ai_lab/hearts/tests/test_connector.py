"""Check the CGMacros cases: calendar time, meal context on the timeline, held-out answers, and images."""

from datetime import datetime, timedelta
from io import BytesIO
import pickle

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from timenet.client import TimeNet
from timenet.dataset import IrregularAxis, RegularAxis
from timenet.errors import TimeFFormatError
from timenet.types import InputModality, Split, TimeInterval, TimePoint
from timenet.writer import TimeFWriter
from timenet_connectors.datasets.yang_ai_lab.hearts.connector import HeartsConnector
from timenet_connectors.datasets.yang_ai_lab.hearts.records import frame_times_us


START = datetime(2025, 1, 1, 0, 0)  # naive, as the release writes its timestamps
MINUTE_US = 60_000_000
DAY_US = 24 * 60 * MINUTE_US


def _cgm_frame(start, values, *, skip=(), index_start=0, **columns):
    """A CGMacros frame: text timestamps one minute apart, with gaps at the skipped positions."""
    stamps = [str(start + timedelta(minutes=i)) for i in range(len(values) + len(skip)) if i not in skip]
    frame = pd.DataFrame({"Timestamp": pd.Series(stamps, dtype=object), "Libre GL": list(values), **columns})
    frame.index = range(index_start, index_start + len(frame))
    return frame


def _jpeg(color):
    buffer = BytesIO()
    Image.new("RGB", (3, 2), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _meal(stamp, kind, calories, carbs):
    return {
        "Timestamp": str(stamp),
        "Meal Type": kind,
        "Calories": calories,
        "Carbs": carbs,
        "Protein": 8.0,
        "Fat": 12.0,
        "Fiber": 1.0,
    }


MEAL_TIME = START + timedelta(hours=1)
FORECAST = [100.0 + index for index in range(30)]
IMPUTED = [90.0 + index for index in range(30)]
CASES = {
    "a1c_classification": {
        "subject_id": "CGMacros-001",
        "window_df": _cgm_frame(START, [100.0, 101.0, 102.0]),
        "GT": "normal",
    },
    "cgm_stat_calculation": {
        "subject_id": "CGMacros-001",
        "cgm": _cgm_frame(START, [60.0, 100.0, 100.0, 100.0])[["Libre GL", "Timestamp"]],
        "GT": {"below": 25.0, "above": 0.0},
    },
    "fasting_glu_prediction": {"subject_id": "CGMacros-001", "window_df": _cgm_frame(START, [90.0, 91.0]), "GT": 98.0},
    "iauc_calculation": {
        "subject_id": "CGMacros-001",
        "meal_time": pd.Timestamp(START),
        "cgm_df": pd.DataFrame(
            {"Time (min)": [0.0, 1.0, 2.0], "CGM (mg/dL)": [80.0, 90.0, 85.0]}, index=[4468, 4469, 4470]
        ),
        "GT": 12.5,
    },
    "meal_react_comparison": {
        "A": _cgm_frame(START, [74.0, 90.0, 120.0]),
        "B": _cgm_frame(START + timedelta(days=3), [79.0, 85.0, 90.0]),
        "A_subject": "CGMacros-023",
        "B_subject": "CGMacros-004",
        "A_meal": {"subject_id": "CGMacros-023", "diabetes_status": "Prediabetes/Diabetes"},
        "B_meal": {"subject_id": "CGMacros-004", "diabetes_status": "Normal"},
        "GT": {"A": "Prediabetes/Diabetes", "B": "Normal"},
    },
    "meal_time_localization": {
        "subject_id": "CGMacros-032",
        "window_df": _cgm_frame(START, [87.0, 88.0, 90.0, 110.0, 117.0], skip=(3,), timestamp_min=[0, 1, 2, 4, 5]),
        "window_start": 28_933_920,
        "window_end": 28_934_040,
        "GT": np.int64(4),
    },
    "meal_forecasting": {
        "subject_id": "CGMacros-048",
        "reference_cgm_df": _cgm_frame(START - timedelta(days=2), [58.0, 60.0, 61.0, 65.0], skip=(2,)),
        "reference_meal_info_df": pd.DataFrame(
            [
                _meal(START - timedelta(days=2, minutes=-1), "lunch", 1180.0, 81.0),
                _meal(START - timedelta(days=2, minutes=-3), "snack", 20.0, 2.0),
            ]
        ),
        "window_df": _cgm_frame(START, [83.0 + index / 10 for index in range(60)], index_start=7697),
        "meal_info": _meal(MEAL_TIME, "snack", 179.0, 10.0),
        "meal_time": pd.Timestamp(MEAL_TIME),
        "GT": FORECAST,
    },
    "meal_forecasting_no_ref_meal_info": {
        "subject_id": "CGMacros-003",
        "window_df": _cgm_frame(START, [68.0 + index / 10 for index in range(60)]),
        "meal_info": _meal(MEAL_TIME, "snack", 179.0, 10.0),
        "meal_time": pd.Timestamp(MEAL_TIME),
        "GT": FORECAST,
    },
    "meal_img_classification": {
        "subject_id": "CGMacros-009",
        "window_df": _cgm_frame(START, [90.0, 92.0, 95.0, 120.0, 140.0]),
        "meal_info": {"Meal Type": "dinner", "Image path": "/scratch/photos/b.jpg"},
        "meal_time": pd.Timestamp(START + timedelta(minutes=2)),
        "image_mapping": {
            f"{name}.jpg": _jpeg(color)
            for name, color in zip("abcd", [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)], strict=True)
        },
        "GT": "b.jpg",
    },
    "non_meal_imputation_hr": {
        "subject_id": "CGMacros-014",
        "window_df": _cgm_frame(
            START,
            [86.0, 85.0, 84.0, 83.0, *([0.0] * 30), 70.0, 71.0, 72.0, 73.0, 74.0, 75.0],
            skip=(2,),
            index_start=4006,
            HR=[70.0] * 40,
            timestamp_min=[0, 1, *range(3, 41)],
        ),
        "mask_indices": pd.Index(range(4010, 4040)),
        "mask_start": str(START + timedelta(minutes=5)),
        "mask_end": str(START + timedelta(minutes=34)),
        "GT": IMPUTED,
    },
}


def _write_tree(root, cases):
    for task, payload in cases.items():
        directory = root / "cgmacros" / task
        directory.mkdir(parents=True)
        (directory / "0.pkl").write_bytes(pickle.dumps(payload))


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("hearts")
    _write_tree(root, CASES)
    return HeartsConnector().convert([root])


def _task(dataset, task):
    return next(item for item in dataset.tasks if item.id == f"hearts-cgmacros-{task}-00-task")


def _by_key(annotations):
    return {annotation.key: annotation for annotation in annotations}


def test_every_case_is_a_test_item_with_its_provenance(dataset):
    assert len(dataset.tasks) == len(CASES)
    assert {task.split for task in dataset.tasks} == {Split.TEST}
    assert dataset.annotations == ()
    for task in dataset.tasks:
        assert task.metadata == {
            "corpus": "cgmacros",
            "task": task.id.removeprefix("hearts-cgmacros-").removesuffix("-00-task"),
            "testcase_idx": 0,
        }
        assert task.annotations == ()


def test_cgm_records_keep_local_calendar_time_without_inventing_utc(dataset):
    (record,) = _task(dataset, "a1c_classification").inputs
    assert record.start_time.timestamp is None
    assert record.subject_ids == ()
    assert record.metadata == {"corpus": "cgmacros"}
    annotations = _by_key(record.annotations)
    assert annotations["subject_id"].value == "cgmacros:CGMacros-001"
    start = annotations["recording_start_local"]
    assert (type(start.value), start.value) == (str, "2025-01-01T00:00:00.000000")
    assert [source.name for source in record.sources] == ["window_df"]
    (signal,) = record.signals
    assert signal.name == "Libre GL"
    assert isinstance(signal.time_axis, RegularAxis)
    assert signal.to_arrow().to_pylist() == [100.0, 101.0, 102.0]


def test_localization_keeps_the_gap_and_answers_in_minutes(dataset):
    task = _task(dataset, "meal_time_localization")
    (signal,) = task.inputs[0].signals  # timestamp_min becomes no signal
    assert isinstance(signal.time_axis, IrregularAxis)
    assert signal.time_offsets_us().tolist() == [0, 1 * MINUTE_US, 2 * MINUTE_US, 4 * MINUTE_US, 5 * MINUTE_US]
    assert task.targets == (TimePoint.micros(4 * MINUTE_US),)


def test_forecast_places_the_meal_and_its_context_on_the_shared_clock(dataset):
    task = _task(dataset, "meal_forecasting")
    (record,) = task.inputs
    assert record.start_time.timestamp is None
    assert _by_key(record.annotations)["recording_start_local"].value == "2024-12-30T00:00:00.000000"
    assert [source.name for source in record.sources] == ["reference_cgm_df", "window_df"]
    reference, window = record.signals
    assert isinstance(reference.time_axis, IrregularAxis) and reference.n_values == 4
    assert window.span_us == (2 * DAY_US, 2 * DAY_US + 60 * MINUTE_US)
    meal_us = 2 * DAY_US + 60 * MINUTE_US
    assert record.time_span == TimeInterval.micros(0, meal_us + 30 * MINUTE_US)
    (answer,) = task.targets
    assert answer.start_time is record.start_time
    assert answer.subject_ids == ()
    assert answer.metadata == {"corpus": "cgmacros"}
    assert [annotation.key for annotation in answer.annotations] == ["subject_id", "recording_start_local"]
    assert _by_key(answer.annotations)["subject_id"].value == "cgmacros:CGMacros-048"
    assert answer.signals[0].span_us == (meal_us, meal_us + 30 * MINUTE_US)
    assert answer.signals[0].to_arrow().to_pylist() == FORECAST
    context = task.input_annotations
    assert [annotation.key for annotation in context] == [
        *(["meal_type", "calories", "carbs", "protein", "fat", "fiber"] * 2),
        "meal_time",
    ]
    assert context[-1].span == TimePoint.micros(meal_us)
    assert (context[1].value, context[1].unit, context[1].span) == (
        1180.0,
        "kilocalorie",
        TimePoint.micros(1 * MINUTE_US),
    )
    assert "A meal event occurred at 2025-01-01 01:00:00.\n" in task.prompt
    assert "The information for this meal" not in task.prompt
    assert "of source 'reference_cgm_df'" in task.prompt


def test_forecast_with_meal_info_tells_the_meal_in_the_prompt_and_on_the_timeline(dataset):
    task = _task(dataset, "meal_forecasting_no_ref_meal_info")
    assert (
        "A meal event occurred at 2025-01-01 01:00:00. The information for this meal is: Timestamp: 2025-01-01 "
        "01:00:00, Meal Type: snack, Calories: 179.0, Carbs: 10.0, Protein: 8.0, Fat: 12.0, Fiber: 1.0.\n"
    ) in task.prompt
    meal_us = 60 * MINUTE_US
    context = _by_key(task.input_annotations)
    assert context["meal_time"].span == TimePoint.micros(meal_us)
    assert (context["calories"].value, context["calories"].span) == (179.0, TimePoint.micros(meal_us))
    assert context["meal_type"].value == "snack"


def test_imputation_marks_the_zeroed_stretch_and_holds_its_readings_apart(dataset):
    task = _task(dataset, "non_meal_imputation_hr")
    (record,) = task.inputs
    assert [(signal.name, signal.spec.spec_type) for signal in record.signals] == [
        ("HR", "heart_rate"),
        ("Libre GL", "cgm"),
    ]
    assert all(isinstance(signal.time_axis, IrregularAxis) for signal in record.signals)
    (mask,) = task.input_annotations
    assert mask.key == "cgm_mask"
    assert mask.span == TimeInterval.micros(5 * MINUTE_US, 35 * MINUTE_US)
    (answer,) = task.targets
    assert answer.start_time is record.start_time
    assert answer.signals[0].span_us == (5 * MINUTE_US, 35 * MINUTE_US)
    assert isinstance(answer.signals[0].time_axis, RegularAxis)
    assert answer.signals[0].to_arrow().to_pylist() == IMPUTED
    assert "start from 2025-01-01 00:05:00 and end at 2025-01-01 00:34:00" in task.prompt
    assert "using the heart rate and remaining CGM data" in task.prompt


def test_compared_windows_are_records_of_their_own_subject(dataset):
    task = _task(dataset, "meal_react_comparison")
    assert [record.id for record in task.inputs] == [
        "hearts-cgmacros-meal_react_comparison-00-A",
        "hearts-cgmacros-meal_react_comparison-00-B",
    ]
    assert [_by_key(record.annotations)["subject_id"].value for record in task.inputs] == [
        "cgmacros:CGMacros-023",
        "cgmacros:CGMacros-004",
    ]
    assert [source.name for record in task.inputs for source in record.sources] == ["A", "B"]
    assert task.targets == ("B",)
    assert task.target_schema == "hearts-options-meal_react_comparison"
    vocabulary = next(
        annotation for annotation in dataset.registered_annotations if annotation.id == task.target_schema
    )
    assert vocabulary.value == ["A", "B"]


def test_meal_photographs_are_image_signals_and_the_meal_is_marked(dataset):
    task = _task(dataset, "meal_img_classification")
    assert task.resolved_input_modalities == {InputModality.TEXT, InputModality.TIME_SERIES, InputModality.IMAGE}
    (record,) = task.inputs
    assert [source.name for source in record.sources] == ["window_df", "image_mapping"]
    images = {signal.name: signal for signal in record.signals if signal.name.endswith(".jpg")}
    assert list(images) == ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    assert images["b.jpg"].spec.value_shape == (2, 3, 3)
    assert task.targets == ("b.jpg",)
    (marker,) = task.input_annotations
    assert (marker.key, marker.span) == ("meal_time", TimePoint.micros(2 * MINUTE_US))


def test_scalar_answers_are_numeric_targets(dataset):
    statistics = _task(dataset, "cgm_stat_calculation")
    assert (statistics.targets, str(statistics.unit), statistics.target_name) == (
        (25.0, 0.0),
        "percent",
        "time_below_and_above_range",
    )
    area = _task(dataset, "iauc_calculation")
    assert (area.targets, str(area.unit)) == ((12.5,), "mg*min/dL")
    assert area.inputs[0].start_time.timestamp is None
    assert _by_key(area.inputs[0].annotations)["subject_id"].value == "cgmacros:CGMacros-001"
    assert area.inputs[0].signals[0].span_us == (0, 3 * MINUTE_US)
    assert _task(dataset, "fasting_glu_prediction").targets == (98.0,)


def test_values_round_trip_through_the_writer(dataset, tmp_path):
    with TimeFWriter(tmp_path / "registry", dataset, values_backend="zarr") as writer:
        writer.write()
    read = TimeNet(registry=tmp_path / "registry").load("yang-ai-lab/hearts", auto_build=False)
    forecast = _task(read, "meal_forecasting")
    assert forecast.metadata == {"corpus": "cgmacros", "task": "meal_forecasting", "testcase_idx": 0}
    assert forecast.inputs[0].metadata == {"corpus": "cgmacros"}
    assert forecast.targets[0].metadata == {"corpus": "cgmacros"}
    assert _by_key(forecast.inputs[0].annotations)["subject_id"].value == "cgmacros:CGMacros-048"
    assert _by_key(forecast.targets[0].annotations)["subject_id"].value == "cgmacros:CGMacros-048"
    assert forecast.targets[0].signals[0].to_arrow().to_pylist() == FORECAST
    reference, window = forecast.inputs[0].signals
    assert reference.time_offsets_us().tolist() == [0, MINUTE_US, 3 * MINUTE_US, 4 * MINUTE_US]
    assert window.to_arrow().to_pylist()[:2] == [83.0, 83.1]
    assert _task(read, "meal_img_classification").inputs[0].signals[1].to_arrow().to_numpy_ndarray().shape == (
        1,
        2,
        3,
        3,
    )


def test_rejects_a_mask_that_disagrees_with_its_bounds(tmp_path):
    payload = dict(CASES["non_meal_imputation_hr"], mask_end=str(START + timedelta(minutes=33)))
    _write_tree(tmp_path, {"non_meal_imputation_hr": payload})
    with pytest.raises(TimeFFormatError, match="mask_indices do not name"):
        HeartsConnector().convert([tmp_path])


def test_rejects_a_frame_column_it_cannot_place(tmp_path):
    payload = dict(CASES["a1c_classification"], window_df=_cgm_frame(START, [100.0, 101.0], Steps=[3.0, 4.0]))
    _write_tree(tmp_path, {"a1c_classification": payload})
    with pytest.raises(TimeFFormatError, match="only known value columns"):
        HeartsConnector().convert([tmp_path])


@pytest.mark.parametrize(
    "timestamps",
    [pd.Series([], dtype="datetime64[ns]"), pd.Series([pd.NaT], dtype="datetime64[ns]")],
)
def test_rejects_empty_or_missing_frame_times(timestamps):
    frame = pd.DataFrame({"Timestamp": timestamps})
    with pytest.raises(TimeFFormatError, match=r"has no values|has missing values"):
        frame_times_us(frame, "Timestamp")


def test_rejects_frames_that_mix_wall_clock_and_relative_time(tmp_path):
    relative = pd.DataFrame({"Timestamp": [0.0, 1.0], "Libre GL": [83.0, 84.0]})
    payload = dict(CASES["meal_forecasting"], window_df=relative)
    _write_tree(tmp_path, {"meal_forecasting": payload})
    with pytest.raises(TimeFFormatError, match="mixes wall-clock and relative time columns"):
        HeartsConnector().convert([tmp_path])
