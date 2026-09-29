"""Check the HEARTS task families that the older proposal left out."""

from datetime import datetime, timedelta
from io import BytesIO
import pickle

import pandas as pd
from PIL import Image
import pytest

from timenet.dataset import Record
from timenet.errors import TimeFFormatError
from timenet.types import ForecastingTask, InputModality, TSEditingTask
from timenet_connectors.datasets.yang_ai_lab.hearts.connector import HeartsConnector, HeartsSource


def _case(root, corpus, task, payload):
    directory = root / corpus / task
    directory.mkdir(parents=True)
    with (directory / "0.pkl").open("wb") as handle:
        pickle.dump(payload, handle)


def _window(with_hr=False):
    frame = pd.DataFrame(
        {
            "Timestamp": pd.Series(
                [str(value) for value in pd.date_range("2025-01-01", periods=120, freq="min")], dtype=object
            ),
            "Libre GL": [float(80 + index % 20) for index in range(120)],
            "timestamp_min": list(range(120)),
        }
    )
    if with_hr:
        frame["HR"] = [70.0] * 120
    return frame


def _jpeg():
    buffer = BytesIO()
    Image.new("RGB", (3, 2), (255, 0, 0)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_held_out_targets_images_and_recordless_symptoms(tmp_path):
    impute_window = _window(with_hr=True).drop(columns=["timestamp_min"])
    impute_window.loc[40:, "Timestamp"] = [
        str(datetime.fromisoformat(value) + timedelta(minutes=1)) for value in impute_window.loc[40:, "Timestamp"]
    ]
    impute_window.index = range(4006, 4126)
    _case(
        tmp_path,
        "cgmacros",
        "meal_forecasting_no_ref_meal_info",
        {
            "window_df": _window(),
            "meal_info": {"carbs": 20.0},
            "meal_time": "2025-01-01 01:00:00",
            "GT": [100.0 + index for index in range(30)],
        },
    )
    _case(
        tmp_path,
        "cgmacros",
        "non_meal_imputation_hr",
        {
            "window_df": impute_window,
            "mask_start": "2025-01-01 00:30:00",
            "mask_end": "2025-01-01 01:00:00",
            "mask_indices": list(range(4036, 4066)),
            "GT": [90.0] * 30,
        },
    )
    _case(
        tmp_path,
        "cgmacros",
        "meal_img_classification",
        {
            "window_df": _window(),
            "meal_time": "2025-01-01 01:00:00",
            "image_mapping": {f"{name}.jpg": _jpeg() for name in "abcd"},
            "GT": "b.jpg",
        },
    )
    _case(
        tmp_path,
        "coswara",
        "cough_covid_status_classification_symptoms_only",
        {"symptoms": {"fever": True, "cough": False}, "subject_id": "person-1", "GT": "healthy"},
    )

    dataset = HeartsConnector().convert([HeartsSource(tmp_path)])
    tasks = {task.id: task for task in dataset.iter_tasks()}
    forecast = tasks["hearts-cgmacros-meal_forecasting_no_ref_meal_info-00-qa"]
    impute = tasks["hearts-cgmacros-non_meal_imputation_hr-00-qa"]
    meal = tasks["hearts-cgmacros-meal_img_classification-00-qa"]
    symptoms = tasks["hearts-coswara-cough_covid_status_classification_symptoms_only-00-qa"]

    assert isinstance(forecast, ForecastingTask)
    assert isinstance(impute, TSEditingTask)
    assert forecast.targets is not None and isinstance(forecast.targets[0], Record)
    assert impute.targets is not None and isinstance(impute.targets[0], Record)
    assert forecast.targets[0].id not in {record.id for record in forecast.inputs}
    assert forecast.targets[0].annotations[0].value == "2025-01-01 01:00:00"
    assert impute.targets[0].signals[0].to_arrow().to_pylist() == [90.0] * 30
    assert forecast.targets[0].start_time is forecast.inputs[0].start_time
    assert impute.targets[0].start_time is impute.inputs[0].start_time
    forecast_span = forecast.targets[0].signals[0].span_us
    impute_span = impute.targets[0].signals[0].span_us
    assert forecast_span is not None and forecast_span[0] == 60 * 60_000_000
    assert impute_span is not None and impute_span[0] == 30 * 60_000_000
    assert impute.targets[0].signals[0].time_offsets_us()[10] == 41 * 60_000_000
    assert InputModality.IMAGE in meal.resolved_input_modalities
    assert {signal.name for signal in meal.inputs[0].signals} >= {"a.jpg", "b.jpg", "c.jpg", "d.jpg"}
    assert symptoms.inputs == ()
    assert symptoms.input_modalities == frozenset({InputModality.TEXT, InputModality.NO_INPUT})
    assert any(annotation.key == "symptoms" for annotation in symptoms.input_annotations)


def test_imputation_rejects_mask_boundaries_that_disagree_with_indices(tmp_path):
    _case(
        tmp_path,
        "cgmacros",
        "non_meal_imputation_cgm_only",
        {
            "window_df": _window(),
            "mask_start": "2025-01-01 00:30:00",
            "mask_end": "2025-01-01 01:00:00",
            "mask_indices": list(range(30, 60)),
            "GT": [90.0] * 30,
        },
    )

    with pytest.raises(TimeFFormatError, match="mask boundaries disagree"):
        HeartsConnector().convert([HeartsSource(tmp_path)])
