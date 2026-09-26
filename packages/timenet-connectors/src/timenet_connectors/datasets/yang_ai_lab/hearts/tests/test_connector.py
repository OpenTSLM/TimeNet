"""Check the HEARTS task families that the older proposal left out."""

from io import BytesIO
import pickle

import numpy as np
import pandas as pd
from PIL import Image

from timenet.dataset import Record
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
            "Timestamp": np.array(
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
            "window_df": _window(with_hr=True),
            "mask_start": "2025-01-01 00:30:00",
            "mask_end": "2025-01-01 00:59:00",
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
    assert InputModality.IMAGE in meal.input_modalities
    assert {signal.name for signal in meal.inputs[0].signals} >= {"a.jpg", "b.jpg", "c.jpg", "d.jpg"}
    assert symptoms.inputs == ()
    assert symptoms.input_modalities == frozenset({InputModality.TEXT, InputModality.NO_INPUT})
    assert any(annotation.key == "symptoms" for annotation in symptoms.input_annotations)
