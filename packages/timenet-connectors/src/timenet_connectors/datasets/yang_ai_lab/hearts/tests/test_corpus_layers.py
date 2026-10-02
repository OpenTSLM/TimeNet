"""Check the five taskless HEARTS corpus record layers."""

import pickle

import pytest

from timenet.types import TimeInterval
from timenet_connectors.datasets.yang_ai_lab.hearts.tests.test_audio import CASES as AUDIO_CASES
from timenet_connectors.datasets.yang_ai_lab.hearts.tests.test_connector import (
    CASES as CGMACROS_CASES,
    DAY_US,
    MINUTE_US,
)
from timenet_connectors.datasets.yang_ai_lab.hearts.tests.test_harespod import CASES as HARESPOD_CASES
from timenet_connectors.datasets.yang_ai_lab.hearts_cgmacros.connector import HeartsCgmacrosConnector
from timenet_connectors.datasets.yang_ai_lab.hearts_coswara.connector import HeartsCoswaraConnector
from timenet_connectors.datasets.yang_ai_lab.hearts_coughvid.connector import HeartsCoughvidConnector
from timenet_connectors.datasets.yang_ai_lab.hearts_harespod.connector import HeartsHarespodConnector
from timenet_connectors.datasets.yang_ai_lab.hearts_vctk.connector import HeartsVctkConnector


_LAYERS = (
    (HeartsCgmacrosConnector, "cgmacros", "meal_forecasting", CGMACROS_CASES["meal_forecasting"], 1),
    (HeartsCoswaraConnector, "coswara", "audio_classification", AUDIO_CASES["coswara", "audio_classification"], 1),
    (
        HeartsCoughvidConnector,
        "coughvid",
        "diagnosis_classification",
        AUDIO_CASES["coughvid", "diagnosis_classification"],
        1,
    ),
    (
        HeartsHarespodConnector,
        "harespod",
        "altitude_ranking_respiration",
        HARESPOD_CASES["altitude_ranking_respiration"],
        3,
    ),
    (
        HeartsVctkConnector,
        "vctk",
        "waveform_temporal_direction_detection",
        AUDIO_CASES["vctk", "waveform_temporal_direction_detection"],
        1,
    ),
)


@pytest.mark.parametrize("layer", _LAYERS)
def test_layer_builds_only_reusable_records(tmp_path, layer):
    connector_type, corpus, task, payload, record_count = layer
    directory = tmp_path / corpus / task
    directory.mkdir(parents=True)
    (directory / "0.pkl").write_bytes(pickle.dumps(payload))

    dataset = connector_type().convert([tmp_path])

    assert len(dataset.records) == record_count
    assert dataset.tasks == ()
    assert dataset.metadata.dataset_id == f"yang-ai-lab/hearts-{corpus}"
    assert all(record.metadata["corpus"] == corpus for record in dataset.records)


def test_forecast_horizon_is_structural_parent_data(tmp_path):
    directory = tmp_path / "cgmacros" / "meal_forecasting"
    directory.mkdir(parents=True)
    (directory / "0.pkl").write_bytes(pickle.dumps(CGMACROS_CASES["meal_forecasting"]))

    dataset = HeartsCgmacrosConnector().convert([tmp_path])

    assert dataset.records[0].time_span == TimeInterval.micros(0, 2 * DAY_US + 90 * MINUTE_US)
