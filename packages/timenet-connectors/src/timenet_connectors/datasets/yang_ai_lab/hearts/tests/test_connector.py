from fractions import Fraction
import json
from pathlib import Path
import pickle
import shutil
from typing import Any

import huggingface_hub
import numpy as np
import pandas as pd
import pytest

from timenet.connectors import BaseConnector
from timenet.dataset.axis import IrregularAxis, RegularAxis
from timenet.engine import store_dataset
from timenet.errors import TimeFFormatError, TimeNetDownloadError
from timenet.reader import TimeFReader
from timenet.registry import DatasetVersion
from timenet.types import AnswerTask, ClassificationTask, TemporalLocalizationTask
from timenet_connectors.datasets.yang_ai_lab.hearts import series as series_module
from timenet_connectors.datasets.yang_ai_lab.hearts.connector import (
    _ID_PREFIX,
    HF_REPO,
    HF_REVISION,
    HeartsConnector,
    HeartsSource,
    _subject_ids,
)
from timenet_connectors.datasets.yang_ai_lab.hearts.pickles import load_payload
from timenet_connectors.datasets.yang_ai_lab.hearts.tasks import EXCLUDED, TASK_DEFINITIONS, name_vocabulary


# Synthetic payloads shaped exactly like the released pickles this connector converts. Every series
# below is a straight linear ramp, which no real recording is, so a reader can tell these from
# released data at a glance. They hold no bytes from the real dataset, whose upstream corpora are
# human-subject recordings under unstated licence terms.

_RESPIRATION = np.linspace(0.26, 0.69, 200)
_HEART_RATE = np.linspace(0.50, 0.69, 20)
_SIGNAL = np.linspace(-0.18, 0.18, 441, dtype=np.float32)
_WAVEFORM = np.linspace(-0.13, 0.15, 320, dtype=np.float32)
_COUGH = np.linspace(-0.42, 0.44, 480, dtype=np.float32)
# The released MFCC answers hold plain Python floats: 13 means and 13 standard deviations.
_MFCC_MEAN = [float(value) for value in np.linspace(-380.5, 12.25, 13)]
_MFCC_STD = [float(value) for value in np.linspace(1.5, 40.25, 13)]
# Twenty-three one-minute steps then thirteen seven-minute ones: the released windows are not uniform.
_MEAL_MINUTES = [*range(23), *range(29, 114, 7)]
_MEAL_GLUCOSE = np.linspace(75.6, 96.0, len(_MEAL_MINUTES))
_IAUC_MINUTES = [float(minute) for minute in [*range(45), *range(52, 129, 7)]]
_IAUC_GLUCOSE = np.linspace(55.3, 120.5, len(_IAUC_MINUTES))

_SYMPTOMS_DIR = "coswara/cough_covid_status_classification_with_symptoms"
_SYMPTOMS_RECORD = "hearts-coswara-cough_covid_status_classification_with_symptoms-00"


def _elapsed_frame(minutes, values, column, extra_index=False):
    offsets = pd.to_timedelta(np.asarray(minutes, dtype="int64"), unit="m")
    frame = pd.DataFrame(
        {
            "Timestamp": [str(stamp) for stamp in pd.Timestamp("2023-11-05 12:19:00") + offsets],
            column: values,
        }
    )
    if extra_index:
        frame["timestamp_min"] = np.asarray(minutes, dtype="int64")
    return frame


def _paced_frame(values, step_us, column):
    return pd.DataFrame(
        {
            "timestamp": pd.to_timedelta(np.arange(len(values)) * step_us, unit="us"),
            column: values,
        }
    )


def _payloads():
    return {
        "cgmacros/iauc_calculation/0.pkl": {
            "subject_id": "CGMacros-007",
            "meal_time": pd.Timestamp("2023-11-06 08:31:00"),
            "cgm_df": pd.DataFrame({"Time (min)": _IAUC_MINUTES, "CGM (mg/dL)": _IAUC_GLUCOSE}),
            "GT": 859.8333333333333,
        },
        # As the release states this case: timestamp_min counts from zero inside the window, while
        # window_start and window_end are absolute minute counts 120 apart. Both readings are here
        # because both are in the released payload.
        "cgmacros/meal_time_localization/7.pkl": {
            "subject_id": "CGMacros-007",
            "window_df": _elapsed_frame(_MEAL_MINUTES, _MEAL_GLUCOSE, "Libre GL", extra_index=True),
            "window_start": 28319779,
            "window_end": 28319899,
            "GT": np.int64(4),
        },
        "coswara/audio_classification/30.pkl": {
            "subject_id": "tOlOwYDEHCRx1QegeMOPaydVcgv1",
            "audio_type": "vowel-o",
            "data": {
                "audio_type": "vowel-o",
                "signal": _SIGNAL,
                "sr": 44100,
                "duration": len(_SIGNAL) / 44100,
                "quality": 2,
                "subject_id": "tOlOwYDEHCRx1QegeMOPaydVcgv1",
            },
            "GT": "speech",
        },
        f"{_SYMPTOMS_DIR}/0.pkl": {
            "subject_id": "JxtCdZIjW1VjLM92IYEhw3xz3pS2",
            "audio_type": "cough-shallow",
            "data": {
                "audio_type": "cough-shallow",
                "signal": _SIGNAL,
                "sr": 48000,
                "duration": len(_SIGNAL) / 48000,
                "quality": 2,
                "subject_id": "JxtCdZIjW1VjLM92IYEhw3xz3pS2",
            },
            "symptoms": {"cough": True, "fever": False, "cold": False},
            "GT": "healthy",
        },
        "coughvid/mfcc_mean_std/12.pkl": {
            "subject_id": "1f0f4a1e-2a4a-4d0a-9d4e-2f6a1b3c5d7e",
            "audio": _COUGH,
            "sr": 48000,
            "GT": {"mfcc_mean": list(_MFCC_MEAN), "mfcc_std": list(_MFCC_STD)},
        },
        "harespod/hr_resp_pairing/0.pkl": {
            "subject_id": "223b",
            "respiration_dfs": {
                "respiration_A": _paced_frame(_RESPIRATION, 10_000, "rsp"),
                "respiration_B": _paced_frame(_RESPIRATION[::-1], 10_000, "rsp"),
            },
            "hr_dfs": {
                "hr_1": _paced_frame(_HEART_RATE, 1_000_000, "hr"),
                "hr_2": _paced_frame(_HEART_RATE[::-1], 1_000_000, "hr"),
            },
            "GT": {"A": "1", "B": "2"},
        },
        "vctk/waveform_temporal_direction_detection/39.pkl": {
            "speaker_id": "p361",
            "recording_id": "p361_p361_309",
            "waveform": _WAVEFORM,
            "GT": 0,
        },
        # Excluded: the answer is a series of held-out values, so the case has no TimeF record.
        "cgmacros/meal_forecasting_no_ref/0.pkl": {
            "subject_id": "CGMacros-005",
            "window_df": _elapsed_frame(list(range(60)), np.linspace(118.0, 124.9, 60), "Libre GL"),
            "meal_time": pd.Timestamp("2020-08-22 16:48:00"),
            "GT": [118.4, 118.0, 117.6],
        },
    }


def _write_tree(root: Path) -> Path:
    for relative, payload in _payloads().items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=4)
    return root


def _write_case(root: Path, relative: str, payload: dict[str, Any]) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    return path


def _write_empty_case_files(root: Path) -> Path:
    # What a complete fetch of the pinned revision looks like to the count guard: every in-scope
    # case present, and nothing read.
    for definition in TASK_DEFINITIONS:
        directory = root / definition.source / definition.task
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(definition.n_items):
            (directory / f"{index}.pkl").touch()
    return root


@pytest.fixture(scope="module")
def tree(tmp_path_factory) -> Path:
    return _write_tree(tmp_path_factory.mktemp("hearts"))


@pytest.fixture(scope="module")
def dataset(tree):
    return HeartsConnector().convert([HeartsSource(root=tree)])


def _record(dataset, record_id):
    return next(record for record in dataset.records if record.record_id == record_id)


def _task(dataset, record_id):
    # The tasks stream, so they are read off the source rather than held on the dataset.
    return next(task for task in dataset.iter_tasks() if task.record_ids == (record_id,))


def test_is_a_connector():
    assert isinstance(HeartsConnector(), BaseConnector)


def test_metadata():
    metadata = HeartsConnector().metadata()
    assert metadata.dataset_id == "yang-ai-lab/hearts"
    assert str(metadata.license) == "other"


def test_converts_one_record_per_test_case_in_canonical_order(dataset):
    assert [record.record_id for record in dataset.records] == [
        "hearts-cgmacros-iauc_calculation-00",
        "hearts-cgmacros-meal_time_localization-07",
        "hearts-coswara-audio_classification-30",
        _SYMPTOMS_RECORD,
        "hearts-coughvid-mfcc_mean_std-12",
        "hearts-harespod-hr_resp_pairing-00",
        "hearts-vctk-waveform_temporal_direction_detection-39",
    ]


def test_every_id_this_connector_writes_is_built_on_one_prefix(dataset):
    written = (
        [record.record_id for record in dataset.records]
        + [series.time_series_id for record in dataset.records for series in record.time_series]
        + [annotation.id for record in dataset.records for annotation in record.annotations]
        + [annotation.id for annotation in dataset.registered_annotations]
        + [task.id for task in dataset.iter_tasks()]
    )
    assert written
    assert not [identifier for identifier in written if not str(identifier).startswith(f"{_ID_PREFIX}-")]


def test_excluded_task_directories_produce_no_record(dataset):
    assert "cgmacros/meal_forecasting_no_ref" in EXCLUDED
    assert not [record for record in dataset.records if "meal_forecasting" in record.record_id]


def test_a_file_not_named_after_its_index_fails_by_name(tmp_path):
    _write_case(tmp_path, "vctk/waveform_temporal_direction_detection/last.pkl", {"waveform": _WAVEFORM, "GT": 0})
    with pytest.raises(TimeFFormatError, match="not a test-case index"):
        HeartsConnector().convert([HeartsSource(root=tmp_path)])


def test_schema_derives(dataset):
    schema = dataset.derive_schema()
    assert {spec.spec_type for spec in schema.time_series_specs} == {
        "cgm",
        "audio_coswara",
        "audio_coughvid",
        "audio_vctk",
        "respiration_norm",
        "heart_rate_norm",
    }


def test_a_uniform_frame_gets_a_cadence_and_a_wavering_one_gets_offsets(dataset):
    pairing = _record(dataset, "hearts-harespod-hr_resp_pairing-00")
    axes = {series.signal: series.time_axis for series in pairing.time_series}
    assert axes["respiration_dfs.respiration_A.rsp"] == RegularAxis(period_us=Fraction(10_000))
    assert axes["hr_dfs.hr_1.hr"] == RegularAxis(period_us=Fraction(1_000_000))

    localization = _record(dataset, "hearts-cgmacros-meal_time_localization-07").time_series[0]
    assert localization.time_axis == IrregularAxis(first_us=0, last_us=113 * 60_000_000)
    offsets = localization.time_offsets_us()
    assert len(offsets) == localization.n_values == len(_MEAL_MINUTES)
    assert offsets[0] == 0


def test_audio_rate_comes_from_the_payload_and_vctk_from_the_reference_implementation(dataset):
    coswara = _record(dataset, "hearts-coswara-audio_classification-30").time_series[0]
    assert coswara.signal == "data.signal"
    assert coswara.time_axis == RegularAxis.from_rate_hz(44_100)
    vctk = _record(dataset, "hearts-vctk-waveform_temporal_direction_detection-39").time_series[0]
    assert vctk.time_axis == RegularAxis.from_rate_hz(16_000)


def test_values_are_bit_exact_and_keep_their_dtype(dataset):
    respiration = next(
        series
        for series in _record(dataset, "hearts-harespod-hr_resp_pairing-00").time_series
        if series.signal == "respiration_dfs.respiration_A.rsp"
    )
    assert np.array_equal(respiration.to_numpy(), _RESPIRATION)
    assert respiration.to_numpy().dtype == np.dtype("float64")
    audio = _record(dataset, "hearts-coswara-audio_classification-30").time_series[0]
    assert np.array_equal(audio.to_numpy(), _SIGNAL)
    assert audio.to_numpy().dtype == np.dtype("float32")


def test_convert_builds_no_values_and_keeps_no_frames(tmp_path, monkeypatch):
    root = _write_tree(tmp_path / "tree")
    real_array = series_module.pa.array

    def refuse(*args, **kwargs):
        pytest.fail("convert() built an Arrow array; the values must stay behind the lazy loader")

    load_payload.cache_clear()
    monkeypatch.setattr(series_module.pa, "array", refuse)
    built = HeartsConnector().convert([HeartsSource(root=root)])
    monkeypatch.setattr(series_module.pa, "array", real_array)

    # Nothing convert returned holds a frame: drop the cache and the files, and no loader can read.
    load_payload.cache_clear()
    shutil.rmtree(root)
    with pytest.raises(OSError):
        _record(built, "hearts-vctk-waveform_temporal_direction_detection-39").time_series[0].to_arrow()


def test_loaders_can_be_called_again(dataset):
    series = _record(dataset, "hearts-vctk-waveform_temporal_direction_detection-39").time_series[0]
    assert np.array_equal(series.to_numpy(), series.to_numpy())


def test_audio_type_is_absent_because_the_release_derives_an_answer_from_it(dataset):
    keys = {annotation.key for record in dataset.records for annotation in record.annotations}
    assert "audio_type" not in keys
    assert "hearts_source" in keys


def test_the_provenance_annotations_say_where_a_record_came_from(dataset):
    record = _record(dataset, "hearts-coswara-audio_classification-30")
    values = {a.key: a.value for a in record.annotations if a.source == "hearts:provenance"}
    assert values["hearts_source"] == "coswara"
    assert values["hearts_task"] == "audio_classification"
    assert values["testcase_idx"] == 30
    assert values["audio_quality"] == 2


def test_subject_ids_are_namespaced_by_their_corpus(dataset):
    assert _record(dataset, "hearts-cgmacros-iauc_calculation-00").subject_ids == ("cgmacros-CGMacros-007",)
    assert _record(dataset, "hearts-vctk-waveform_temporal_direction_detection-39").subject_ids == ("vctk-p361",)


def test_a_two_window_case_carries_both_of_its_subjects():
    payload = {"A_subject": "CGMacros-003", "B_subject": "CGMacros-041"}
    assert _subject_ids("cgmacros", payload) == ("cgmacros-CGMacros-003", "cgmacros-CGMacros-041")
    assert _subject_ids("cgmacros", {}) == ()


def test_the_task_stream_gives_the_same_tasks_when_read_again(dataset):
    # set_task_stream reads its source more than once, so a generator that empties itself writes no
    # task at all. This stream walks the tree again, so reading it twice reads every payload twice.
    once = [(task.id, task.record_ids) for task in dataset.iter_tasks()]
    again = [(task.id, task.record_ids) for task in dataset.iter_tasks()]
    assert once == again
    assert len(once) == len(dataset.records)


def test_every_streamed_task_names_its_own_record_and_carries_a_distinct_id(dataset):
    # Nothing sets a streamed task's record_ids, and the dataset holds no task list, so its
    # duplicate-id check does not run. Both are pinned here instead.
    assert [task.record_ids for task in dataset.iter_tasks()] == [(record.record_id,) for record in dataset.records]
    ids = [task.id for task in dataset.iter_tasks()]
    assert len(set(ids)) == len(ids)


def test_classification_carries_the_label_and_the_id_of_its_vocabulary(dataset):
    task = _task(dataset, "hearts-coswara-audio_classification-30")
    assert isinstance(task, ClassificationTask)
    assert task.target == "speech"
    assert task.target_schema == name_vocabulary("hearts", "audio_classification")
    options = next(a for a in dataset.registered_annotations if a.id == task.target_schema)
    assert options.id in task.input_annotation_ids
    assert options.value == ["speech", "cough", "breathing"]


def test_symptoms_ride_as_an_agent_input_annotation(dataset):
    task = _task(dataset, _SYMPTOMS_RECORD)
    symptoms = next(a for a in _record(dataset, _SYMPTOMS_RECORD).annotations if a.key == "symptoms")
    assert symptoms.id in task.input_annotation_ids
    assert symptoms.value == {"cough": True, "fever": False, "cold": False}


def test_a_numpy_boolean_in_an_agent_input_becomes_plain_python(tmp_path):
    # The writer serializes an annotation value with json.dumps and no default=. np.bool_ is not a
    # bool subclass, so a NumPy scalar left in here fails the write after the whole download.
    payload = dict(_payloads()[f"{_SYMPTOMS_DIR}/0.pkl"])
    payload["symptoms"] = {"cough": np.bool_(True), "fever": np.bool_(False)}
    _write_case(tmp_path, f"{_SYMPTOMS_DIR}/0.pkl", payload)
    built = HeartsConnector().convert([HeartsSource(root=tmp_path)])
    symptoms = next(a for a in _record(built, _SYMPTOMS_RECORD).annotations if a.key == "symptoms")
    assert symptoms.value == {"cough": True, "fever": False}
    assert [type(value) for value in symptoms.value.values()] == [bool, bool]
    assert json.loads(json.dumps(symptoms.value)) == symptoms.value


def test_a_meal_minute_lands_inside_the_window_the_record_covers(dataset):
    # The writer measures a streamed localization target against the span the record's series
    # cover, so the round-trip test is also the check that the offsets and the answer are counted
    # from the same instant.
    task = _task(dataset, "hearts-cgmacros-meal_time_localization-07")
    assert isinstance(task, TemporalLocalizationTask)
    assert [span.start_us for span in task.target] == [4 * 60_000_000]


def test_a_float_answer_carries_no_digit_the_source_lacks(dataset):
    task = _task(dataset, "hearts-coughvid-mfcc_mean_std-12")
    assert isinstance(task, AnswerTask)
    parsed = json.loads(task.target)
    assert all(isinstance(value, float) for value in parsed["mfcc_mean"] + parsed["mfcc_std"])
    assert [repr(value) for value in parsed["mfcc_mean"]] == [repr(value) for value in _MFCC_MEAN]
    assert [repr(value) for value in parsed["mfcc_std"]] == [repr(value) for value in _MFCC_STD]


def test_harespod_values_declare_no_unit_and_say_why(dataset):
    record = _record(dataset, "hearts-harespod-hr_resp_pairing-00")
    assert {str(series.spec.unit_value) for series in record.time_series} == {"dimensionless"}
    normalized = next(a for a in record.annotations if a.key == "values_normalized")
    assert normalized.value is True
    assert "scaling constants" in str(normalized.description)


def test_a_tree_with_nothing_in_scope_fails(tmp_path):
    with pytest.raises(TimeFFormatError, match="no test case"):
        HeartsConnector().convert([HeartsSource(root=tmp_path)])


def test_download_asks_for_the_pin_and_only_the_directories_in_scope(monkeypatch, tmp_path):
    asked: dict[str, str] = {}
    patterns: list[str] = []

    def fake_snapshot_download(repo_id: str, **kwargs: Any) -> str:
        asked["repo_id"] = repo_id
        asked["repo_type"] = kwargs["repo_type"]
        asked["revision"] = kwargs["revision"]
        asked["cache_dir"] = kwargs["cache_dir"]
        patterns.extend(kwargs["allow_patterns"])
        return str(_write_empty_case_files(tmp_path / "snapshot"))

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    sources = HeartsConnector().download(tmp_path / "cache")

    assert asked == {
        "repo_id": HF_REPO,
        "repo_type": "dataset",
        "revision": HF_REVISION,
        "cache_dir": str(tmp_path / "cache"),
    }
    assert len(patterns) == len(TASK_DEFINITIONS) == 21
    assert "coswara/audio_classification/*.pkl" in patterns
    assert not [pattern for pattern in patterns if pattern.removesuffix("/*.pkl") in EXCLUDED]
    assert sources[0].root == tmp_path / "snapshot"
    assert sum(definition.n_items for definition in TASK_DEFINITIONS) == 1005


def test_download_refuses_a_revision_that_moved(monkeypatch, tmp_path):
    definition = TASK_DEFINITIONS[0]

    def fake_snapshot_download(repo_id: str, **kwargs: Any) -> str:
        root = _write_empty_case_files(tmp_path / "snapshot")
        (root / definition.source / definition.task / "0.pkl").unlink()
        return str(root)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    with pytest.raises(TimeNetDownloadError, match="test cases at revision") as refusal:
        HeartsConnector().download(tmp_path / "cache")
    assert definition.directory in str(refusal.value)


def test_convert_round_trips_through_the_writer(tmp_path, dataset):
    dataset.derive_schema()
    version_dir = store_dataset(dataset, tmp_path / "out")
    with TimeFReader(DatasetVersion.open_local(version_dir)) as reader:
        restored = reader.read()
    assert len(restored.records) == len(dataset.records)
    assert len(restored.tasks) == len(list(dataset.iter_tasks()))
    assert {annotation.key for annotation in restored.registered_annotations} == {"answer_options"}
    audio = next(r for r in restored.records if r.record_id == "hearts-coswara-audio_classification-30")
    assert np.array_equal(audio.time_series[0].to_numpy(), _SIGNAL)
