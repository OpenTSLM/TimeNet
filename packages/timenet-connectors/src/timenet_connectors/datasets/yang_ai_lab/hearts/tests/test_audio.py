"""Check the audio cases: waveforms at their stated rate, symptoms as flags, and how answers are spelt."""

import pickle

import numpy as np
import pytest

from timenet.client import TimeNet
from timenet.dataset import RegularAxis
from timenet.types import InputModality
from timenet.writer import TimeFWriter
from timenet_connectors.datasets.yang_ai_lab.hearts.connector import HeartsConnector


SYMPTOMS = {
    "cough": True,
    "fever": False,
    "cold": False,
    "diarrhoea": False,
    "loss_of_smell": True,
    "mp": False,
    "st": False,
    "bd": False,
    "ftg": False,
}
BREATHING = [0.0, 0.1, -0.1, 0.05]
MFCC_MEAN = [float(index) for index in range(13)]
MFCC_STD = [index / 2 for index in range(13)]


def _coswara(kind, values, **extra):
    signal = np.asarray(values, dtype=np.float32)
    data = {"audio_type": kind, "signal": signal, "sr": 48000, "duration": len(values) / 48000, "quality": 2}
    return {"subject_id": "gJ6kVzTn", "audio_type": kind, "data": {**data, "subject_id": "gJ6kVzTn"}, **extra}


def _coughvid(**extra):
    return {"subject_id": "bc80179d", "audio": np.asarray([0.5, -0.5, 0.25], dtype=np.float32), "sr": 48000, **extra}


CASES = {
    ("coswara", "audio_classification"): _coswara("breathing-deep", BREATHING, GT="breathing"),
    ("coswara", "cough_covid_status_classification_with_symptoms"): _coswara(
        "cough-heavy", [0.2, -0.2, 0.1], symptoms=SYMPTOMS, GT="covid_positive"
    ),
    ("coswara", "cough_covid_status_classification_symptoms_only"): {
        "subject_id": "kDI4V7d7",
        "symptoms": SYMPTOMS,
        "GT": "healthy",
    },
    ("coughvid", "cough_detection_good_qual"): _coughvid(GT=True),
    ("coughvid", "diagnosis_classification"): _coughvid(GT="upper_infection"),
    ("coughvid", "mfcc_mean_std"): _coughvid(GT={"mfcc_mean": MFCC_MEAN, "mfcc_std": MFCC_STD}),
    ("vctk", "waveform_temporal_direction_detection"): {
        "speaker_id": "p361",
        "recording_id": "p361_p361_398",
        "waveform": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        "GT": 1,
    },
}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("hearts")
    for (corpus, task), payload in CASES.items():
        directory = root / corpus / task
        directory.mkdir(parents=True)
        (directory / "0.pkl").write_bytes(pickle.dumps(payload))
    return HeartsConnector().convert([root])


def _task(dataset, corpus, task):
    return next(item for item in dataset.tasks if item.id == f"hearts-{corpus}-{task}-00-task")


def _by_key(annotations):
    return {annotation.key: annotation for annotation in annotations}


def test_waveforms_are_audio_signals_at_the_stated_rate(dataset):
    task = _task(dataset, "coswara", "audio_classification")
    assert task.resolved_input_modalities == {InputModality.TEXT, InputModality.AUDIO}
    (record,) = task.inputs
    assert record.start_time.timestamp is None
    assert record.subject_ids == ()
    assert record.metadata == {"corpus": "coswara"}
    assert [source.name for source in record.sources] == ["data.signal"]
    (signal,) = record.signals
    assert (signal.name, signal.spec.spec_type, signal.n_values) == ("audio", "audio", 4)
    assert signal.time_axis == RegularAxis.from_rate_hz(48000)
    assert signal.to_arrow().to_pylist() == pytest.approx(BREATHING)
    annotations = _by_key(record.annotations)
    assert annotations["subject_id"].value == "coswara:gJ6kVzTn"
    assert annotations["audio_quality"].value == 2
    assert task.targets == ("breathing",)
    assert task.metadata == {"corpus": "coswara", "task": "audio_classification", "testcase_idx": 0}
    assert task.prompt.startswith("Analyze the audio in the signal 'audio'.\n\nClassify the audio")


def test_vctk_waveforms_read_at_16_khz_and_answer_with_a_direction(dataset):
    task = _task(dataset, "vctk", "waveform_temporal_direction_detection")
    (record,) = task.inputs
    assert record.subject_ids == ()
    assert record.metadata == {"corpus": "vctk", "recording_id": "p361_p361_398"}
    assert _by_key(record.annotations)["subject_id"].value == "vctk:p361"
    assert record.signals[0].time_axis == RegularAxis.from_rate_hz(16_000)
    assert task.targets == ("reversed",)
    assert (
        "sampled at 16000 Hz. The waveform may be playing forward (normal) or time-reversed (backward). Speaker: p361."
        in task.prompt
    )


def test_symptoms_are_flags_on_the_record_and_spelt_out_in_the_prompt(dataset):
    task = _task(dataset, "coswara", "cough_covid_status_classification_with_symptoms")
    flags = {annotation.key: annotation for annotation in task.input_annotations}
    assert {key: flag.value for key, flag in flags.items()} == SYMPTOMS
    assert flags["loss_of_smell"].description == "Loss of smell"
    assert all(annotation in task.inputs[0].annotations for annotation in task.input_annotations)
    assert (
        "Symptoms: Cough: yes, Fever: no, Cold: no, Diarrhoea: no, Loss of smell: yes, Muscle pain: no, "
        "Sore throat: no, Breathing difficulty: no, Fatigue: no\n\n"
    ) in task.prompt
    assert task.targets == ("covid_positive",)


def test_symptoms_only_case_has_no_record_and_keeps_its_flags_on_the_task(dataset):
    task = _task(dataset, "coswara", "cough_covid_status_classification_symptoms_only")
    assert task.inputs == ()
    assert task.input_annotations == ()
    assert task.resolved_input_modalities == {InputModality.TEXT, InputModality.NO_INPUT}
    assert {annotation.key: annotation.value for annotation in task.annotations} == SYMPTOMS
    assert task.prompt.startswith(
        "Analyze the subject's symptoms to classify the subject as either healthy or covid positive."
    )


def test_answers_are_spelt_as_the_harness_scores_them(dataset):
    assert _task(dataset, "coughvid", "cough_detection_good_qual").targets == ("true",)
    diagnosis = _task(dataset, "coughvid", "diagnosis_classification")
    assert diagnosis.targets == ("upper_infection",)
    assert diagnosis.target_schema == "hearts-options-diagnosis_classification"
    coefficients = _task(dataset, "coughvid", "mfcc_mean_std")
    assert coefficients.targets == (*MFCC_MEAN, *MFCC_STD)
    assert (coefficients.unit, coefficients.target_name) == (None, "mfcc_mean_std")
    assert [source.name for source in coefficients.inputs[0].sources] == ["audio"]


def test_audio_values_round_trip_through_the_writer(dataset, tmp_path):
    with TimeFWriter(tmp_path / "registry", dataset, values_backend="zarr") as writer:
        writer.write()
    read = TimeNet(registry=tmp_path / "registry").load("yang-ai-lab/hearts", auto_build=False)
    task = _task(read, "coswara", "audio_classification")
    (signal,) = task.inputs[0].signals
    assert signal.to_arrow().to_pylist() == pytest.approx(BREATHING)
    assert signal.time_axis == RegularAxis.from_rate_hz(48000)
    assert task.inputs[0].metadata == {"corpus": "coswara"}
    assert _by_key(task.inputs[0].annotations)["subject_id"].value == "coswara:gJ6kVzTn"
    assert task.metadata == {"corpus": "coswara", "task": "audio_classification", "testcase_idx": 0}
