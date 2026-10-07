"""Task definitions and prompts for HEARTS coswara."""

from pathlib import Path

from timenet.types import ClassificationTask
from timenet_connectors.datasets.yang_ai_lab.hearts_core.release import TaskDef, Waveform, load_prompts, vocabularies


SYMPTOM_NAMES: dict[str, str] = {
    "cough": "Cough",
    "fever": "Fever",
    "cold": "Cold",
    "diarrhoea": "Diarrhoea",
    "loss_of_smell": "Loss of smell",
    "mp": "Muscle pain",
    "st": "Sore throat",
    "bd": "Breathing difficulty",
    "ftg": "Fatigue",
}

_COVID = ("healthy", "covid_positive")

_COSWARA = Waveform(("data", "signal"), rate=("data", "sr"))
AUDIO_QUALITY_DESCRIPTION = "The Coswara annotators' rating of the recording's audio quality."

TASKS: dict[str, TaskDef] = {
    "coswara/audio_classification": TaskDef(
        ClassificationTask, inputs=(_COSWARA,), options=("speech", "cough", "breathing")
    ),
    "coswara/cough_covid_status_classification": TaskDef(ClassificationTask, inputs=(_COSWARA,), options=_COVID),
    "coswara/cough_covid_status_classification_with_symptoms": TaskDef(
        ClassificationTask, inputs=(_COSWARA,), options=_COVID
    ),
    "coswara/cough_covid_status_classification_symptoms_only": TaskDef(ClassificationTask, options=_COVID),
    "coswara/speech_covid_status_classification": TaskDef(ClassificationTask, inputs=(_COSWARA,), options=_COVID),
}

PROMPTS = load_prompts(Path(__file__).with_name("prompts.yaml"))
VOCABULARIES = vocabularies(TASKS)
