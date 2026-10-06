"""Task definitions and prompts for HEARTS coughvid."""

from pathlib import Path

from timenet.types import ClassificationTask, ScalarPredictionTask
from timenet_connectors.datasets.yang_ai_lab.hearts_core.release import TaskDef, Waveform, load_prompts, vocabularies


_YES_NO = ("true", "false")

_COUGHVID = Waveform(("audio",), rate=("sr",))
_COUGH_DETECTION = TaskDef(ClassificationTask, inputs=(_COUGHVID,), options=_YES_NO)

TASKS: dict[str, TaskDef] = {
    "coughvid/cough_detection_good_qual": _COUGH_DETECTION,
    "coughvid/cough_detection_poor_qual": _COUGH_DETECTION,
    "coughvid/covid_status_classification": _COUGH_DETECTION,
    "coughvid/diagnosis_classification": TaskDef(
        ClassificationTask,
        inputs=(_COUGHVID,),
        options=("upper_infection", "lower_infection", "obstructive_disease", "COVID-19", "healthy_cough"),
    ),
    "coughvid/health_status_classification": _COUGH_DETECTION,
    "coughvid/mfcc_mean_std": TaskDef(
        ScalarPredictionTask, inputs=(_COUGHVID,), target_name="mfcc_mean_std", fields=("mfcc_mean", "mfcc_std")
    ),
}

PROMPTS = load_prompts(Path(__file__).with_name("prompts.yaml"))
VOCABULARIES = vocabularies(TASKS)
