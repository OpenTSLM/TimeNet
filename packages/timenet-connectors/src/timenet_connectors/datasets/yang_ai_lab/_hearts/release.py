"""Shared description of the pinned HEARTS source release and its task directories.

Every test case is one pickle holding a dictionary. A DataFrame in it becomes a Source whose value
columns become Signals, a waveform array becomes one audio Signal, and the case's answer becomes
the task its directory's :class:`TaskDef` names. The prompts live in ``prompts.yaml`` beside this
module, keyed by task directory.
"""

from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from string import Template

import yaml

from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    ForecastingTask,
    InputModality,
    ScalarPredictionTask,
    Task,
    TemporalLocalizationTask,
    TimeSeriesSpec,
    TSCorrespondenceTask,
    TSEditingTask,
    ureg,
)


REPO = "yang-ai-lab/HEARTS"
REVISION = "7c18df521ae36cbc6b61e17782f1ac08dc378ea1"
"""Pinned source revision for reproducible builds."""
ANSWER_KEY = "GT"
"""The key holding a case's answer."""
HELD_OUT_READINGS = 30
"""How many one-minute readings a forecast or imputation answer holds."""
US_PER_MINUTE = 60_000_000
"""Whole microseconds in one minute."""

CGM = TimeSeriesSpec(spec_type="cgm", name="Interstitial glucose", unit_value=ureg.Unit("mg/dL"), dtype="float64")
ACTIVITY_CALORIES = TimeSeriesSpec(
    spec_type="activity_calories", name="Activity calories", unit_value=ureg.kilocalorie, dtype="float64"
)
HEART_RATE = TimeSeriesSpec(spec_type="heart_rate", name="Heart rate", unit_value=ureg.Unit("bpm"), dtype="float64")
# HARESPOD's release ships these three min-max scaled to the unit interval, without the constants.
RESPIRATION_NORM = TimeSeriesSpec(
    spec_type="respiration_norm", name="Respiration (min-max scaled)", unit_value=ureg.dimensionless, dtype="float64"
)
SPO2_NORM = TimeSeriesSpec(
    spec_type="spo2_norm", name="Oxygen saturation (min-max scaled)", unit_value=ureg.dimensionless, dtype="float64"
)
HEART_RATE_NORM = TimeSeriesSpec(
    spec_type="heart_rate_norm", name="Heart rate (min-max scaled)", unit_value=ureg.dimensionless, dtype="float64"
)
AUDIO = TimeSeriesSpec(
    spec_type="audio",
    name="Audio waveform",
    unit_value=ureg.dimensionless,
    dtype="float32",
    modality=InputModality.AUDIO,
)

COLUMN_SPECS: dict[str, TimeSeriesSpec] = {
    "Libre GL": CGM,
    "CGM (mg/dL)": CGM,
    "HR": HEART_RATE,
    "Calories (Activity)": ACTIVITY_CALORIES,
    "rsp": RESPIRATION_NORM,
    "spo": SPO2_NORM,
    "hr": HEART_RATE_NORM,
}
"""The spec of every value column a released frame holds, by the column name the release uses."""
TIME_COLUMNS = ("Timestamp", "timestamp", "Time (min)")
"""The column that places a frame's rows in time. CGMacros writes naive wall-clock text, HARESPOD
naive datetimes or timedeltas from the segment's start, and the iAUC frames minutes since the meal."""
DERIVED_COLUMNS = ("timestamp_min",)
"""Columns the release derives from the time column. The harness never showed them, so they become no Signal."""

MEAL_COLUMNS: tuple[tuple[str, str, str | None], ...] = (
    ("Meal Type", "meal_type", None),
    ("Calories", "calories", "kilocalorie"),
    ("Carbs", "carbs", "gram"),
    ("Protein", "protein", "gram"),
    ("Fat", "fat", "gram"),
    ("Fiber", "fiber", "gram"),
)
"""A meal's columns in the release, the annotation key each becomes, and its unit."""
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
"""Coswara's symptom keys, and the names the harness spelt them out with."""
PHOTOGRAPHS = ("a.jpg", "b.jpg", "c.jpg", "d.jpg")
"""The meal photographs of a case, in option order."""
DESCRIPTIONS: dict[str, str] = {
    "answer_options": "The closed set of answers the reference harness accepts for this task.",
    "subject_id": "The subject identifier, qualified by the HEARTS corpus directory.",
    "recording_start_local": (
        "The local date and time of the first sample, exactly as the source states it. The source names no time zone."
    ),
    "meal_time": "When the meal the task asks about started.",
    "cgm_mask": "The stretch whose glucose readings the release zeroed and asks the model to impute.",
    "audio_quality": "The Coswara annotators' rating of the recording's audio quality.",
}
"""Descriptions stored in the dataset schema for the annotations the connector attaches, by key."""


@unique
class Answer(StrEnum):
    """How a classification answer is spelt in the release."""

    LABEL = "label"
    """The label itself."""
    BOOLEAN = "boolean"
    """True or False, which the harness scores as the labels ``true`` and ``false``."""
    OPTION_INDEX = "option_index"
    """The position of the label among the options."""
    NORMAL_WINDOW = "normal_window"
    """A window-to-status mapping; the label is the window whose subject is ``Normal``."""


@dataclass(frozen=True)
class Waveform:
    """A waveform array of the case, sampled at a rate the case states or the release implies."""

    keys: tuple[str, ...]
    """The key path of the array inside the test case."""
    rate: tuple[str, ...] | int
    """The key path of the sampling rate in hertz, or the rate itself when the case states none."""
    quality: tuple[str, ...] | None = None
    """The key path of the annotators' quality rating, when the corpus has one."""


@dataclass(frozen=True)
class TaskDef:
    """One task directory: what its cases hold, and which TimeF task their answer becomes."""

    task_type: type[Task]
    """The task class the directory's answer maps to."""
    inputs: tuple[str | tuple[str, ...] | Waveform, ...] = ()
    """The input records of a case, in the order the prompt names them. An entry is the key of a
    frame, a tuple of frame keys that share one record and one clock, the key of a dictionary of
    frames that becomes one record per entry, or a waveform."""
    candidates: tuple[str, ...] = ()
    """Keys of dictionaries of frames whose entries are the candidates a pairing task matches to."""
    options: tuple[str, ...] = ()
    """The closed answer vocabulary of a classification task."""
    answer: Answer = Answer.LABEL
    """How a classification label is spelt in the source answer."""
    unit: str | None = None
    """The unit of a scalar answer."""
    target_name: str | None = None
    """The name of the quantity a scalar answer predicts."""
    fields: tuple[str, ...] = ()
    """The keys of a mapping answer, in the order their values become scalar targets."""
    meal_info: bool = False
    """Whether the case tells the model about the meal it forecasts from."""
    symptoms: bool = False
    """Whether the case tells the model the subject's symptoms."""
    images: bool = False
    """Whether the case supplies four meal photographs."""


_COSWARA = Waveform(("data", "signal"), rate=("data", "sr"), quality=("data", "quality"))
_COUGHVID = Waveform(("audio",), rate=("sr",))
# The release states no rate for VCTK. Its harness describes the waveforms as 16 kHz, and their
# pitch periods bear that out: read at 48 kHz, the speakers' voices would sit near 600 Hz.
_VCTK = Waveform(("waveform",), rate=16_000)
_COVID = ("healthy", "covid_positive")
_YES_NO = ("true", "false")
_FORECAST = TaskDef(ForecastingTask, inputs=("window_df",))
_FORECAST_WITH_REFERENCE = TaskDef(ForecastingTask, inputs=(("reference_cgm_df", "window_df"),))
_IMPUTATION = TaskDef(TSEditingTask, inputs=("window_df",))
_COUGH_DETECTION = TaskDef(ClassificationTask, inputs=(_COUGHVID,), options=_YES_NO, answer=Answer.BOOLEAN)
_RANKING = TaskDef(AnswerTask, inputs=("segment_dfs",))

TASKS: dict[str, TaskDef] = {
    "cgmacros/a1c_classification": TaskDef(
        ClassificationTask, inputs=("window_df",), options=("normal", "prediabetes", "diabetes")
    ),
    "cgmacros/cgm_stat_calculation": TaskDef(
        ScalarPredictionTask,
        inputs=("cgm",),
        unit="percent",
        target_name="time_below_and_above_range",
        fields=("below", "above"),
    ),
    "cgmacros/fasting_glu_prediction": TaskDef(
        ScalarPredictionTask, inputs=("window_df",), unit="mg/dL", target_name="fasting_glucose"
    ),
    "cgmacros/iauc_calculation": TaskDef(
        ScalarPredictionTask, inputs=("cgm_df",), unit="mg*min/dL", target_name="postprandial_iauc"
    ),
    "cgmacros/meal_react_comparison": TaskDef(
        ClassificationTask, inputs=("A", "B"), options=("A", "B"), answer=Answer.NORMAL_WINDOW
    ),
    "cgmacros/meal_time_localization": TaskDef(TemporalLocalizationTask, inputs=("window_df",)),
    "cgmacros/meal_forecasting": _FORECAST_WITH_REFERENCE,
    "cgmacros/meal_forecasting_meal_info": TaskDef(
        ForecastingTask, inputs=_FORECAST_WITH_REFERENCE.inputs, meal_info=True
    ),
    "cgmacros/meal_forecasting_no_ref": _FORECAST,
    "cgmacros/meal_forecasting_no_ref_meal_info": TaskDef(ForecastingTask, inputs=_FORECAST.inputs, meal_info=True),
    "cgmacros/meal_img_classification": TaskDef(
        ClassificationTask, inputs=("window_df",), options=PHOTOGRAPHS, images=True
    ),
    "cgmacros/non_meal_imputation_calories": _IMPUTATION,
    "cgmacros/non_meal_imputation_cgm_only": _IMPUTATION,
    "cgmacros/non_meal_imputation_hr": _IMPUTATION,
    "coswara/audio_classification": TaskDef(
        ClassificationTask, inputs=(_COSWARA,), options=("speech", "cough", "breathing")
    ),
    "coswara/cough_covid_status_classification": TaskDef(ClassificationTask, inputs=(_COSWARA,), options=_COVID),
    "coswara/cough_covid_status_classification_with_symptoms": TaskDef(
        ClassificationTask, inputs=(_COSWARA,), options=_COVID, symptoms=True
    ),
    "coswara/cough_covid_status_classification_symptoms_only": TaskDef(
        ClassificationTask, options=_COVID, symptoms=True
    ),
    "coswara/speech_covid_status_classification": TaskDef(ClassificationTask, inputs=(_COSWARA,), options=_COVID),
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
    "harespod/altitude_ranking_respiration": _RANKING,
    "harespod/altitude_ranking_spo2": _RANKING,
    "harespod/hr_resp_pairing": TaskDef(TSCorrespondenceTask, inputs=("respiration_dfs",), candidates=("hr_dfs",)),
    "harespod/spo2_resp_pairing": TaskDef(TSCorrespondenceTask, inputs=("respiration_dfs",), candidates=("spo_dfs",)),
    "vctk/waveform_temporal_direction_detection": TaskDef(
        ClassificationTask, inputs=(_VCTK,), options=("forward", "reversed"), answer=Answer.OPTION_INDEX
    ),
}
"""Every task directory this connector converts, in the order it walks them."""

PROMPTS: dict[str, Template] = {
    directory: Template(text)
    for directory, text in yaml.safe_load(Path(__file__).with_name("prompts.yaml").read_text(encoding="utf-8")).items()
}
"""The prompt of each task directory, with a placeholder for each value the harness filled in per case."""

VOCABULARIES: dict[str, Annotation] = {
    directory: Annotation(
        key="answer_options",
        value=list(definition.options),
        description=DESCRIPTIONS["answer_options"],
        id=f"hearts-options-{directory.partition('/')[2]}",
    )
    for directory, definition in TASKS.items()
    if definition.options
}
"""The reusable answer vocabulary of each classification directory, registered once per dataset."""
