"""Map all thirty HEARTS task directories to TimeF tasks.

Each definition declares the prompt, input fields, and answer type. Classification
vocabularies use the same annotation ID in ``target_schema`` and ``input_annotations``.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
import json
from typing import Any

import numpy as np

from timenet.dataset import Record
from timenet.errors import TimeFFormatError
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    ForecastingTask,
    InputModality,
    ScalarPredictionTask,
    Task,
    TemporalLocalizationTask,
    TimePoint,
    TSEditingTask,
)
from timenet_connectors.datasets.yang_ai_lab.hearts import prompts
from timenet_connectors.datasets.yang_ai_lab.hearts.pickles import require_pandas


_VCTK_DIRECTIONS = ("forward", "reversed")
"""``waveform_temporal_direction_detection`` stores its answer as 0 or 1 and scores this wording."""
_NORMAL_STATUS = "Normal"
"""``meal_react_comparison`` asks which window is the non-diabetic subject's."""
_US_PER_MINUTE = 60_000_000

OPTIONS_KEY = "answer_options"
_OPTIONS_DESCRIPTION = "The closed set of answers the reference harness accepts for this task."


@dataclass(frozen=True)
class TaskDefinition:
    """One released task directory and how it becomes a TimeF task."""

    source: str
    """The corpus directory, such as ``harespod``."""
    task: str
    """The task directory inside it, such as ``hr_resp_pairing``."""
    task_type: type[Task]
    """The task class this directory's answer maps to."""
    n_items: int
    """How many test cases the pinned revision publishes here."""
    prompt: str
    """The prompt, with file references replaced by signal names."""
    options: tuple[str, ...] = ()
    """The closed answer vocabulary, empty when the answer is free text or a number."""
    unit: str | None = None
    """The unit of a scalar answer."""
    target_name: str | None = None
    """The name of the quantity a scalar answer predicts."""
    input_keys: tuple[str, ...] = field(default=())
    """Payload fields exposed as input annotations."""
    series_keys: tuple[str, ...] | None = None
    """Input series fields. None selects all fields except the answer."""
    no_record: bool = False
    """Whether the task has no input record."""
    images: bool = False
    """Whether the case supplies meal photographs as image signals."""

    @property
    def input_modalities(self) -> frozenset[InputModality]:
        """Return the declared input modalities.

        Returns:
            The declared input modalities, including the prompt text.
        """
        if self.no_record:
            return frozenset({InputModality.TEXT, InputModality.NO_INPUT})
        kinds = {
            InputModality.TEXT,
            InputModality.AUDIO if self.source in {"coswara", "coughvid", "vctk"} else InputModality.TIME_SERIES,
        }
        if self.images:
            kinds.add(InputModality.IMAGE)
        return frozenset(kinds)

    @property
    def directory(self) -> str:
        """Return the source task directory.

        Returns:
            The ``<source>/<task>`` path inside the release.
        """
        return f"{self.source}/{self.task}"


def name_vocabulary(id_prefix: str, task: str) -> str:
    """Return the shared answer-vocabulary annotation ID.

    Returns:
        The annotation id, stable across every test case in the directory.
    """
    return f"{id_prefix}-options-{task}"


TASK_DEFINITIONS: tuple[TaskDefinition, ...] = (
    TaskDefinition(
        "cgmacros",
        "a1c_classification",
        ClassificationTask,
        43,
        prompts.A1C_PROMPT,
        options=("normal", "prediabetes", "diabetes"),
    ),
    TaskDefinition("cgmacros", "cgm_stat_calculation", AnswerTask, 45, prompts.CGM_STAT_PROMPT),
    TaskDefinition(
        "cgmacros",
        "fasting_glu_prediction",
        ScalarPredictionTask,
        45,
        prompts.FASTING_GLU_PROMPT,
        unit="mg/dL",
        target_name="fasting_glucose",
    ),
    TaskDefinition(
        "cgmacros",
        "iauc_calculation",
        ScalarPredictionTask,
        50,
        prompts.IAUC_PROMPT,
        unit="mg*min/dL",
        target_name="postprandial_iauc",
    ),
    TaskDefinition(
        "cgmacros", "meal_react_comparison", ClassificationTask, 50, prompts.MEAL_REACT_PROMPT, options=("A", "B")
    ),
    TaskDefinition("cgmacros", "meal_time_localization", TemporalLocalizationTask, 50, prompts.MEAL_TIME_PROMPT),
    TaskDefinition(
        "cgmacros",
        "meal_forecasting",
        ForecastingTask,
        50,
        prompts.FORECAST_PROMPT,
        input_keys=("reference_meal_info_df", "meal_time"),
        series_keys=("reference_cgm_df", "window_df"),
    ),
    TaskDefinition(
        "cgmacros",
        "meal_forecasting_meal_info",
        ForecastingTask,
        50,
        prompts.FORECAST_PROMPT,
        input_keys=("reference_meal_info_df", "meal_info", "meal_time"),
        series_keys=("reference_cgm_df", "window_df"),
    ),
    TaskDefinition(
        "cgmacros",
        "meal_forecasting_no_ref",
        ForecastingTask,
        50,
        prompts.FORECAST_PROMPT,
        input_keys=("meal_time",),
        series_keys=("window_df",),
    ),
    TaskDefinition(
        "cgmacros",
        "meal_forecasting_no_ref_meal_info",
        ForecastingTask,
        50,
        prompts.FORECAST_PROMPT,
        input_keys=("meal_info", "meal_time"),
        series_keys=("window_df",),
    ),
    TaskDefinition(
        "cgmacros",
        "meal_img_classification",
        ClassificationTask,
        50,
        prompts.MEAL_IMAGE_PROMPT,
        options=("a.jpg", "b.jpg", "c.jpg", "d.jpg"),
        input_keys=("meal_time",),
        series_keys=("window_df",),
        images=True,
    ),
    TaskDefinition(
        "cgmacros",
        "non_meal_imputation_calories",
        TSEditingTask,
        50,
        prompts.IMPUTE_PROMPT,
        input_keys=("mask_start", "mask_end"),
        series_keys=("window_df",),
    ),
    TaskDefinition(
        "cgmacros",
        "non_meal_imputation_cgm_only",
        TSEditingTask,
        50,
        prompts.IMPUTE_PROMPT,
        input_keys=("mask_start", "mask_end"),
        series_keys=("window_df",),
    ),
    TaskDefinition(
        "cgmacros",
        "non_meal_imputation_hr",
        TSEditingTask,
        50,
        prompts.IMPUTE_PROMPT,
        input_keys=("mask_start", "mask_end"),
        series_keys=("window_df",),
    ),
    TaskDefinition(
        "coswara",
        "audio_classification",
        ClassificationTask,
        50,
        prompts.COSWARA_AUDIO_CLASS_PROMPT,
        options=("speech", "cough", "breathing"),
    ),
    TaskDefinition(
        "coswara",
        "cough_covid_status_classification",
        ClassificationTask,
        50,
        prompts.COSWARA_COUGH_PROMPT,
        options=("healthy", "covid_positive"),
    ),
    TaskDefinition(
        "coswara",
        "cough_covid_status_classification_with_symptoms",
        ClassificationTask,
        50,
        prompts.COSWARA_COUGH_SYMPTOMS_PROMPT,
        options=("healthy", "covid_positive"),
        input_keys=("symptoms",),
    ),
    TaskDefinition(
        "coswara",
        "cough_covid_status_classification_symptoms_only",
        ClassificationTask,
        50,
        prompts.SYMPTOMS_ONLY_PROMPT,
        options=("healthy", "covid_positive"),
        input_keys=("symptoms",),
        no_record=True,
    ),
    TaskDefinition(
        "coswara",
        "speech_covid_status_classification",
        ClassificationTask,
        50,
        prompts.COSWARA_SPEECH_PROMPT,
        options=("healthy", "covid_positive"),
    ),
    TaskDefinition(
        "coughvid",
        "cough_detection_good_qual",
        ClassificationTask,
        50,
        prompts.COUGHVID_DETECTION_PROMPT,
        options=("true", "false"),
    ),
    TaskDefinition(
        "coughvid",
        "cough_detection_poor_qual",
        ClassificationTask,
        22,
        prompts.COUGHVID_DETECTION_PROMPT,
        options=("true", "false"),
    ),
    TaskDefinition(
        "coughvid",
        "covid_status_classification",
        ClassificationTask,
        50,
        prompts.COUGHVID_COVID_PROMPT,
        options=("true", "false"),
    ),
    TaskDefinition(
        "coughvid",
        "diagnosis_classification",
        ClassificationTask,
        50,
        prompts.COUGHVID_DIAGNOSIS_PROMPT,
        options=("upper_infection", "lower_infection", "obstructive_disease", "COVID-19", "healthy_cough"),
    ),
    TaskDefinition(
        "coughvid",
        "health_status_classification",
        ClassificationTask,
        50,
        prompts.COUGHVID_HEALTH_PROMPT,
        options=("true", "false"),
    ),
    TaskDefinition("coughvid", "mfcc_mean_std", AnswerTask, 50, prompts.COUGHVID_MFCC_PROMPT),
    TaskDefinition("harespod", "altitude_ranking_respiration", AnswerTask, 50, prompts.ALTITUDE_RESPIRATION_PROMPT),
    TaskDefinition("harespod", "altitude_ranking_spo2", AnswerTask, 50, prompts.ALTITUDE_SPO2_PROMPT),
    TaskDefinition("harespod", "hr_resp_pairing", AnswerTask, 50, prompts.HR_RESP_PAIRING_PROMPT),
    TaskDefinition("harespod", "spo2_resp_pairing", AnswerTask, 50, prompts.SPO2_RESP_PAIRING_PROMPT),
    TaskDefinition(
        "vctk",
        "waveform_temporal_direction_detection",
        ClassificationTask,
        50,
        prompts.VCTK_DIRECTION_PROMPT,
        options=("forward", "reversed"),
    ),
)
"""Every task directory this connector converts, in the order it walks them."""

TASK_TYPES: tuple[type[Task], ...] = tuple(dict.fromkeys(definition.task_type for definition in TASK_DEFINITIONS))
"""Task classes declared in the table."""


def option_annotations(id_prefix: str) -> tuple[Annotation, ...]:
    """Build answer vocabularies for attachment to the dataset.

    Returns:
        One annotation per task directory that has a closed vocabulary.
    """
    return tuple(
        Annotation(
            key=OPTIONS_KEY,
            value=list(definition.options),
            description=_OPTIONS_DESCRIPTION,
            id=name_vocabulary(id_prefix, definition.task),
        )
        for definition in TASK_DEFINITIONS
        if definition.options
    )


def _label(definition: TaskDefinition, answer: Any) -> str:
    """Convert a classification answer to one of the declared options.

    Returns:
        The label, always one of the definition's options.

    Raises:
        TimeFFormatError: If the answer does not resolve to one of the options.
    """
    if definition.task == "waveform_temporal_direction_detection":
        index = int(answer)
        if not 0 <= index < len(_VCTK_DIRECTIONS):
            raise TimeFFormatError(f"{definition.directory} answer {answer!r} is not a direction index")
        label = _VCTK_DIRECTIONS[index]
    elif definition.task == "meal_react_comparison":
        normal = [window for window, status in dict(answer).items() if status == _NORMAL_STATUS]
        if len(normal) != 1:
            raise TimeFFormatError(
                f"meal_react_comparison expects exactly one {_NORMAL_STATUS!r} window, got {answer!r}"
            )
        label = str(normal[0])
    elif isinstance(answer, bool | np.bool_):
        label = "true" if answer else "false"
    else:
        label = str(answer)
    if label not in definition.options:
        raise TimeFFormatError(
            f"{definition.directory} answer {answer!r} maps to {label!r}, which is not one of "
            f"{list(definition.options)}"
        )
    return label


def _plain_scalar(value: np.generic) -> Any:
    """Convert a NumPy scalar to a Python value.

    Returns:
        The Python equivalent of the scalar.
    """
    return value.item()


def plain(value: Any) -> Any:  # noqa: PLR0911 - each source value shape has one conversion
    """Convert source values to JSON-compatible Python types.

    Returns:
        The same value, with every NumPy scalar and array replaced by its Python equivalent.

    Raises:
        TimeFFormatError: If the payload contains a type this connector cannot serialize.
    """
    if isinstance(value, np.generic):
        return _plain_scalar(value)
    if isinstance(value, require_pandas().DataFrame):
        return [plain(row) for row in value.to_dict(orient="records")]
    if isinstance(value, date | datetime):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [plain(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TimeFFormatError(f"HEARTS payload holds unsupported annotation value {type(value).__name__}")


def annotation_value(value: Any) -> str | int | float | bool | list[str]:
    """Convert a source value to an annotation value.

    Structured values become JSON strings because annotations cannot store mappings or mixed lists.

    Returns:
        The value an annotation stores.
    """
    rebuilt = plain(value)
    if isinstance(rebuilt, str | int | float | bool):
        return rebuilt
    if isinstance(rebuilt, list) and all(isinstance(item, str) for item in rebuilt):
        return rebuilt
    return json.dumps(rebuilt, sort_keys=True)


def _answer_text(answer: Any) -> str:
    """Serialize structured answers as JSON and scalar answers as text.

    Returns:
        Canonical JSON for a mapping or a sequence, the plain text of the value otherwise.
    """
    if isinstance(answer, dict | list | tuple):
        return json.dumps(plain(answer), sort_keys=True)
    return str(answer)


def build_task(  # noqa: PLR0913, PLR0917 - source case data and target must be explicit
    definition: TaskDefinition,
    record: Record | None,
    answer: Any,
    input_annotations: tuple[Annotation, ...],
    id_prefix: str,
    case_id: str,
    target_record: Record | None = None,
) -> Task:
    """Build a task with explicit input and target references.

    Returns:
        The task, typed by what the directory's answer is.

    Raises:
        TimeFFormatError: If the answer does not fit the task type.
    """
    task_id = f"{case_id}-qa"
    shared: dict[str, Any] = {
        "id": task_id,
        "prompt": definition.prompt,
        "input_annotations": input_annotations,
        "inputs": () if record is None else (record,),
        "input_modalities": definition.input_modalities,
    }
    if definition.task_type in {ForecastingTask, TSEditingTask}:
        if target_record is None:
            raise TimeFFormatError(f"{definition.directory} needs a held-out target record")
        return definition.task_type(targets=(target_record,), **shared)
    if definition.task_type is ClassificationTask:
        return ClassificationTask(
            targets=(_label(definition, answer),),
            target_schema=name_vocabulary(id_prefix, definition.task),
            **shared,
        )
    if definition.task_type is ScalarPredictionTask:
        return ScalarPredictionTask(
            targets=(float(answer),), unit=definition.unit, target_name=definition.target_name, **shared
        )
    if definition.task_type is TemporalLocalizationTask:
        # Preserve fractional minutes when converting to whole microseconds.
        micros = round(float(answer) * _US_PER_MINUTE)
        if micros < 0:
            raise TimeFFormatError(f"{definition.directory} answer {answer!r} is a negative minute offset")
        return TemporalLocalizationTask(targets=(TimePoint.micros(micros),), **shared)
    return AnswerTask(targets=(_answer_text(answer),), **shared)
