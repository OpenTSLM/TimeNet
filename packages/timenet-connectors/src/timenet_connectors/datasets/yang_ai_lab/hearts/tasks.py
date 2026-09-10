"""The task table: one row per released task directory that this connector converts.

Nine of the thirty released directories are out of scope and :data:`EXCLUDED` records why. The
twenty-one that remain map to the task type that matches the answer the reference harness scores: a
label to :class:`~timenet.types.ClassificationTask`, a number with a unit to
:class:`~timenet.types.ScalarPredictionTask`, a moment on the timeline to
:class:`~timenet.types.TemporalLocalizationTask`, and everything whose answer is a JSON object or a
list to :class:`~timenet.types.AnswerTask`.

The prompts are transcribed from ``exp/<source>/<task>.py`` in the release. Two kinds of edit are
made to the released text. The release points the agent at files in a sandbox directory that a TimeF
consumer does not have, so every such reference names the signal that carries the same values
instead. The release also interpolates per-case values into some prompts, and one prompt is shared
by every case of its directory, so those are dropped. This connector's README lists every edit.

A ``target_schema`` is a different thing from a task type. It must equal the id of the registered
annotation that holds the answer vocabulary, so :func:`name_vocabulary` builds both from one string.
"""

from dataclasses import dataclass, field
import json
from typing import Any

import numpy as np

from timenet.errors import TimeFFormatError
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    ScalarPredictionTask,
    Task,
    TemporalLocalizationTask,
    TimePoint,
)


EXCLUDED: dict[str, str] = {
    "cgmacros/meal_img_classification": "the payload is mostly JPEG photographs of participants' meals",
    "coswara/cough_covid_status_classification_symptoms_only": "the payload carries no time series",
    "cgmacros/meal_forecasting": "the answer is a series of values the input must not contain",
    "cgmacros/meal_forecasting_meal_info": "the answer is a series of values the input must not contain",
    "cgmacros/meal_forecasting_no_ref": "the answer is a series of values the input must not contain",
    "cgmacros/meal_forecasting_no_ref_meal_info": "the answer is a series of values the input must not contain",
    "cgmacros/non_meal_imputation_calories": "the answer is a series of values the input must not contain",
    "cgmacros/non_meal_imputation_cgm_only": "the answer is a series of values the input must not contain",
    "cgmacros/non_meal_imputation_hr": "the answer is a series of values the input must not contain",
}
"""Released task directories this connector does not convert, and the reason for each."""

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
    """The question, transcribed from the release with its file references rewritten to signals."""
    options: tuple[str, ...] = ()
    """The closed answer vocabulary, empty when the answer is free text or a number."""
    unit: str | None = None
    """The unit of a scalar answer."""
    target_name: str | None = None
    """The name of the quantity a scalar answer predicts."""
    input_keys: tuple[str, ...] = field(default=())
    """Payload keys the reference harness shows the agent besides the series."""

    @property
    def directory(self) -> str:
        """The directory this definition describes.

        Returns:
            The ``<source>/<task>`` path inside the release.
        """
        return f"{self.source}/{self.task}"


def name_vocabulary(id_prefix: str, task: str) -> str:
    """Build the id of one task directory's shared answer-options annotation.

    A ``ClassificationTask`` states this id in ``target_schema`` and references the annotation
    through ``input_annotation_ids``, so both ends read it from here.

    Args:
        id_prefix: The connector's id prefix.
        task: The task directory the vocabulary belongs to.

    Returns:
        The annotation id, stable across every test case in the directory.
    """
    return f"{id_prefix}-options-{task}"


_A1C_PROMPT = (
    "The continuous glucose monitors (CGM) data for this subject is in signal "
    "'window_df.Libre GL', in mg/dL. Your task is to predict the disease status based on CGM "
    "data. There are 3 different status: normal, prediabetes, and diabetes. Please analyze the "
    "entire CGM time series and output your final prediction of disease status as a JSON object "
    'without any other text in the following format:\n{\n    "disease_status": [string, '
    "normal|prediabetes|diabetes]\n}"
)
_CGM_STAT_PROMPT = (
    "The continuous glucose monitors (CGM) data for this subject is in signal 'cgm.Libre GL', "
    "in mg/dL. Calculate percentage of time CGM is below and above normal range (70 - 180 "
    "mg/dL). Please calculate and output your final answer as a JSON object without any other "
    'text in the following format:\n{\n    "below": [float, percentage of time CGM < 70 mg/dL],'
    '\n    "above": [float, percentage of time CGM > 180 mg/dL]\n}'
)
_FASTING_GLU_PROMPT = (
    "The continuous glucose monitors (CGM) data for this subject is in signal "
    "'window_df.Libre GL', in mg/dL. Your task is to predict the subject's fasting blood glucose "
    "value (in mg/dL) based on the entire CGM time series. Please output your final prediction as "
    'a JSON object without any other text in the following format:\n{\n    "fasting_glu": [float, '
    "fasting blood glucose value in mg/dL]\n}"
)
_IAUC_PROMPT = (
    "The CGM (continuous glucose monitoring) data for 2 hours after a meal is in signal "
    "'cgm_df.CGM (mg/dL)', in mg/dL, and the signal's time axis gives the minutes since the meal "
    "started.\nPlease calculate the incremental Area Under the Curve (iAUC) for the postprandial "
    "glucose response. Use first CGM value as baseline. Please output your calculated iAUC value "
    'as a JSON object without any other text in the following format:\n{\n    "iauc": [number]\n}'
)
_MEAL_REACT_PROMPT = (
    "You are given two CGM data windows (A and B), each is a 4-hour window (1 hour before and 3 "
    "hours after a meal) from two different subjects. Both subjects ate a meal with similar "
    "carbohydrate and calorie content, but one subject is normal (non-diabetes) and the other has "
    "prediabetes or diabetes. The windows are in signals 'A.Libre GL' and 'B.Libre GL', in "
    "mg/dL.\n\nYour task: Based on the CGM data, decide which window (A or B) is from the normal "
    "subject and which is from the prediabetes or diabetes subject. Output your answer as a JSON "
    'object with the following format (no extra text):\n{\n    "normal_subject": "A"  # or "B"\n}'
)
_MEAL_TIME_PROMPT = (
    "The continuous glucose monitors (CGM) data for this subject is in signal "
    "'window_df.Libre GL', in mg/dL, and the signal's time axis gives the time of each reading "
    "(in integer minutes). There is exactly one meal event in this 2-hour window. Please analyze "
    "the CGM data and output your final answer as a JSON object without any other text in the "
    "following format:\n"
    '{\n    "meal_timestamp": [float, timestamp (in minutes) when the meal starts]\n}'
)
_COSWARA_AUDIO_CLASS_PROMPT = (
    "Analyze the audio in signal 'data.signal'.\n\nClassify the audio as one of the following "
    "categories:\n- speech: spoken words or sounds like counting or vowels\n- cough: coughing "
    "sounds\n- breathing: breathing sounds (deep or shallow)\n\nOutput your classification in "
    'JSON format:\n{\n    "prediction": "speech" or "cough" or "breathing"\n}'
)
_COSWARA_COUGH_PROMPT = (
    "Analyze the cough audio in signal 'data.signal'.\n\nBased on the sound of the cough, "
    "classify the subject as either healthy or covid positive.\n\nOutput your classification in "
    'JSON format:\n{\n    "prediction": "healthy" or "covid_positive"\n}'
)
_COSWARA_COUGH_SYMPTOMS_PROMPT = (
    "Analyze the cough audio in signal 'data.signal' and the subject's symptoms to classify the "
    "subject as either healthy or covid positive.\n\nThe symptoms are in the 'symptoms' "
    "annotation.\n\nBased on the sound of the cough and the symptoms, classify the subject as "
    "either healthy or covid positive.\n\nOutput your classification in JSON format:\n{\n    "
    '"prediction": "healthy" or "covid_positive"\n}'
)
_COSWARA_SPEECH_PROMPT = (
    "Analyze the speech audio in signal 'data.signal'.\n\nBased on the sound of the speech, "
    "classify the subject as either healthy or covid positive.\n\nOutput your classification in "
    'JSON format:\n{\n    "prediction": "healthy" or "covid_positive"\n}'
)
_COUGHVID_DETECTION_PROMPT = (
    "One audio recording is in signal 'audio'. It is a cough audio sampled at 48 kHz. Analyze "
    "the audio and determine if a cough is present in the recording.\n\nOutput your answer in the "
    'following JSON format without any other text:\n{\n    "has_cough": true/false\n}'
)
_COUGHVID_COVID_PROMPT = (
    "One audio recording is in signal 'audio', which is sampled at 48 kHz. Analyze the audio and "
    "determine if the subject has COVID-19.\n\nOutput your answer in the following JSON format "
    'without any other text:\n{\n    "is_covid": true/false\n}'
)
_COUGHVID_DIAGNOSIS_PROMPT = (
    "One audio recording is in signal 'audio', which is sampled at 48 kHz. Analyze the audio and "
    "determine the most likely diagnosis from the following options:\n- upper_infection\n- "
    "lower_infection\n- obstructive_disease\n- COVID-19\n- healthy_cough\n\nOutput your answer in "
    'the following JSON format without any other text:\n{\n    "diagnosis": "diagnosis_option"\n}'
)
_COUGHVID_HEALTH_PROMPT = (
    "One audio recording is in signal 'audio', which is sampled at 48 kHz. Analyze the audio and "
    "determine if the subject is healthy.\n\nHealthy is defined as the subject having no "
    "underlying health conditions can be recognized from the audio.\n\nOutput your answer in the "
    'following JSON format without any other text:\n{\n    "is_healthy": true/false\n}'
)
_COUGHVID_MFCC_PROMPT = (
    "One audio recording is in signal 'audio'. It is a cough audio sampled at 48 kHz. Calculate "
    "the Mel Frequency Cepstral Coefficients (MFCCs) with 13 coefficients, then compute the mean "
    "and standard deviation of each MFCC over time. Please output your final answer in the "
    'following JSON format without any other text:\n{\n    "mfcc_mean": [list of 13 floats],\n    '
    '"mfcc_std": [list of 13 floats]\n}'
)
_ALTITUDE_RESPIRATION_PROMPT = (
    "You are given 5-minute segments of respiration signals in signals 'segment_dfs.A.rsp', "
    "'segment_dfs.B.rsp' and 'segment_dfs.C.rsp'. Each segment corresponds to one of the altitude "
    "ranges provided below:\n\n- 1.5k-2k meters\n- 2k-2.5k meters\n- 2.5k-3k meters\n- 3k-3.5k "
    "meters\n- 3.5k-4k meters\n\nYour task is to analyze the respiration signals and rank the "
    "segments from highest altitude to lowest altitude.\n\nOutput your ranking as a list of "
    'labels in JSON format:\n{\n    "ranking": ["label from highest altitude", "label from mid '
    'altitude", "label from lowest altitude"]\n}'
)
_ALTITUDE_SPO2_PROMPT = (
    "You are given 5-minute segments of SpO2 (oxygen saturation) signals in signals "
    "'segment_dfs.A.spo', 'segment_dfs.B.spo' and 'segment_dfs.C.spo'. Each segment corresponds "
    "to one of the altitude ranges provided below:\n\n- 1.5k-2k meters\n- 2k-2.5k meters\n- "
    "2.5k-3k meters\n- 3k-3.5k meters\n- 3.5k-4k meters\n\nYour task is to analyze the SpO2 "
    "signals and rank the segments from highest altitude to lowest altitude.\n\nOutput your "
    'ranking as a list of labels in JSON format:\n{\n    "ranking": ["label from highest '
    'altitude", "label from mid altitude", "label from lowest altitude"]\n}'
)
_HR_RESP_PAIRING_PROMPT = (
    "You are given 5-minute segments of respiration and heart rate signals from the same subject "
    "at different altitude phases.\n\nThe respiration signals are in signals "
    "'respiration_dfs.respiration_A.rsp' and 'respiration_dfs.respiration_B.rsp'.\n\nThe heart "
    "rate signals are in signals 'hr_dfs.hr_1.hr' and 'hr_dfs.hr_2.hr'.\n\nYour task is to "
    "analyze the signals and pair which heart rate signal corresponds to which respiration "
    "signal, based on them being from the same altitude phase. The pairing should reflect which "
    "heart rate matches which respiration from the same phase.\n\nOutput your pairing as a "
    "dictionary in JSON format without any other text, for example:\n{\n    "
    '"pairing": {"A": "1", "B": "2"} # or {"A": "2", "B": "1"}\n}'
)
_SPO2_RESP_PAIRING_PROMPT = (
    "You are given 5-minute segments of respiration and SpO2 (oxygen saturation) signals from the "
    "same subject at different altitude phases.\n\nThe respiration signals are in signals "
    "'respiration_dfs.respiration_A.rsp' and 'respiration_dfs.respiration_B.rsp'.\n\nThe SpO2 "
    "signals are in signals 'spo_dfs.spo_1.spo' and 'spo_dfs.spo_2.spo'.\n\nYour task is to "
    "analyze the signals and pair which SpO2 signal corresponds to which respiration signal, "
    "based on them being from the same altitude phase. The pairing should reflect which SpO2 "
    "matches which respiration from the same phase.\n\nOutput your pairing as a dictionary in "
    'JSON format without any other text, for example:\n{\n    "pairing": {"A": "1", "B": "2"} # '
    'or {"A": "2", "B": "1"}\n}'
)
_VCTK_DIRECTION_PROMPT = (
    "A raw audio waveform is in signal 'waveform'. This is a mono audio signal sampled at 16000 "
    "Hz. The waveform may be playing forward (normal) or time-reversed (backward). Determine "
    "whether the waveform is playing in the forward direction or has been time-reversed. Output "
    "your final answer as a JSON object without any other text, in the following format:\n{\n    "
    '"direction": "[forward|reversed]",\n    "reason": "[explanation of your choice]"\n}'
)


TASK_DEFINITIONS: tuple[TaskDefinition, ...] = (
    TaskDefinition(
        "cgmacros",
        "a1c_classification",
        ClassificationTask,
        43,
        _A1C_PROMPT,
        options=("normal", "prediabetes", "diabetes"),
    ),
    TaskDefinition("cgmacros", "cgm_stat_calculation", AnswerTask, 45, _CGM_STAT_PROMPT),
    TaskDefinition(
        "cgmacros",
        "fasting_glu_prediction",
        ScalarPredictionTask,
        45,
        _FASTING_GLU_PROMPT,
        unit="mg/dL",
        target_name="fasting_glucose",
    ),
    TaskDefinition(
        "cgmacros",
        "iauc_calculation",
        ScalarPredictionTask,
        50,
        _IAUC_PROMPT,
        unit="mg*min/dL",
        target_name="postprandial_iauc",
    ),
    TaskDefinition("cgmacros", "meal_react_comparison", ClassificationTask, 50, _MEAL_REACT_PROMPT, options=("A", "B")),
    TaskDefinition("cgmacros", "meal_time_localization", TemporalLocalizationTask, 50, _MEAL_TIME_PROMPT),
    TaskDefinition(
        "coswara",
        "audio_classification",
        ClassificationTask,
        50,
        _COSWARA_AUDIO_CLASS_PROMPT,
        options=("speech", "cough", "breathing"),
    ),
    TaskDefinition(
        "coswara",
        "cough_covid_status_classification",
        ClassificationTask,
        50,
        _COSWARA_COUGH_PROMPT,
        options=("healthy", "covid_positive"),
    ),
    TaskDefinition(
        "coswara",
        "cough_covid_status_classification_with_symptoms",
        ClassificationTask,
        50,
        _COSWARA_COUGH_SYMPTOMS_PROMPT,
        options=("healthy", "covid_positive"),
        input_keys=("symptoms",),
    ),
    TaskDefinition(
        "coswara",
        "speech_covid_status_classification",
        ClassificationTask,
        50,
        _COSWARA_SPEECH_PROMPT,
        options=("healthy", "covid_positive"),
    ),
    TaskDefinition(
        "coughvid",
        "cough_detection_good_qual",
        ClassificationTask,
        50,
        _COUGHVID_DETECTION_PROMPT,
        options=("true", "false"),
    ),
    TaskDefinition(
        "coughvid",
        "cough_detection_poor_qual",
        ClassificationTask,
        22,
        _COUGHVID_DETECTION_PROMPT,
        options=("true", "false"),
    ),
    TaskDefinition(
        "coughvid",
        "covid_status_classification",
        ClassificationTask,
        50,
        _COUGHVID_COVID_PROMPT,
        options=("true", "false"),
    ),
    TaskDefinition(
        "coughvid",
        "diagnosis_classification",
        ClassificationTask,
        50,
        _COUGHVID_DIAGNOSIS_PROMPT,
        options=("upper_infection", "lower_infection", "obstructive_disease", "COVID-19", "healthy_cough"),
    ),
    TaskDefinition(
        "coughvid",
        "health_status_classification",
        ClassificationTask,
        50,
        _COUGHVID_HEALTH_PROMPT,
        options=("true", "false"),
    ),
    TaskDefinition("coughvid", "mfcc_mean_std", AnswerTask, 50, _COUGHVID_MFCC_PROMPT),
    TaskDefinition("harespod", "altitude_ranking_respiration", AnswerTask, 50, _ALTITUDE_RESPIRATION_PROMPT),
    TaskDefinition("harespod", "altitude_ranking_spo2", AnswerTask, 50, _ALTITUDE_SPO2_PROMPT),
    TaskDefinition("harespod", "hr_resp_pairing", AnswerTask, 50, _HR_RESP_PAIRING_PROMPT),
    TaskDefinition("harespod", "spo2_resp_pairing", AnswerTask, 50, _SPO2_RESP_PAIRING_PROMPT),
    TaskDefinition(
        "vctk",
        "waveform_temporal_direction_detection",
        ClassificationTask,
        50,
        _VCTK_DIRECTION_PROMPT,
        options=("forward", "reversed"),
    ),
)
"""Every task directory this connector converts, in the order it walks them."""

TASK_TYPES: tuple[type[Task], ...] = tuple(dict.fromkeys(definition.task_type for definition in TASK_DEFINITIONS))
"""The task classes the stream yields, read off the table so the two cannot disagree."""


def option_annotations(id_prefix: str) -> tuple[Annotation, ...]:
    """Build the shared answer-vocabulary annotations the closed-answer tasks reference.

    Args:
        id_prefix: The connector's id prefix.

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
    """Return the label a classification test case is scored against.

    Args:
        definition: The task directory's definition.
        answer: The payload's ``GT`` value.

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
    """Rebuild one NumPy scalar as the Python value that serializes to the same number.

    A float keeps the shortest decimal text that reads back as the same value at its own dtype,
    which is the text ``str`` gives for a NumPy scalar. Widening a float32 with ``item`` instead
    would print the float32 representation error as digits the source never held.

    Args:
        value: A NumPy scalar read out of a payload.

    Returns:
        The Python equivalent of the scalar.
    """
    if isinstance(value, np.floating):
        return float(str(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value.item()


def plain(value: Any) -> Any:
    """Rebuild a node of a payload out of plain Python types.

    Every value that reaches the writer passes through here, an answer and an annotation alike. The
    writer serializes both with :func:`json.dumps`, which cannot encode a NumPy scalar as a number.

    Args:
        value: A node read out of a payload.

    Returns:
        The same value, with every NumPy scalar and array replaced by its Python equivalent.
    """
    if isinstance(value, np.generic):
        return _plain_scalar(value)
    if isinstance(value, np.ndarray):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [plain(item) for item in value]
    return value


def _answer_text(answer: Any) -> str:
    """Serialize a structured answer to the one string an answer task holds.

    NumPy scalars are rebuilt as Python numbers first. ``np.float32`` and ``np.int64`` are not
    instances of ``float`` or ``int``, so :func:`json.dumps` cannot encode them as numbers and
    would write each one as a quoted string instead. A number is written at the shortest text
    that reads back as the source value at its source dtype.

    Args:
        answer: The payload's ``GT`` value.

    Returns:
        Canonical JSON for a mapping or a sequence, the plain text of the value otherwise.
    """
    if isinstance(answer, dict | list | tuple):
        return json.dumps(plain(answer), sort_keys=True, default=str)
    return str(answer)


def build_task(
    definition: TaskDefinition,
    record_id: str,
    answer: Any,
    input_annotation_ids: tuple[str, ...],
    id_prefix: str,
) -> Task:
    """Build the task for one test case.

    The task is streamed rather than attached, so it carries its own ``record_ids``.

    Args:
        definition: The task directory's definition.
        record_id: The owning record's id.
        answer: The payload's ``GT`` value.
        input_annotation_ids: The annotations the reference harness shows its agent.
        id_prefix: The connector's id prefix.

    Returns:
        The task, typed by what the directory's answer is.

    Raises:
        TimeFFormatError: If the answer does not fit the task type.
    """
    task_id = f"{record_id}-qa"
    shared: dict[str, Any] = {
        "id": task_id,
        "prompt": definition.prompt,
        "input_annotation_ids": input_annotation_ids,
        "record_ids": (record_id,),
    }
    if definition.task_type is ClassificationTask:
        return ClassificationTask(
            target=_label(definition, answer),
            target_schema=name_vocabulary(id_prefix, definition.task),
            **shared,
        )
    if definition.task_type is ScalarPredictionTask:
        return ScalarPredictionTask(
            target=float(answer), unit=definition.unit, target_name=definition.target_name, **shared
        )
    if definition.task_type is TemporalLocalizationTask:
        # The release's own prompt declares this answer a float, so the minutes are carried to the
        # whole microsecond a TimePoint stores rather than read as an int.
        micros = round(float(answer) * _US_PER_MINUTE)
        if micros < 0:
            raise TimeFFormatError(f"{definition.directory} answer {answer!r} is a negative minute offset")
        return TemporalLocalizationTask(target=(TimePoint.micros(micros),), **shared)
    return AnswerTask(target=_answer_text(answer), **shared)
