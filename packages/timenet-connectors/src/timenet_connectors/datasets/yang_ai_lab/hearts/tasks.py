"""Turn a HEARTS case into its TimeF task, with its answer placed on the input record's clock."""

from collections.abc import Mapping
import json
from typing import Any

import numpy as np

from timenet.dataset import Record, TimeFDataset
from timenet.errors import TimeFFormatError
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    ForecastingTask,
    ScalarPredictionTask,
    Split,
    Task,
    TemporalLocalizationTask,
    TimeInterval,
    TimePoint,
    TSCorrespondenceTask,
    TSEditingTask,
)
from timenet_connectors.datasets.yang_ai_lab.hearts.records import (
    Built,
    Case,
    CaseRecords,
    answer_record,
    case_records,
    frame_times_us,
    moment_us,
)
from timenet_connectors.datasets.yang_ai_lab.hearts.release import (
    ANSWER_KEY,
    DESCRIPTIONS,
    HELD_OUT_READINGS,
    MEAL_COLUMNS,
    PROMPTS,
    SYMPTOM_NAMES,
    US_PER_MINUTE,
    VOCABULARIES,
    Answer,
    TaskDef,
)


def convert_case(  # noqa: PLR0911 - one return per task type the release holds
    dataset: TimeFDataset,
    case: Case,
    records: CaseRecords | None = None,
) -> Task:
    """Build a case's task from new or already imported input records.

    When ``records`` is omitted, this keeps the standalone behavior and registers newly constructed
    records. A composed connector supplies imported parent records and only task-owned answer records
    are added here.

    Returns:
        The task, not yet registered.

    Raises:
        TimeFFormatError: If a ranking answer does not order the case's records.
    """
    definition, answer = case.definition, case.payload[ANSWER_KEY]
    owns_records = records is None
    records = case_records(case) if records is None else records
    inputs = list(records.inputs)
    candidates = list(records.candidates)
    if owns_records:
        for built in (*inputs, *candidates):
            dataset.add_record(record=built.record)
    shared: dict[str, Any] = {
        "id": f"{case.id}-task",
        "prompt": _prompt(case),
        "inputs": tuple(built.record for built in inputs),
        "split": Split.TEST,
        "metadata": {"corpus": case.corpus, "task": case.task, "testcase_idx": case.index},
    }
    if definition.task_type is ForecastingTask:
        context, target = _forecast(case, inputs[0])
        dataset.add_record(record=target)
        return ForecastingTask(targets=(target,), input_annotations=context, **shared)
    if definition.task_type is TSEditingTask:
        mask, target = _imputation(case, inputs[0])
        dataset.add_record(record=target)
        return TSEditingTask(targets=(target,), input_annotations=(mask,), **shared)
    if definition.task_type is TSCorrespondenceTask:
        pool = tuple(built.record for built in candidates)
        return TSCorrespondenceTask(candidate_records=pool, targets=_matches(case, shared["inputs"], pool), **shared)
    if definition.task_type is ClassificationTask:
        return _classification(case, inputs, shared)
    if definition.task_type is ScalarPredictionTask:
        return ScalarPredictionTask(
            targets=_scalars(definition, answer), unit=definition.unit, target_name=definition.target_name, **shared
        )
    if definition.task_type is TemporalLocalizationTask:
        return TemporalLocalizationTask(targets=(TimePoint.micros(round(float(answer) * US_PER_MINUTE)),), **shared)
    names = sorted(record.id.rsplit("-", 1)[1] for record in shared["inputs"])
    if sorted(str(label) for label in answer) != names:
        raise TimeFFormatError(f"HEARTS {case.id} answer {answer!r} does not order the records {names}")
    return AnswerTask(targets=(json.dumps([str(label) for label in answer]),), **shared)


def _prompt(case: Case) -> str:
    """Fill the directory's prompt with the values the harness formatted in for this case.

    Returns:
        The prompt.
    """
    payload = case.payload
    values = {key: str(payload[key]) for key in ("meal_time", "mask_start", "mask_end") if key in payload}
    if "speaker_id" in payload:
        values["speaker"] = str(payload["speaker_id"])
    if case.definition.meal_info:
        values["meal_info"] = ", ".join(f"{key}: {value}" for key, value in payload["meal_info"].items())
    if case.definition.symptoms:
        values["symptoms"] = ", ".join(
            f"{SYMPTOM_NAMES[key]}: {'yes' if value else 'no'}" for key, value in payload["symptoms"].items()
        )
    return PROMPTS[case.directory].substitute(values)


def _classification(case: Case, inputs: list[Built], shared: Mapping[str, Any]) -> ClassificationTask:
    """Build a classification task, with the meal marker or symptom flags the case shows the model.

    Returns:
        The task. A case without a recording keeps its symptom flags on the task itself.
    """
    definition = case.definition
    context: list[Annotation] = []
    if definition.images:
        record, origin_us = inputs[0]
        context.append(record.annotate(_meal_marker(moment_us(case.payload["meal_time"]) - origin_us)))
    flags = _symptom_flags(case) if definition.symptoms else []
    if inputs:
        context.extend(inputs[0].record.add_annotations(flags))
    task = ClassificationTask(
        targets=(_label(definition, case.payload[ANSWER_KEY]),),
        target_schema=VOCABULARIES[case.directory].id,
        input_annotations=tuple(context),
        **shared,
    )
    if not inputs:
        for flag in flags:
            task.annotate(flag)
    return task


def _label(definition: TaskDef, answer: Any) -> str:
    """Spell a source answer as one of the directory's options.

    Returns:
        The label.

    Raises:
        TimeFFormatError: If the answer does not name an option.
    """
    if definition.answer is Answer.NORMAL_WINDOW:
        normal = [window for window, status in answer.items() if status == "Normal"]
        label = str(normal[0]) if len(normal) == 1 else repr(answer)
    elif definition.answer is Answer.BOOLEAN:
        if not isinstance(answer, bool | np.bool_):
            raise TimeFFormatError(f"HEARTS answer {answer!r} is not boolean")
        label = "true" if bool(answer) else "false"
    elif definition.answer is Answer.OPTION_INDEX:
        label = definition.options[int(answer)]
    else:
        label = str(answer)
    if label not in definition.options:
        raise TimeFFormatError(f"HEARTS answer {answer!r} is not one of {list(definition.options)}")
    return label


def _scalars(definition: TaskDef, answer: Any) -> tuple[float, ...]:
    """Flatten a numeric answer, a mapping of numbers or lists of numbers, in field order.

    Returns:
        The numeric targets.
    """
    parts = [answer[field] for field in definition.fields] if definition.fields else [answer]
    return tuple(float(value) for part in parts for value in (part if isinstance(part, list) else [part]))


def _symptom_flags(case: Case) -> list[Annotation]:
    """Build one boolean annotation per symptom, named as the harness spelt it out.

    Returns:
        The flags in source order.

    Raises:
        TimeFFormatError: If a symptom flag is not boolean.
    """
    symptoms = case.payload["symptoms"]
    invalid = {key: value for key, value in symptoms.items() if not isinstance(value, bool | np.bool_)}
    if invalid:
        raise TimeFFormatError(f"HEARTS {case.id} has non-boolean symptom flags {invalid}")
    return [Annotation(key=key, value=bool(value), description=SYMPTOM_NAMES[key]) for key, value in symptoms.items()]


def _matches(case: Case, inputs: tuple[Record, ...], pool: tuple[Record, ...]) -> tuple[Record, ...]:
    """Resolve a pairing answer, a map of input suffix to candidate suffix, to candidates in input order.

    Returns:
        The matched candidates.

    Raises:
        TimeFFormatError: If the answer does not pair every input with a candidate of its own.
    """
    answer = case.payload[ANSWER_KEY]
    candidates = {record.id.rsplit("_", 1)[1]: record for record in pool}
    try:
        matched = tuple(candidates[str(answer[record.id.rsplit("_", 1)[1]])] for record in inputs)
    except KeyError as exc:
        raise TimeFFormatError(f"HEARTS {case.id} answer {answer!r} names no candidate for every input") from exc
    if len({record.id for record in matched}) != len(matched):
        raise TimeFFormatError(f"HEARTS {case.id} answer {answer!r} pairs two inputs with one candidate")
    return matched


def _forecast(case: Case, built: Built) -> tuple[tuple[Annotation, ...], Record]:
    """Place the meals the model may see on the input's clock and build the held-out answer record.

    Returns:
        The task's context annotations, and the answer record.
    """
    payload, (record, origin_us) = case.payload, built
    meal_us = moment_us(payload["meal_time"]) - origin_us
    context: list[Annotation] = []
    if "reference_meal_info_df" in payload:
        for meal in payload["reference_meal_info_df"].to_dict("records"):
            context.extend(record.add_annotations(_meal_annotations(meal, moment_us(meal["Timestamp"]) - origin_us)))
    context.append(record.annotate(_meal_marker(meal_us)))
    if case.definition.meal_info:
        context.extend(record.add_annotations(_meal_annotations(payload["meal_info"], meal_us)))
    offsets = meal_us + np.arange(HELD_OUT_READINGS, dtype=np.int64) * US_PER_MINUTE
    return tuple(context), answer_record(case, built, offsets, _readings(case))


def _imputation(case: Case, built: Built) -> tuple[Annotation, Record]:
    """Mark the zeroed stretch of the input and build the record holding its true readings.

    Returns:
        The mask annotation, and the answer record.

    Raises:
        TimeFFormatError: If the mask indices and the mask bounds disagree.
    """
    payload, (record, origin_us) = case.payload, built
    frame = payload["window_df"]
    positions = frame.index.get_indexer(np.asarray(payload["mask_indices"]))
    times, _ = frame_times_us(frame, "Timestamp")
    bounds = (moment_us(payload["mask_start"]), moment_us(payload["mask_end"]))
    if bool((positions < 0).any()) or (int(times[positions[0]]), int(times[positions[-1]])) != bounds:
        raise TimeFFormatError(f"HEARTS {case.id} mask_indices do not name the window rows from mask_start to mask_end")
    offsets = times[positions] - origin_us
    mask = record.annotate(
        Annotation(
            key="cgm_mask",
            span=TimeInterval.micros(int(offsets[0]), int(offsets[-1]) + US_PER_MINUTE),
            description=DESCRIPTIONS["cgm_mask"],
        )
    )
    return mask, answer_record(case, built, offsets, _readings(case))


def _readings(case: Case) -> np.ndarray:
    """Read the thirty held-out readings of a forecast or imputation answer.

    Returns:
        The readings as float64.

    Raises:
        TimeFFormatError: If the answer does not hold exactly thirty readings.
    """
    answer = case.payload[ANSWER_KEY]
    if len(answer) != HELD_OUT_READINGS:
        raise TimeFFormatError(f"HEARTS {case.id} needs {HELD_OUT_READINGS} readings as its answer, got {len(answer)}")
    return np.asarray(answer, dtype=np.float64)


def _meal_marker(at_us: int) -> Annotation:
    """Build the marker for the meal a task asks about.

    Returns:
        The meal marker.
    """
    return Annotation(key="meal_time", span=TimePoint.micros(at_us), description=DESCRIPTIONS["meal_time"])


def _meal_annotations(meal: Mapping[str, Any], at_us: int) -> list[Annotation]:
    """Describe one meal at its position on a record's timeline.

    Returns:
        The meal's type and content, one annotation each.
    """
    point = TimePoint.micros(at_us)
    return [
        Annotation(key=key, value=str(meal[column]) if unit is None else float(meal[column]), unit=unit, span=point)
        for column, key, unit in MEAL_COLUMNS
    ]
