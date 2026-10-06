"""Convert CGMacros benchmark answers, meals, and masks into tasks."""

from collections.abc import Mapping
from typing import Annotated, Any

import numpy as np
from pydantic import Field, FiniteFloat, TypeAdapter

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import IrregularAxis
from timenet.errors import TimeFFormatError
from timenet.types import (
    Annotation,
    ClassificationTask,
    ForecastingTask,
    ScalarPredictionTask,
    Task,
    TemporalLocalizationTask,
    TimeInterval,
    TimePoint,
    TSEditingTask,
)
from timenet_connectors.datasets.yang_ai_lab.hearts_cgmacros.release import (
    CGM,
    DESCRIPTIONS,
    HELD_OUT_READINGS,
    MEAL_COLUMNS,
    PROMPTS,
    UNITS,
    VOCABULARIES,
)
from timenet_connectors.datasets.yang_ai_lab.hearts_core.cases import Case, frame_times_us, moment_us
from timenet_connectors.datasets.yang_ai_lab.hearts_core.records import Built
from timenet_connectors.datasets.yang_ai_lab.hearts_core.release import ANSWER_KEY, US_PER_MINUTE
from timenet_connectors.datasets.yang_ai_lab.hearts_core.tasks import classification_task, scalar_task, task_fields
from timenet_connectors.time_axes import axis_for_offsets


_READINGS_ADAPTER = TypeAdapter(
    Annotated[tuple[FiniteFloat, ...], Field(min_length=HELD_OUT_READINGS, max_length=HELD_OUT_READINGS)]
)


def convert_case(dataset: TimeFDataset, case: Case, records: tuple[Built, ...]) -> Task:
    """Build a CGMacros task and register any held-out answer record.

    Returns:
        The task, not yet registered.
    """
    inputs = tuple(built.record for built in records)
    prompt = _prompt(case)
    definition = case.definition
    if definition.task_type is ForecastingTask:
        context, target = _forecast(case, records[0])
        dataset.add_record(record=target)
        return ForecastingTask(targets=(target,), input_annotations=context, **task_fields(case, inputs, prompt))
    if definition.task_type is TSEditingTask:
        mask, target = _imputation(case, records[0])
        dataset.add_record(record=target)
        return TSEditingTask(targets=(target,), input_annotations=(mask,), **task_fields(case, inputs, prompt))
    if definition.task_type is ScalarPredictionTask:
        return scalar_task(case, inputs, prompt, unit=UNITS[case.task])
    if definition.task_type is TemporalLocalizationTask:
        return TemporalLocalizationTask(
            targets=(TimePoint.micros(round(float(case.payload[ANSWER_KEY]) * US_PER_MINUTE)),),
            **task_fields(case, inputs, prompt),
        )
    return _classification(case, records, prompt)


def _prompt(case: Case) -> str:
    """Fill the meal and mask placeholders.

    Returns:
        The case's prompt.
    """
    payload = case.payload
    values = {key: str(payload[key]) for key in ("meal_time", "mask_start", "mask_end") if key in payload}
    if case.task.endswith("_meal_info"):
        values["meal_info"] = ", ".join(f"{key}: {value}" for key, value in payload["meal_info"].items())
    return PROMPTS[case.directory].substitute(values)


def _classification(case: Case, records: tuple[Built, ...], prompt: str) -> ClassificationTask:
    """Resolve comparison answers and attach the meal marker for image tasks.

    Returns:
        The classification task.
    """
    answer = case.payload[ANSWER_KEY]
    if case.task == "meal_react_comparison":
        normal = [window for window, status in answer.items() if status == "Normal"]
        label = str(normal[0]) if len(normal) == 1 else repr(answer)
    else:
        label = str(answer)
    inputs = tuple(built.record for built in records)
    task = classification_task(case, inputs, prompt, VOCABULARIES[case.directory], label=label)
    if case.task == "meal_img_classification":
        record, origin_us = records[0]
        marker = record.annotate(
            Annotation(
                key="meal_time",
                span=TimePoint.micros(moment_us(case.payload["meal_time"]) - origin_us),
                description=DESCRIPTIONS["meal_time"],
            )
        )
        task.input_annotations = (marker,)
    return task


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
    context.append(
        record.annotate(
            Annotation(key="meal_time", span=TimePoint.micros(meal_us), description=DESCRIPTIONS["meal_time"])
        )
    )
    if case.task.endswith("_meal_info"):
        context.extend(record.add_annotations(_meal_annotations(payload["meal_info"], meal_us)))
    offsets = meal_us + np.arange(HELD_OUT_READINGS, dtype=np.int64) * US_PER_MINUTE
    return tuple(context), _answer_record(case, built, offsets, _readings(case))


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
    return mask, _answer_record(case, built, offsets, _readings(case))


def _readings(case: Case) -> np.ndarray:
    """Read the thirty held-out readings of a forecast or imputation answer.

    Returns:
        The readings as float64.

    """
    return np.asarray(_READINGS_ADAPTER.validate_python(case.payload[ANSWER_KEY]), dtype=np.float64)


def _answer_record(case: Case, built: Built, offsets: np.ndarray, values: np.ndarray) -> Record:
    """Build the child-owned answer Record for a forecast or imputation task.

    Returns:
        The answer record on the imported input's clock.
    """
    record_id = f"{case.id}-GT"
    ids = {"id": f"{record_id}-Libre GL", "source_id": record_id}
    axis = axis_for_offsets(offsets)
    if isinstance(axis, IrregularAxis):
        signal = Signal.from_irregular(values, time_offsets_us=offsets, spec=CGM, name="Libre GL", **ids)
    else:
        signal = Signal.from_values(values, spec=CGM, name="Libre GL", time_axis=axis, **ids)
    record = Record(
        record_id=record_id,
        start_time=built.record.start_time,
        sources=(Source(id=f"{record_id}-source", name="HEARTS held-out answer", signals=(signal,)),),
        metadata=dict(built.record.metadata),
    )
    record.add_annotations(
        annotation
        for annotation in built.record.annotations
        if annotation.key in {"recording_start_local", "subject_id"}
    )
    return record


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
