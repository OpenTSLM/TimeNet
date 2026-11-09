"""Build the tasks of each recording from the annotations its record already carries.

Nothing here reads a file. No task carries a prompt, because the release states no question in
words.
"""

from collections.abc import Iterator, Sequence

from timenet.dataset import Record
from timenet.errors import TimeNetConnectorError
from timenet.types import (
    US_PER_S,
    Annotation,
    ClassificationTask,
    LocalizationMode,
    ScalarPredictionTask,
    Task,
    TemporalLocalizationTask,
    TimeInterval,
)
from timenet.types.text import read_value
from timenet_connectors.datasets.physionet.sleep_edfx.labels import (
    ASLEEP_LABELS,
    CONDITION_LABELS,
    CONDITIONS,
    SEX_LABELS,
    SLEEP_STAGES,
    STAGE_LABELS,
    AnnotationKey,
    Question,
)
from timenet_connectors.demographics import AGE_UNIT, SEXES


# The epoch the release scores in, from the 1968 Rechtschaffen and Kales manual.
EPOCH_MICROSECONDS = 30 * US_PER_S


def build_epoch_tasks(recording_id: str, span_annotations: Sequence[Annotation]) -> Iterator[ClassificationTask]:
    """Give one sleep-stage task for each scored 30 s epoch.

    A scoring entry covers a run of equal epochs, so each entry is walked in 30 s steps. Unscored
    time gets no task.

    Yields:
        One task for each epoch, in entry order.

    Raises:
        TimeNetConnectorError: If an entry has no span, an unknown label, or does not divide into
            whole epochs.
    """
    for stage in span_annotations:
        label = read_value(stage.value)
        if stage.span is None:
            raise TimeNetConnectorError(
                f"{recording_id}: a scored entry of {label!r} states no span, so it holds no epoch"
            )

        if not isinstance(label, str) or label not in STAGE_LABELS:
            raise TimeNetConnectorError(
                f"{recording_id}: the scored entry at {stage.span.start_us} us states {label!r}, "
                f"which is not a label this release writes"
            )

        start = stage.span.start_us
        length = stage.span.exclusive_end - start
        # The onset and the duration fail for different reasons, so each gets its own message.
        if start % EPOCH_MICROSECONDS:
            raise TimeNetConnectorError(
                f"{recording_id}: a scored entry starts at {start} us, which is not a boundary of "
                f"the {EPOCH_MICROSECONDS} us epoch"
            )

        if length % EPOCH_MICROSECONDS:
            raise TimeNetConnectorError(
                f"{recording_id}: a scored entry lasts {length} us, which is not a whole number of "
                f"{EPOCH_MICROSECONDS} us epochs"
            )

        for onset in range(start, stage.span.exclusive_end, EPOCH_MICROSECONDS):
            yield ClassificationTask(
                # The label the technician wrote. Merging stage 3 with stage 4 is for whoever trains.
                targets=(label,),
                label_set=SLEEP_STAGES,
                scope=TimeInterval.micros(onset, onset + EPOCH_MICROSECONDS),
            )


def build_record_tasks(
    recording_id: str,
    non_span_annotations: Sequence[Annotation],
    span_annotations: Sequence[Annotation],
) -> Iterator[Task]:
    """Give the questions about a whole recording: age, sex, condition, and the sleep period.

    The study, the night, the start clock, and the note get no task, because the dataset states
    them beside the record.

    Yields:
        The whole-record tasks, in a stable order.

    Raises:
        TimeNetConnectorError: If a fact is missing, stated twice, or has a value the release does
            not write.
    """
    stated: dict[str, object] = {}
    for one in non_span_annotations:
        if one.key in stated:
            raise TimeNetConnectorError(f"{recording_id}: states {one.key!r} twice, so it has no one answer for it")

        stated[one.key] = read_value(one.value)

    yield ScalarPredictionTask(
        # A float with a unit keeps the type that a regression metric needs.
        targets=(_whole_years(recording_id, stated),),
        unit=AGE_UNIT,
        target_name=Question.AGE,
    )

    yield ClassificationTask(
        targets=(_one_of(recording_id, stated, AnnotationKey.SEX, SEX_LABELS),),
        label_set=SEXES,
    )

    # Only the telemetry sheet states a condition.
    if AnnotationKey.CONDITION in stated:
        yield ClassificationTask(
            targets=(_one_of(recording_id, stated, AnnotationKey.CONDITION, CONDITION_LABELS),),
            label_set=CONDITIONS,
        )

    night = find_sleep_period(span_annotations)
    if night is not None:
        yield TemporalLocalizationTask(
            targets=(night,),
            # The wake time around the night is unmarked, not another class.
            mode=LocalizationMode.SPARSE,
        )


def find_sleep_period(span_annotations: Sequence[Annotation]) -> TimeInterval | None:
    """Give the interval from the first scored sleep epoch to the end of the last.

    Returns:
        The interval, or ``None`` where the scoring holds no sleep.
    """
    spans = [one.span for one in span_annotations if read_value(one.value) in ASLEEP_LABELS and one.span is not None]
    if not spans:
        return None

    start = min(span.start_us for span in spans)
    end = max(span.exclusive_end for span in spans)

    return TimeInterval.micros(start, end)


def _whole_years(recording_id: str, stated: dict[str, object]) -> float:
    if AnnotationKey.AGE not in stated:
        raise TimeNetConnectorError(f"{recording_id}: states no age, so it cannot be asked for one")

    age = stated[AnnotationKey.AGE]
    if not isinstance(age, int) or isinstance(age, bool):
        raise TimeNetConnectorError(f"{recording_id}: states an age of {age!r}, which is not a whole number of years")

    return float(age)


def _one_of(recording_id: str, stated: dict[str, object], key: AnnotationKey, permitted: Sequence[str]) -> str:
    if key not in stated:
        raise TimeNetConnectorError(f"{recording_id}: states no {key}, so it cannot be asked for one")

    value = stated[key]
    if value not in permitted:
        raise TimeNetConnectorError(f"{recording_id}: states a {key} of {value!r}, which the release does not write")

    return str(value)


def _sleep_stages(record: Record) -> list[Annotation]:
    """Give each sleep stage of a record once, although every scored signal carries it.

    Returns:
        The stages, in the order the signals first carry them.
    """
    stages: dict[tuple[object, object], Annotation] = {}
    for signal in record.signals:
        for one in signal.annotations:
            if one.key == AnnotationKey.SLEEP_STAGE:
                stages[one.span, read_value(one.value)] = one
    return list(stages.values())


def iter_tasks(records: Sequence[Record]) -> Iterator[Task]:
    """Give every task of a build, one recording at a time.

    This reads no file. It expands the annotations that the records already carry.

    Yields:
        The epoch tasks of each recording, then its whole-record tasks.

    Raises:
        TimeNetConnectorError: If a scoring cannot be expanded into whole epochs.
    """  # ruff: ignore[docstring-extraneous-exception] (raised by build_epoch_tasks, not directly here)
    for record in records:
        span_annotations = [one for one in record.annotations if one.span is not None]
        non_span_annotations = [one for one in record.annotations if one.span is None]
        stages = _sleep_stages(record)

        recording_id = str(record.metadata["recording_id"])
        for task in build_epoch_tasks(recording_id, stages):
            task.inputs = (record,)
            yield task
        for task in build_record_tasks(recording_id, non_span_annotations, stages):
            task.inputs = (record,)
            yield task
        moment = build_lights_off_task(recording_id, record.time_span, span_annotations)
        if moment is not None:
            moment.inputs = (record,)
            yield moment


def build_lights_off_task(
    recording_id: str,  # ruff: ignore[unused-function-argument] - keeps builder signatures and error context consistent
    session: TimeInterval | None,
    span_annotations: Sequence[Annotation],
) -> TemporalLocalizationTask | None:
    """Ask where the lights went out, when the recording holds that moment.

    Three recordings start seconds after the lights went out. The offset then wraps to about a day
    later, past the end of the session, and those recordings get no task.

    Returns:
        The task, or ``None`` where the recording states no lights-off moment or does not hold it.
    """
    stated = [one for one in span_annotations if one.key == AnnotationKey.LIGHTS_OFF]
    if not stated or session is None:
        return None

    at = stated[0].span
    if at is None or at.start_us < session.start_us or at.exclusive_end > session.exclusive_end:
        return None

    return TemporalLocalizationTask(
        targets=(at,),
        mode=LocalizationMode.SPARSE,
    )
