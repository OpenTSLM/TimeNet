"""Build the tasks of one recording from the annotations it already carries.

Nothing here opens a file. ``connector.py`` passes the annotations, and every function takes
values and gives tasks.

Every task takes a named id. The default is a fresh uuid, so two builds of one archive give two
sets of ids that nothing can compare. An epoch task is named for its position from the start of
the recording, so a correction to one entry leaves every later id alone.

No task carries a prompt. The release states no question in words.
"""

from collections.abc import Iterator, Sequence

from timenet.dataset import Sample
from timenet.errors import TimeFFormatError
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
from timenet_connectors.datasets.physionet.sleep_edfx.keys import AnnotationKey, Condition, Question


# The epoch the release scores in, from the 1968 Rechtschaffen and Kales manual.
EPOCH_MICROSECONDS = 30 * US_PER_S


# The labels the release writes for a scored epoch. Five name a stage of sleep. `Sleep stage W`
# is wake, `Movement time` marks a disturbed epoch, and `Sleep stage ?` marks one the
# technician left unscored.
STAGE_LABELS = (
    "Sleep stage W",
    "Sleep stage 1",
    "Sleep stage 2",
    "Sleep stage 3",
    "Sleep stage 4",
    "Sleep stage R",
    "Movement time",
    "Sleep stage ?",
)

# The five labels that say the subject was asleep. None of the other three bounds a night.
ASLEEP_LABELS = frozenset(STAGE_LABELS[1:6])

# The sexes, after each sheet is decoded. The two sheets code the column with opposite
# meanings, so a raw code never reaches a task.
SEX_LABELS = ("F", "M")

# The two nights of the telemetry study.
CONDITION_LABELS = tuple(Condition)

# The name of each closed set. A task states one of these in ``target_schema``. The annotation
# that holds the set uses the same string as its id, so one lookup resolves it.
_VOCABULARIES = (
    (AnnotationKey.SLEEP_STAGE, STAGE_LABELS),
    (AnnotationKey.SEX, SEX_LABELS),
    (AnnotationKey.CONDITION, CONDITION_LABELS),
)


def name_vocabulary(id_prefix: str, key: AnnotationKey) -> str:
    """Give the id of the annotation that holds one closed set.

    Args:
        id_prefix: The prefix every id of this connector carries.
        key: The key whose values the set holds.

    Returns:
        The id, which a task also states as its ``target_schema``.
    """
    return f"{id_prefix}-vocabulary-{key}"


def build_vocabularies(id_prefix: str) -> list[Annotation]:
    """Give one annotation for each closed set a task draws its target from.

    These belong to no sample, so the caller registers them rather than attaching them.

    A set is one annotation and not one for each member. One annotation for each member states
    that the values exist without stating that they are the whole set.

    Args:
        id_prefix: The prefix every id of this connector carries.

    Returns:
        One annotation for each set, in a stable order.
    """
    return [
        Annotation(
            # Each set takes its own key. The schema holds one descriptor for each key, so
            # three sets under one key are three descriptions of the same thing, which
            # ``derive_schema`` refuses.
            key=f"{key}_vocabulary",
            value=list(labels),
            description=f"The values a {key!r} target takes, as the release writes them.",
            id=name_vocabulary(id_prefix, key),
        )
        for key, labels in _VOCABULARIES
    ]


def name_epoch_task(sample_id: str, epoch_index: int) -> str:
    """Give the id of the task that asks about one epoch.

    Args:
        sample_id: The sample the epoch belongs to.
        epoch_index: The epoch's position, counted in 30 s steps from the start of the
            recording.

    Returns:
        The task id.
    """
    return f"{sample_id}-epoch-{epoch_index}"


def name_sample_task(sample_id: str, question: Question) -> str:
    """Give the id of a task that asks about a whole sample.

    Args:
        sample_id: The sample the question is about.
        question: What it asks.

    Returns:
        The task id.
    """
    return f"{sample_id}-{question}"


def build_epoch_tasks(
    sample_id: str, id_prefix: str, span_annotations: Sequence[Annotation]
) -> Iterator[ClassificationTask]:
    """Give one classification task for each epoch a scoring labels.

    The hypnogram stores a run of equal epochs as one entry, so an entry covers one epoch or
    hundreds. Each entry is walked in steps of 30 s.

    The scoring does not tile its recording. Some start after the signals do, and a few hold a
    hole in the middle. Unscored time gets no task, and nothing here fills it in.

    A scope covers its epoch and names no series, so a model can read any channel.

    Args:
        sample_id: The sample the scoring belongs to.
        id_prefix: The prefix every id of this connector carries.
        span_annotations: Its sleep-stage annotations, from ``annotations.build``.

    Yields:
        One task for each epoch, in the order the entries state them.

    Raises:
        TimeFFormatError: If an entry states no span, states a label the release does not
            write, or does not divide into whole epochs. Each message names the recording and
            the value that is wrong.
    """
    schema = name_vocabulary(id_prefix, AnnotationKey.SLEEP_STAGE)
    for stage in span_annotations:
        if stage.span is None:
            raise TimeFFormatError(f"{sample_id}: the scored entry {stage.id!r} states no span, so it holds no epoch")

        if stage.value not in STAGE_LABELS:
            raise TimeFFormatError(
                f"{sample_id}: the scored entry {stage.id!r} states {stage.value!r}, "
                f"which is not a label this release writes"
            )

        start = stage.span.start_us
        length = stage.span.exclusive_end - start
        # The onset and the duration fail for different reasons. One message for both names
        # the wrong number and sends a reader the wrong way.
        if start % EPOCH_MICROSECONDS:
            raise TimeFFormatError(
                f"{sample_id}: a scored entry starts at {start} us, which is not a boundary of "
                f"the {EPOCH_MICROSECONDS} us epoch"
            )

        if length % EPOCH_MICROSECONDS:
            raise TimeFFormatError(
                f"{sample_id}: a scored entry lasts {length} us, which is not a whole number of "
                f"{EPOCH_MICROSECONDS} us epochs"
            )

        for onset in range(start, stage.span.exclusive_end, EPOCH_MICROSECONDS):
            yield ClassificationTask(
                # The label the technician wrote. Merging stage 3 with stage 4 is a decision
                # for whoever trains.
                target=stage.value,
                target_schema=schema,
                scope=TimeInterval.micros(onset, onset + EPOCH_MICROSECONDS),
                sample_ids=(sample_id,),
                id=name_epoch_task(sample_id, onset // EPOCH_MICROSECONDS),
            )


def build_sample_tasks(
    sample_id: str,
    id_prefix: str,
    non_span_annotations: Sequence[Annotation],
    span_annotations: Sequence[Annotation],
) -> Iterator[Task]:
    """Give the questions about a whole recording rather than a region of one.

    Each takes an unset ``scope``. Age, sex and the drug condition come from the subject table.
    Where the subject slept comes from the scoring.

    Provenance is left out. The study, the night, the header clock and the demographics note
    carry no span either, so the caller passes those in too. A task for one of them asks a
    model to recover a fact the dataset states beside it.

    Args:
        sample_id: The sample these questions are about.
        id_prefix: The prefix every id of this connector carries.
        non_span_annotations: Its recording-metadata annotations, from ``SubjectTables``.
        span_annotations: Its sleep-stage annotations, which bound the night.

    Yields:
        The whole-sample tasks of that recording, in a stable order.

    Raises:
        TimeFFormatError: If a fact this asks about is missing, stated twice, or states a value
            the release does not write.
    """  # noqa: DOC502 (raised by the readers below, not directly here)
    stated = _state_of(sample_id, non_span_annotations)
    yield ScalarPredictionTask(
        # A float with a unit keeps the type a regression metric needs. As a string, a
        # one-year error reads as two unequal labels.
        target=_whole_years(sample_id, stated),
        unit="year",
        target_name=Question.AGE,
        sample_ids=(sample_id,),
        id=name_sample_task(sample_id, Question.AGE),
    )

    yield ClassificationTask(
        # The decoded letter, never the sheet's code. The two sheets code the column with
        # opposite meanings, so one code means two opposite things.
        target=_one_of(sample_id, stated, AnnotationKey.SEX, SEX_LABELS),
        target_schema=name_vocabulary(id_prefix, AnnotationKey.SEX),
        sample_ids=(sample_id,),
        id=name_sample_task(sample_id, Question.SEX),
    )

    # Only the telemetry sheet states a condition, so this one is optional.
    if AnnotationKey.CONDITION in stated:
        yield ClassificationTask(
            target=_one_of(sample_id, stated, AnnotationKey.CONDITION, CONDITION_LABELS),
            target_schema=name_vocabulary(id_prefix, AnnotationKey.CONDITION),
            sample_ids=(sample_id,),
            id=name_sample_task(sample_id, Question.CONDITION),
        )

    night = find_sleep_period(span_annotations)
    if night is not None:
        yield TemporalLocalizationTask(
            target=(night,),
            # The interval does not tile the recording. A cassette recording is mostly wake on
            # either side of one night, and that time is unmarked rather than something else.
            mode=LocalizationMode.SPARSE,
            sample_ids=(sample_id,),
            id=name_sample_task(sample_id, Question.SLEEP_PERIOD),
        )


def find_sleep_period(span_annotations: Sequence[Annotation]) -> TimeInterval | None:
    """Give the interval from the first scored epoch that is not wake to the end of the last.

    ``Sleep stage W``, ``Movement time`` and ``Sleep stage ?`` do not bound it, because none of
    the three says the subject was asleep.

    Args:
        span_annotations: The sleep-stage annotations of one recording.

    Returns:
        The interval, or ``None`` where the scoring holds no sleep. TimeF refuses an interval of
        zero length, so a recording with no sleep carries no question about its night.
    """
    spans = [one.span for one in span_annotations if one.value in ASLEEP_LABELS and one.span is not None]
    if not spans:
        return None

    start = min(span.start_us for span in spans)
    end = max(span.exclusive_end for span in spans)
    if end <= start:
        return None

    return TimeInterval.micros(start, end)


def _state_of(sample_id: str, non_span_annotations: Sequence[Annotation]) -> dict[str, object]:
    """Read the metadata annotations of one recording into a lookup by key.

    Args:
        sample_id: The sample they belong to, for the error message.
        non_span_annotations: Its metadata annotations.

    Returns:
        The value each key states.

    Raises:
        TimeFFormatError: If one key is stated twice. Keeping the last answers the question
            with whichever fact came second.
    """
    stated: dict[str, object] = {}
    for one in non_span_annotations:
        if one.key in stated:
            raise TimeFFormatError(f"{sample_id}: states {one.key!r} twice, so it has no one answer for it")

        stated[one.key] = one.value
    return stated


def _whole_years(sample_id: str, stated: dict[str, object]) -> float:
    """Give the subject's age as a float, from the value the subject table stated.

    ``Annotation.value`` is untyped, and the sheet decoder two modules away is what makes an
    age a whole number. This states it again where the task is built.

    Args:
        sample_id: The sample the age belongs to, for the error message.
        stated: What its metadata annotations state.

    Returns:
        The age in whole years.

    Raises:
        TimeFFormatError: If no age is stated, or the value is not a whole number.
    """
    if AnnotationKey.AGE not in stated:
        raise TimeFFormatError(f"{sample_id}: states no age, so it cannot be asked for one")

    age = stated[AnnotationKey.AGE]
    if not isinstance(age, int) or isinstance(age, bool):
        raise TimeFFormatError(f"{sample_id}: states an age of {age!r}, which is not a whole number of years")

    return float(age)


def _one_of(sample_id: str, stated: dict[str, object], key: AnnotationKey, permitted: Sequence[str]) -> str:
    """Give one stated value, after a check that the release writes it.

    Args:
        sample_id: The sample it belongs to, for the error message.
        stated: What its metadata annotations state.
        key: The key to read.
        permitted: The values the release writes for that key.

    Returns:
        The value.

    Raises:
        TimeFFormatError: If the key is absent, or its value is outside the set.
    """
    if key not in stated:
        raise TimeFFormatError(f"{sample_id}: states no {key}, so it cannot be asked for one")

    value = stated[key]
    if value not in permitted:
        raise TimeFFormatError(f"{sample_id}: states a {key} of {value!r}, which the release does not write")

    return str(value)


def iter_tasks(samples: Sequence[Sample], id_prefix: str) -> Iterator[Task]:
    """Give every task of a build, one recording at a time.

    This reads no file. ``convert`` already read each scoring, and a second read of a
    hypnogram can disagree with the annotations the samples carry.

    The writer calls the source more than once, so the caller passes a callable that gives a
    fresh iterator each time.

    Args:
        samples: The samples of the build, each carrying its annotations.
        id_prefix: The prefix every id of this connector carries.

    Yields:
        The epoch tasks of each recording, then its whole-sample tasks.

    Raises:
        TimeFFormatError: If a scoring cannot be expanded into whole epochs. The message names
            the recording, because a task is one of hundreds of thousands.
    """  # noqa: DOC502 (raised by build_epoch_tasks, not directly here)
    for sample in samples:
        span_annotations = [one for one in sample.annotations if one.key == AnnotationKey.SLEEP_STAGE]
        # A whole-sample question comes from an annotation with no span. A test on the key
        # instead holds only while the keys hold. ``lights_off`` states a fact about the
        # recording and carries a span, so a key test hands it over.
        non_span_annotations = [one for one in sample.annotations if one.span is None]
        yield from build_epoch_tasks(sample.sample_id, id_prefix, span_annotations)
        yield from build_sample_tasks(sample.sample_id, id_prefix, non_span_annotations, span_annotations)
