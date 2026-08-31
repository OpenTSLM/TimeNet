"""Build the tasks of one recording from the annotations it already carries.

Nothing here opens a file. ``connector.py`` passes the annotations, and every function takes
values and gives tasks.

No task names its own id. Nothing resolves one by id: the writer fills ``Sample.task_ids`` from
the task itself, and no task here states a lineage. Every task takes the generated id.

A ``target_schema`` is a different thing. It must equal the id of the registered annotation that
holds the set, so ``name_vocabulary`` builds both from one string.

No task carries a prompt. The release states no question in words.
"""

from collections.abc import Iterator, Sequence

from timenet.errors import TimeFFormatError
from timenet.types import US_PER_S, Annotation, ClassificationTask, TimeInterval
from timenet_connectors.datasets.physionet.sleep_edfx.keys import AnnotationKey, Condition


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
            key=f"{key}_vocabulary",
            value=list(labels),
            description=f"The values a {key!r} target takes, as the release writes them.",
            id=name_vocabulary(id_prefix, key),
        )
        for key, labels in _VOCABULARIES
    ]


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
            )
