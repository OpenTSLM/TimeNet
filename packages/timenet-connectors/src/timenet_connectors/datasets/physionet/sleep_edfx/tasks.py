"""Build the tasks of one recording from the annotations it already carries.

Nothing here opens a file. ``connector.py`` passes the annotations, and every function takes
values and gives tasks.

Every task takes a named id. The default is a fresh uuid, so two builds of one archive give two
sets of ids that nothing can compare. An epoch task is named for its position from the start of
the recording, so a correction to one entry leaves every later id alone.

No task carries a prompt. The release states no question in words.
"""

from timenet.types import Annotation
from timenet_connectors.datasets.physionet.sleep_edfx.keys import AnnotationKey, Condition, Question


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
