"""Build the CAPTURE-24 benchmark tasks: one classification task per kept window and benchmarked scheme.

The tasks follow the authors' benchmark.py. It runs one benchmark per scheme over every participant's
windows, trains on P001 to P100, and tests on P101 to P151.
"""

from collections.abc import Callable, Iterator, Sequence

from timenet.dataset import Record
from timenet.types import ClassificationTask, Split, TimeInterval
from timenet_connectors.datasets.oxwearables.capture24.schemes import LABEL_SETS, Scheme
from timenet_connectors.datasets.oxwearables.capture24.source import ParticipantMetadata, RecordingSummary
from timenet_connectors.datasets.oxwearables.capture24.windowing import WINDOW_US


_FIRST_TEST_PARTICIPANT = 101
"""The paper's test set is P101 to P151, and benchmark.py tests ``range(101, 152)``."""


def _participant_split(participant: ParticipantMetadata) -> Split:
    """Return the participant's split in the authors' benchmark.py."""
    return Split.TEST if int(participant.pid.removeprefix("P")) >= _FIRST_TEST_PARTICIPANT else Split.TRAIN


def _scheme_tasks(scheme: Scheme, summary: RecordingSummary, record: Record) -> Iterator[ClassificationTask]:
    """Yield one participant's tasks in one scheme: one per kept window, in time order.

    Yields:
        A task whose scope is the window and whose target is the window's class in the scheme.
    """
    label_set = LABEL_SETS[scheme]
    split = _participant_split(summary.participant)
    class_names = label_set.names
    starts_us = summary.window_starts_us.tolist()
    classes = summary.window_classes[scheme].tolist()
    # Leave input_annotations empty: this benchmark classifies activity from acceleration alone.
    # Participant ID, age band, sex, and anonymized start time stay as Record annotations;
    # raw activity labels and their mappings stay on each Source. Neither is selected as model context.
    # Activity annotations reveal the target. An empty input_annotations does not remove annotations
    # from the input Record, so adapters must pass only explicitly selected annotations to the model.
    for start_us, class_index in zip(starts_us, classes, strict=True):
        yield ClassificationTask(
            inputs=(record,),
            scope=TimeInterval.micros(start_us, start_us + WINDOW_US),
            targets=(class_names[class_index],),
            label_set=label_set,
            split=split,
        )


def task_stream(
    summaries: Sequence[RecordingSummary], records: Sequence[Record]
) -> Callable[[], Iterator[ClassificationTask]]:
    """Return a function that starts a new pass over every task.

    The tasks use only the first-pass summary, so each pass gives the same tasks in the same order
    without reading the recordings again.

    Args:
        summaries: Each recording summary from the first pass.
        records: Each participant's Record, in the same order as ``summaries``.

    Returns:
        A function for ``set_task_stream``. Each call yields every scheme's tasks, scheme by scheme,
        with every participant's windows in participant order.
    """

    def tasks() -> Iterator[ClassificationTask]:
        for scheme in LABEL_SETS:
            for summary, record in zip(summaries, records, strict=True):
                yield from _scheme_tasks(scheme, summary, record)

    return tasks
