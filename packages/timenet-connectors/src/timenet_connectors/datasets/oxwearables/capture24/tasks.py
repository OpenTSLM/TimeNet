"""CAPTURE-24's one task: name a 10 s window's activity class in one label scheme, as the benchmark does."""

from timenet.dataset import Record
from timenet.types import ClassificationTask, Split, TimeInterval
from timenet_connectors.datasets.oxwearables.capture24.benchmark import WINDOW_US
from timenet_connectors.datasets.oxwearables.capture24.keys import scheme_vocabulary
from timenet_connectors.datasets.oxwearables.capture24.vocabulary import LabelScheme


def window_task(record: Record, split: Split, scheme: LabelScheme, start_us: int, target: str) -> ClassificationTask:
    """Return one window's class in one scheme, which the authors' ``benchmark.py`` classifies from the window's acceleration.

    The scope is the window's 1000 samples and names no signal, so it covers every signal of the Record.
    The window's samples are the x, y and z of the one segment that holds it, the ``X`` row ``make_windows``
    cuts. The rest of the Record stays available, since the authors' RNN reads 8 consecutive windows and
    their HMM a participant's whole sequence. The benchmark asks no question in words, so the task has no
    prompt.

    Args:
        record: The participant's Record.
        split: The participant's split in the authors' ``benchmark.py``.
        scheme: The label scheme, whose vocabulary key is the task's ``target_schema``.
        start_us: The window's first sample's time from the recording's first sample.
        target: The window's class in ``scheme``.

    Returns:
        The classification task, with the window's class as target.
    """
    return ClassificationTask(
        inputs=(record,),
        scope=TimeInterval.micros(start_us, start_us + WINDOW_US),
        targets=(target,),
        target_schema=scheme_vocabulary(scheme),
        split=split,
    )
