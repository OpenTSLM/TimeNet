"""CAPTURE-24's one task: name a 10 s window's activity class in one label scheme, as the benchmark does."""

from dataclasses import dataclass
from typing import ClassVar

from timenet.dataset import Record
from timenet.types import ClassificationTask, Split, Task, TimeInterval
from timenet_connectors.bases.loader import DatasetTask
from timenet_connectors.datasets.oxwearables.capture24.benchmark import WINDOW_US
from timenet_connectors.datasets.oxwearables.capture24.keys import scheme_vocabulary
from timenet_connectors.datasets.oxwearables.capture24.vocabulary import LabelScheme


@dataclass(frozen=True)
class Capture24WindowTask(DatasetTask):
    """One window's class in one scheme, which the authors' ``benchmark.py`` classifies from the window's acceleration.

    The scope is the window's 1000 samples and names no signal, so it covers every signal of the Record.
    The window's samples are the x, y and z of the one segment that holds it, the ``X`` row ``make_windows``
    cuts. The rest of the Record stays available, since the authors' RNN reads 8 consecutive windows and
    their HMM a participant's whole sequence. The benchmark asks no question in words, so the task has no
    prompt.
    """

    TIMENET_TYPE: ClassVar[type[Task]] = ClassificationTask

    record: Record
    participant_split: Split
    scheme: LabelScheme
    start_us: int
    """The window's first sample's time from the recording's first sample."""
    target: str

    @property
    def split(self) -> Split:
        """Return the task's split: its participant's split in the authors' ``benchmark.py``."""
        return self.participant_split

    def to_timenet_task(self) -> ClassificationTask:
        """Return the classification task, with the window's class as target."""
        return ClassificationTask(
            inputs=(self.record,),
            scope=TimeInterval.micros(self.start_us, self.start_us + WINDOW_US),
            targets=(self.target,),
            target_schema=scheme_vocabulary(self.scheme),
            split=self.split,
        )
