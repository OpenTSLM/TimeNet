"""Seizure intervals of one EEG recording, and the tasks that ask for them.

An EEG seizure database marks where each seizure starts and ends within a recording. How it
states that differs from release to release: CHB-MIT writes seconds from the start of each file
into a summary text file, and Siena writes clock times into a list of its own. Each connector
parses its own format into :class:`RecordingSeizures`. This module then builds the same tasks for
all of them, so two seizure datasets ask the same questions in the same way.

Two tasks come from one recording:

- :meth:`RecordingSeizures.create_localization_task` asks where the seizures are. Its mode is
  exhaustive: the release marks every seizure of a recording, so unmarked time holds none. A
  recording with no seizure gets an empty target, which states that answer explicitly.
- :meth:`RecordingSeizures.iter_window_tasks` asks, for each window of a fixed length, whether a
  seizure occurs in it. A window counts as a seizure window when it overlaps a marked seizure by
  any amount, because a consumer that wants a stricter rule can compute it from the localization
  target, and a looser rule cannot be recovered from a stricter label.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

from timenet.dataset import Record
from timenet.errors import TimeFFormatError
from timenet.types import (
    US_PER_S,
    Annotation,
    ClassificationTask,
    Label,
    LabelSet,
    LocalizationMode,
    Split,
    TemporalLocalizationTask,
    TimeInterval,
)


# The annotation key of a marked seizure, and the name of the label set that classifies windows.
SEIZURE_KEY = "seizure"


class WindowLabel(StrEnum):
    """Whether a window overlaps a marked seizure. The value is the label a task targets."""

    SEIZURE = "seizure"
    NO_SEIZURE = "no seizure"


SEIZURE_WINDOW_LABELS = LabelSet(
    [
        Label(name=WindowLabel.SEIZURE.value, description="The window overlaps a marked seizure by any amount."),
        Label(name=WindowLabel.NO_SEIZURE.value, description="The window overlaps no marked seizure."),
    ],
    annotations=[Annotation("name", value=SEIZURE_KEY)],
)


@dataclass(frozen=True)
class SeizureInterval:
    """One marked seizure: the half-open range ``[start_us, end_us)`` from the recording start."""

    start_us: int
    end_us: int

    def __post_init__(self) -> None:
        if self.start_us < 0 or self.end_us <= self.start_us:
            raise TimeFFormatError(
                f"a seizure from {self.start_us} us to {self.end_us} us does not start at or after the "
                f"recording start and end after it starts"
            )

    @classmethod
    def from_whole_seconds(cls, start_s: int, end_s: int) -> "SeizureInterval":
        """Build the interval that a release states in whole seconds from the recording start.

        Args:
            start_s: The onset, in seconds.
            end_s: The end, in seconds, exclusive.

        Returns:
            The interval in microseconds.
        """
        return cls(start_us=start_s * US_PER_S, end_us=end_s * US_PER_S)

    def overlaps(self, start_us: int, end_us: int) -> bool:
        """Say whether this seizure shares any time with the half-open range ``[start_us, end_us)``.

        Returns:
            True when the two ranges share at least one microsecond.
        """
        return self.start_us < end_us and start_us < self.end_us

    def to_time_interval(self) -> TimeInterval:
        """Give the interval as a TimeF span over the whole record, naming no signal.

        Returns:
            The span.
        """
        return TimeInterval.micros(self.start_us, self.end_us)


@dataclass(frozen=True)
class RecordingSeizures:
    """Every seizure that a release marks in one continuous recording.

    ``seizures`` holds every marked seizure in onset order, and none of them overlap. An empty
    tuple states that the recording holds no seizure. The recording must be continuous: an
    exhaustive answer over a recording with a gap would claim that the gap held no seizure.
    """

    recording_id: str
    duration_us: int
    seizures: tuple[SeizureInterval, ...]

    def __post_init__(self) -> None:
        if self.duration_us <= 0:
            raise TimeFFormatError(f"{self.recording_id}: a recording of {self.duration_us} us holds no time")

        previous_end = 0
        for seizure in self.seizures:
            if seizure.start_us < previous_end:
                raise TimeFFormatError(
                    f"{self.recording_id}: the seizure at {seizure.start_us} us starts before the one before it "
                    f"ends at {previous_end} us"
                )

            if seizure.end_us > self.duration_us:
                raise TimeFFormatError(
                    f"{self.recording_id}: the seizure that ends at {seizure.end_us} us ends after the recording, "
                    f"which lasts {self.duration_us} us"
                )

            previous_end = seizure.end_us

    def create_annotations(self) -> list[Annotation]:
        """Give one whole-record annotation for each marked seizure.

        Returns:
            The annotations, in onset order.
        """
        return [Annotation(SEIZURE_KEY, span=seizure.to_time_interval()) for seizure in self.seizures]

    def create_localization_task(self, record: Record, *, split: Split | None) -> TemporalLocalizationTask:
        """Ask where the seizures of this recording are.

        Args:
            record: The record that holds this recording.
            split: The partition of the task, or ``None`` when the dataset assigns none.

        Returns:
            An exhaustive task whose target holds every seizure, or nothing when there is none.
        """
        return TemporalLocalizationTask(
            inputs=[record],
            targets=tuple(seizure.to_time_interval() for seizure in self.seizures),
            mode=LocalizationMode.EXHAUSTIVE,
            split=split,
        )

    def iter_window_tasks(self, record: Record, *, window_us: int, split: Split | None) -> Iterator[ClassificationTask]:
        """Ask, for each window, whether it overlaps a seizure.

        The windows do not overlap and start at the recording start. A last stretch shorter than
        one window gets no task, because a shorter window would ask a different question.

        A scope covers its window and names no signal, so a model can read any channel.

        Args:
            record: The record that holds this recording.
            window_us: The window length, in microseconds.
            split: The partition of the tasks, or ``None`` when the dataset assigns none.

        Yields:
            One task for each whole window, in time order.

        Raises:
            TimeFFormatError: If the window length is not positive.
        """
        if window_us <= 0:
            raise TimeFFormatError(f"a window of {window_us} us holds no time")

        for start_us in range(0, self.duration_us - window_us + 1, window_us):
            end_us = start_us + window_us
            yield ClassificationTask(
                inputs=[record],
                targets=self.label_window(start_us, end_us).value,
                label_set=SEIZURE_WINDOW_LABELS,
                scope=TimeInterval.micros(start_us, end_us),
                split=split,
            )

    def label_window(self, start_us: int, end_us: int) -> WindowLabel:
        """Give the label of the half-open window ``[start_us, end_us)``.

        Returns:
            ``WindowLabel.SEIZURE`` when the window overlaps any marked seizure.
        """
        if any(seizure.overlaps(start_us, end_us) for seizure in self.seizures):
            return WindowLabel.SEIZURE

        return WindowLabel.NO_SEIZURE
