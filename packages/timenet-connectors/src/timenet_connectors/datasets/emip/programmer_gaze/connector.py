"""The EMIP connector: programmer eye fixations while reading code, labelled with programming expertise.

EMIP (Bednarik et al., 2020) recorded 216 programmers with an SMI RED250 eye tracker at 250 Hz while
each read two short programs, ``rectangle`` and ``vehicle``, in Java, Python or Scala. The release
holds raw gaze samples, one TSV file per participant, and a questionnaire file with the self-reported
programming expertise of each participant.

Each code-reading trial becomes one Record of four irregular signals, one value per fixation: the
gaze x and y position on the 1920x1080 screen, the fixation duration, and the mean pupil diameter.
Each Record gets one ClassificationTask whose target is the participant's self-reported
``expertise_programming`` (none, low, medium or high).

Fixations come from a dispersion-threshold (I-DT) pass over the right-eye samples, as in the
EMIP-Toolkit (Al Madi et al., 2021): a fixation is a run of valid samples whose dispersion
``(max x - min x) + (max y - min y)`` stays within 25 px and that lasts at least 80 ms. A sample is
valid when the tracker marks the right eye valid, the point of regard lies on the screen, and the
pupil diameter is positive. The tracker reports a lost eye (a blink, a look away) as a "valid"
sample at (0, 0) with a zero pupil, so the last two checks drop those. An invalid sample ends the
running fixation, so no fixation spans a blink.
"""

from collections.abc import Iterator, Mapping
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.errors import TimeNetConnectorError
from timenet.types import Annotation, ClassificationTask, Label, LabelSet, TimeSeriesSpec, ureg
from timenet_connectors.download import ensure_archive


EMIP_URL = "https://osf.io/download/3vh28/"
"""``emip_dataset.zip`` in the OSF project https://osf.io/53kts (released September 27, 2018)."""
EMIP_SHA256 = "716c035b6b03d374f2e2531d1744926d0c95b683fa5c312a02d53f5e6a509968"
RELEASE_DIR = "emip_dataset"
METADATA_FILE = "emip_metadata.csv"

SCREEN_WIDTH_PX = 1920.0
SCREEN_HEIGHT_PX = 1080.0
SAMPLE_PERIOD_US = 4_000
"""The tracker samples at 250 Hz."""
MAX_DISPERSION_PX = 25.0
MIN_FIXATION_US = 80_000

# Columns of a sample (SMP) line of the SMI export. A message (MSG) line holds its text in column 3.
_TIME, _TYPE = 0, 1
_MIN_CELLS = 3
"""Time, type and trial: the cells that every data line has."""
_MESSAGE = 3
_R_PUPIL_MM = 12
_R_POR_X, _R_POR_Y = 23, 24
_R_VALIDITY = 27

TASKS = ("rectangle", "vehicle")
"""The two comprehension programs. Their stimulus images are named ``<task>_<language>[2].jpg``."""

EXPERTISE = LabelSet(
    name="programming_expertise",
    labels=(
        Label(name="none", description="The participant reported no programming expertise."),
        Label(name="low", description="The participant reported low programming expertise."),
        Label(name="medium", description="The participant reported medium programming expertise."),
        Label(name="high", description="The participant reported high programming expertise."),
    ),
)

GAZE_X = TimeSeriesSpec(spec_type="gaze_x", name="Gaze X", unit_value=ureg.pixel, dtype="float64")
GAZE_Y = TimeSeriesSpec(spec_type="gaze_y", name="Gaze Y", unit_value=ureg.pixel, dtype="float64")
FIXATION_DURATION = TimeSeriesSpec(
    spec_type="fixation_duration",
    name="Fixation duration",
    unit_value=ureg.millisecond,
    dtype="float64",
)
PUPIL_DIAMETER = TimeSeriesSpec(
    spec_type="pupil_diameter",
    name="Pupil diameter",
    unit_value=ureg.millimeter,
    dtype="float64",
)


@dataclass(frozen=True)
class Fixations:
    """The fixations of one trial, as parallel arrays in time order."""

    onset_us: np.ndarray
    """Fixation onset in microseconds since the stimulus appeared."""
    x_px: np.ndarray
    y_px: np.ndarray
    duration_ms: np.ndarray
    pupil_mm: np.ndarray

    def __len__(self) -> int:
        return len(self.onset_us)


def detect_fixations(samples: list[tuple[int, float, float, float, bool]], onset_us: int) -> Fixations:
    """Group gaze samples into fixations with the I-DT algorithm.

    Args:
        samples: ``(time_us, x_px, y_px, pupil_mm, valid)`` per sample, in time order.
        onset_us: The time the stimulus appeared, the zero of the returned onsets.

    Returns:
        The fixations of the samples.
    """
    rows: list[tuple[int, float, float, float, float]] = []
    window: list[tuple[int, float, float, float]] = []

    def flush() -> None:
        if not window:
            return
        duration_us = window[-1][0] - window[0][0] + SAMPLE_PERIOD_US
        if duration_us >= MIN_FIXATION_US:
            points = np.array(window, dtype=np.float64)
            rows.append(
                (
                    window[0][0] - onset_us,
                    float(points[:, 1].mean()),
                    float(points[:, 2].mean()),
                    duration_us / 1000.0,
                    float(points[:, 3].mean()),
                )
            )

    min_x = max_x = min_y = max_y = 0.0
    for time_us, x, y, pupil, valid in samples:
        if not valid:
            flush()
            window = []
            continue
        if window:
            new_min_x, new_max_x = min(min_x, x), max(max_x, x)
            new_min_y, new_max_y = min(min_y, y), max(max_y, y)
            if (new_max_x - new_min_x) + (new_max_y - new_min_y) <= MAX_DISPERSION_PX:
                window.append((time_us, x, y, pupil))
                min_x, max_x, min_y, max_y = new_min_x, new_max_x, new_min_y, new_max_y
                continue
            flush()
        window = [(time_us, x, y, pupil)]
        min_x = max_x = x
        min_y = max_y = y
    flush()

    table = np.array(rows, dtype=np.float64).reshape(-1, 5)
    return Fixations(
        onset_us=table[:, 0].astype(np.int64),
        x_px=table[:, 1],
        y_px=table[:, 2],
        duration_ms=table[:, 3],
        pupil_mm=table[:, 4],
    )


def _task_of(stimulus: str) -> str | None:
    """Return the comprehension task a stimulus image shows, or ``None`` for any other screen."""
    stem = stimulus.removesuffix(".jpg")
    task, _, _ = stem.partition("_")
    return task if stimulus.endswith(".jpg") and task in TASKS and "_" in stem else None


def read_trials(path: Path) -> Iterator[tuple[str, Fixations]]:
    """Read the code-reading trials of one participant's raw gaze file.

    A trial runs from the message that shows a ``rectangle_*`` or ``vehicle_*`` stimulus image to the
    next message that shows any image. Mouse-click messages do not end a trial.

    Args:
        path: The participant's ``<id>_rawdata.tsv`` file.

    Yields:
        The stimulus name (such as ``rectangle_java2``) and the fixations of each trial, in file order.

    Raises:
        TimeNetConnectorError: If a sample line is malformed.
    """
    stimulus: str | None = None
    onset_us = 0
    samples: list[tuple[int, float, float, float, bool]] = []
    with path.open(encoding="latin-1", newline="") as stream:
        for number, line in enumerate(stream, start=1):
            if line.startswith(("#", "Time")):
                continue
            cells = line.rstrip("\r\n").split("\t")
            if len(cells) < _MIN_CELLS:
                continue
            if cells[_TYPE] == "MSG":
                image = cells[_MESSAGE].split()[-1] if len(cells) > _MESSAGE else ""
                if not image.endswith(".jpg"):
                    continue
                if stimulus is not None:
                    yield stimulus, detect_fixations(samples, onset_us)
                stimulus = image.removesuffix(".jpg") if _task_of(image) else None
                onset_us = int(cells[_TIME])
                samples = []
            elif cells[_TYPE] == "SMP" and stimulus is not None:
                try:
                    time_us = int(cells[_TIME])
                    x, y = float(cells[_R_POR_X]), float(cells[_R_POR_Y])
                    pupil = float(cells[_R_PUPIL_MM])
                except (IndexError, ValueError) as exc:
                    raise TimeNetConnectorError(f"{path.name} line {number}: malformed sample") from exc
                valid = (
                    cells[_R_VALIDITY] != "-1"
                    and 0.0 < x <= SCREEN_WIDTH_PX
                    and 0.0 < y <= SCREEN_HEIGHT_PX
                    and pupil > 0.0
                )
                samples.append((time_us, x, y, pupil, valid))
    if stimulus is not None:
        yield stimulus, detect_fixations(samples, onset_us)


@dataclass(frozen=True)
class Participant:
    """One participant: the raw gaze file and the questionnaire answers."""

    participant_id: int
    gaze_file: Path
    answers: Mapping[str, str]

    @property
    def expertise(self) -> str:
        """The self-reported programming expertise, a label of :data:`EXPERTISE`.

        Raises:
            TimeNetConnectorError: If the answer is not a label of the label set.
        """
        value = self.answers.get("expertise_programming", "")
        if value not in EXPERTISE.names:
            raise TimeNetConnectorError(
                f"participant {self.participant_id}: expertise_programming {value!r} is not one of {EXPERTISE.names}"
            )
        return value

    def create_record(self, stimulus: str, fixations: Fixations) -> Record:
        """Create the Record of one trial, with one irregular Signal per fixation feature.

        Returns:
            The Record, not yet added to a dataset.
        """
        offsets = fixations.onset_us
        signals = (
            Signal.from_irregular(fixations.x_px, time_offsets_us=offsets, spec=GAZE_X, name="x"),
            Signal.from_irregular(fixations.y_px, time_offsets_us=offsets, spec=GAZE_Y, name="y"),
            Signal.from_irregular(
                fixations.duration_ms, time_offsets_us=offsets, spec=FIXATION_DURATION, name="duration"
            ),
            Signal.from_irregular(fixations.pupil_mm, time_offsets_us=offsets, spec=PUPIL_DIAMETER, name="pupil"),
        )
        task = _task_of(f"{stimulus}.jpg") or ""
        return Record(
            sources=(Source(name="Fixations", signals=signals),),
            metadata={
                "participant_id": self.participant_id,
                "stimulus": stimulus,
                "task_id": task,
                "experiment_language": self.answers.get("experiment_language", ""),
            },
            annotations=(
                Annotation(key="expertise_programming", value=self.expertise),
                Annotation(key="expertise_experiment_language", value=self.answers["expertise_experiment_language"]),
                Annotation(key="task", value=task),
                Annotation(key="trial_order", value=int(self.answers[f"order_{task}"])),
                Annotation(key="answer_correct", value=self.answers[f"correct_{task}"] == "1"),
            ),
        )

    def create_classification_task(self, record: Record) -> ClassificationTask:
        """Create the task that asks for the participant's programming expertise.

        Returns:
            The task, not yet added to a dataset.
        """
        return ClassificationTask(inputs=(record,), targets=(self.expertise,), label_set=EXPERTISE)


def read_participants(root: Path) -> list[Participant]:
    """Pair each raw gaze file of the release with its questionnaire row.

    Args:
        root: The extracted ``emip_dataset`` directory.

    Returns:
        The participants in id order.

    Raises:
        TimeNetConnectorError: If a gaze file has no questionnaire row, or the reverse.
    """
    with (root / METADATA_FILE).open(encoding="utf-8-sig", newline="") as stream:
        answers = {int(row["id"]): row for row in csv.DictReader(stream)}
    files = {int(path.name.split("_")[0]): path for path in (root / "rawdata").glob("*_rawdata.tsv")}
    if answers.keys() != files.keys():
        raise TimeNetConnectorError(
            f"gaze files and questionnaire rows differ: {sorted(answers.keys() ^ files.keys())}"
        )
    return [Participant(participant_id=key, gaze_file=files[key], answers=answers[key]) for key in sorted(files)]


@dataclass(frozen=True)
class EmipRelease:
    """The extracted EMIP release."""

    root: Path

    @classmethod
    def at(cls, extracted: Path) -> Self:
        """Locate the release directory inside an extraction directory.

        Returns:
            The release.
        """
        return cls(root=extracted / RELEASE_DIR)


class EmipProgrammerGazeConnector(BaseConnector[EmipRelease]):
    """Connector for the EMIP eye movements in programming dataset."""

    async def download_async(self, cache_dir: Path) -> list[EmipRelease]:  # ruff: ignore[no-self-use]
        """Download and extract the release archive once.

        Args:
            cache_dir: The directory that holds the extracted files.

        Returns:
            The extracted release.
        """
        extracted = await ensure_archive(EMIP_URL, cache_dir, filename="emip_dataset.zip", sha256=EMIP_SHA256)
        return [EmipRelease.at(extracted)]

    def convert(self, raw_refs: list[EmipRelease]) -> TimeFDataset:
        """Create one Record and one ClassificationTask for each code-reading trial.

        A trial with no fixation (the tracker lost the eye for the whole trial) has no series to
        store, so it gets no Record.

        Args:
            raw_refs: The release from :meth:`download_async`.

        Returns:
            The populated dataset.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        for release in raw_refs:
            for participant in read_participants(release.root):
                for stimulus, fixations in read_trials(participant.gaze_file):
                    if len(fixations) == 0:
                        continue
                    record = dataset.add_record(record=participant.create_record(stimulus, fixations))
                    dataset.add_task(task=participant.create_classification_task(record))
        return dataset


CONNECTOR = EmipProgrammerGazeConnector
