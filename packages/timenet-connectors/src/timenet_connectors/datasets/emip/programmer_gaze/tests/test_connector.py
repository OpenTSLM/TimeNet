import csv
from pathlib import Path

import pytest

from timenet.errors import TimeNetConnectorError
from timenet.types import ClassificationTask
from timenet.types.text import read_value
from timenet_connectors.datasets.emip.programmer_gaze.connector import (
    EXPERTISE,
    MIN_FIXATION_US,
    SAMPLE_PERIOD_US,
    EmipProgrammerGazeConnector,
    EmipRelease,
    detect_fixations,
)


_HEADER = [
    "Time", "Type", "Trial", "L Raw X [px]", "L Raw Y [px]", "R Raw X [px]", "R Raw Y [px]",
    "L Dia X [px]", "L Dia Y [px]", "L Mapped Diameter [mm]", "R Dia X [px]", "R Dia Y [px]",
    "R Mapped Diameter [mm]", "L CR1 X [px]", "L CR1 Y [px]", "L CR2 X [px]", "L CR2 Y [px]",
    "R CR1 X [px]", "R CR1 Y [px]", "R CR2 X [px]", "R CR2 Y [px]", "L POR X [px]", "L POR Y [px]",
    "R POR X [px]", "R POR Y [px]", "Timing", "L Validity", "R Validity", "Pupil Confidence",
]  # fmt: skip
_START_US = 1_000_000_000
_SAMPLES_PER_FIXATION = 30
"""120 ms at 250 Hz, above the 80 ms minimum."""


def _sample(time_us: int, x: float, y: float, pupil: float = 3.2, validity: str = "1") -> str:
    cells = ["0"] * len(_HEADER)
    cells[0], cells[1], cells[2] = str(time_us), "SMP", "1"
    cells[12], cells[23], cells[24], cells[27] = f"{pupil}", f"{x}", f"{y}", validity
    return "\t".join(cells)


def _message(time_us: int, text: str) -> str:
    return f"{time_us}\tMSG\t1\t# Message: {text}"


class _Trace:
    """Writes the lines of a raw gaze file with a running clock."""

    def __init__(self) -> None:
        self.now = _START_US
        self.lines = ["## [iView]", "## Sample Rate:\t250", "\t".join(_HEADER)]

    def message(self, text: str) -> None:
        self.lines.append(_message(self.now, text))

    def fixation(self, x: float, y: float, *, samples: int = _SAMPLES_PER_FIXATION, pupil: float = 3.2) -> None:
        for index in range(samples):
            # Jitter of 2 px keeps the run inside the 25 px dispersion.
            self.lines.append(_sample(self.now, x + index % 2 * 2, y, pupil))
            self.now += SAMPLE_PERIOD_US

    def blink(self, samples: int = 10) -> None:
        for _ in range(samples):
            self.lines.append(_sample(self.now, 0.0, 0.0, 0.0))
            self.now += SAMPLE_PERIOD_US


def _participant_file(directory: Path, participant_id: int, *, language: str = "java") -> None:
    trace = _Trace()
    trace.message("instruction_comprehension.jpg")
    trace.fixation(960, 540)
    trace.message(f"rectangle_{language}.jpg")
    trace.fixation(500, 300)
    trace.blink()
    trace.fixation(700, 300, pupil=3.6)
    trace.message("UE-mouseclick left x=479 y=370")
    trace.fixation(900, 320)
    trace.message("mupliple_choice_rectangle.jpg")
    trace.fixation(400, 400)
    trace.message(f"vehicle_{language}.jpg")
    trace.fixation(600, 600)
    trace.message("mupliple_choice_vehicle.jpg")
    path = directory / "rawdata" / f"{participant_id}_rawdata.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(trace.lines) + "\n", encoding="latin-1")


_ANSWERS = {
    1: {"expertise_programming": "high", "expertise_experiment_language": "medium", "experiment_language": "Java"},
    2: {"expertise_programming": "none", "expertise_experiment_language": "none", "experiment_language": "Java"},
}


def _write_metadata(directory: Path, answers: dict[int, dict[str, str]]) -> None:
    fields = [
        "id", "expertise_programming", "expertise_experiment_language", "experiment_language",
        "correct_rectangle", "order_rectangle", "correct_vehicle", "order_vehicle",
    ]  # fmt: skip
    with (directory / "emip_metadata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for participant_id, row in answers.items():
            writer.writerow(
                {
                    "id": participant_id,
                    **row,
                    "correct_rectangle": "1",
                    "order_rectangle": "2",
                    "correct_vehicle": "0",
                    "order_vehicle": "1",
                }
            )


def _release(tmp_path: Path, answers: dict[int, dict[str, str]] = _ANSWERS) -> EmipRelease:
    release = EmipRelease.at(tmp_path)
    release.root.mkdir()
    _write_metadata(release.root, answers)
    for participant_id in answers:
        _participant_file(release.root, participant_id)
    return release


def test_each_code_trial_becomes_a_record_and_a_classification_task(tmp_path):
    dataset = EmipProgrammerGazeConnector().convert([_release(tmp_path)])
    tasks = dataset.tasks_of(ClassificationTask)
    assert len(tasks) == 4
    stimuli = [(task.inputs[0].metadata["participant_id"], task.inputs[0].metadata["stimulus"]) for task in tasks]
    assert stimuli == [(1, "rectangle_java"), (1, "vehicle_java"), (2, "rectangle_java"), (2, "vehicle_java")]
    assert all(task.targets is not None for task in tasks)
    assert [read_value(task.targets[0]) for task in tasks if task.targets is not None] == [
        "high",
        "high",
        "none",
        "none",
    ]
    for task in tasks:
        assert task.label_set == EXPERTISE
        (record,) = task.inputs
        assert sorted(signal.name for signal in record.signals) == ["duration", "pupil", "x", "y"]


def test_a_trial_runs_to_the_next_image_and_a_blink_splits_fixations(tmp_path):
    dataset = EmipProgrammerGazeConnector().convert([_release(tmp_path)])
    record = dataset.tasks_of(ClassificationTask)[0].inputs[0]
    values = {signal.name: signal.to_numpy() for signal in record.signals}
    x, y, duration, pupil = (values[name] for name in ("x", "y", "duration", "pupil"))
    # The blink splits two fixations, and the mouse click does not end the trial.
    assert x == pytest.approx([501.0, 701.0, 901.0], abs=0.1)
    assert y == pytest.approx([300.0, 300.0, 320.0])
    assert duration == pytest.approx([_SAMPLES_PER_FIXATION * SAMPLE_PERIOD_US / 1000] * 3)
    assert pupil == pytest.approx([3.2, 3.6, 3.2])


def test_detect_fixations_drops_runs_shorter_than_the_minimum():
    short = MIN_FIXATION_US // SAMPLE_PERIOD_US - 1
    samples = [(index * SAMPLE_PERIOD_US, 100.0, 100.0, 3.0, True) for index in range(short)]
    samples += [(index * SAMPLE_PERIOD_US, 900.0, 900.0, 3.0, True) for index in range(short, short + 25)]
    fixations = detect_fixations(samples, onset_us=0)
    assert len(fixations) == 1
    assert fixations.onset_us.tolist() == [short * SAMPLE_PERIOD_US]


def test_an_unknown_expertise_label_fails(tmp_path):
    answers = {1: {**_ANSWERS[1], "expertise_programming": "guru"}}
    with pytest.raises(TimeNetConnectorError, match="expertise_programming 'guru'"):
        EmipProgrammerGazeConnector().convert([_release(tmp_path, answers)])


def test_a_gaze_file_without_a_questionnaire_row_fails(tmp_path):
    release = _release(tmp_path)
    _participant_file(release.root, 3)
    with pytest.raises(TimeNetConnectorError, match=r"differ: \[3\]"):
        EmipProgrammerGazeConnector().convert([release])
