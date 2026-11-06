from collections.abc import Sequence
from fractions import Fraction

import numpy as np
import pytest

from timenet.errors import TimeNetConnectorError
from timenet.types import ClassificationTask, ureg
from timenet.types.text import read_value
from timenet_connectors.datasets.time_mqa.tsqa.classification import ACTIVITY, DAPHNET, GAIT_FREEZING, WISDM
from timenet_connectors.datasets.time_mqa.tsqa.connector import TimeMqaConnector
from timenet_connectors.datasets.time_mqa.tsqa.release import ReleasePart


_WISDM_QUESTION = (
    "Following data provides accelerometer data for activity recognition research. We provide 10 timestamps of "
    "accelerometer data, with each timestamp containing X, Y, and Z values, for a total of 30 values. "
    "The recorded Time Serie is [{series}]. Please judge whether this data corresponds to 'Walking' or 'Jogging' "
    "or 'Upstairs' or 'Downstairs' or 'Sitting' or 'Standing' given information above."
)
_DAPHNET_QUESTION = (
    "This dataset captures gait freezing using wearable accelerometers placed on the ankle (lower leg), thigh "
    "(above the knee), and hip, with 3D acceleration data recorded at 64 Hz. The input Time Serie is [{series}]. "
    "Please judge whether this data corresponds to No freeze or Freeze. given information above."
)
_WISDM_VALUES = [round(timestep + axis / 3, 2) for timestep in range(10) for axis in range(3)]
"""Thirty values, the X, Y, and Z values of each timestep in turn, so a wrong layout changes the signals."""
_DAPHNET_VALUES = [-161, 1049, 405, 490, 1000, 20, -165, 1076, 87]


def _row(question: str, values: Sequence[float], answer: str) -> dict[str, str]:
    # Synthetic rows in the shape of the release: the window is printed inside the question.
    printed = ", ".join(str(value) for value in values)
    return {
        "application_domain": "Human Activity Recognition",
        "task_type": "classification",
        "QA_list": f'"question": "{question.format(series=printed)}", "answer": "{answer}"',
    }


_ROWS = [
    _row(_WISDM_QUESTION, _WISDM_VALUES, "Based on the given information, the activity is Jogging."),
    _row(_DAPHNET_QUESTION, _DAPHNET_VALUES, "Based on the given information, the answer is No freeze."),
]


def _convert(write_part, rows=_ROWS):
    return TimeMqaConnector().convert([write_part(ReleasePart.CLASSIFICATION, rows)])


def test_a_wisdm_window_becomes_three_axis_signals_at_20_hz(write_part):
    wisdm, _ = _convert(write_part).tasks_of(ClassificationTask)

    (record,) = wisdm.inputs
    signals = {signal.name: signal for signal in record.signals}
    assert list(signals) == list(WISDM.axes)
    for position, axis in enumerate(WISDM.axes):
        # Exact equality: every third printed value, starting at the position of the axis.
        np.testing.assert_array_equal(signals[axis].to_numpy(), np.array(_WISDM_VALUES[position::3], dtype=np.float64))
        assert signals[axis].spec.unit_value == ureg.meter / ureg.second**2
        assert signals[axis].time_axis.period_us == Fraction(1_000_000, 20)
    assert wisdm.label_set == ACTIVITY
    assert wisdm.targets is not None
    assert read_value(wisdm.targets[0]) == "Jogging"
    assert read_value(wisdm.prompt) == _WISDM_QUESTION.format(series="<series 1>").replace("[<series 1>]", "<series 1>")
    assert wisdm.metadata == {
        "file": "classification",
        "row": 1,
        "application_domain": "Human Activity Recognition",
        "task_type": "classification",
    }


def test_a_daphnet_window_becomes_nine_one_step_signals_at_64_hz(write_part):
    _, daphnet = _convert(write_part).tasks_of(ClassificationTask)

    (record,) = daphnet.inputs
    signals = {signal.name: signal for signal in record.signals}
    assert {axis: signals[axis].to_numpy().tolist() for axis in DAPHNET.axes} == {
        axis: [value] for axis, value in zip(DAPHNET.axes, _DAPHNET_VALUES, strict=True)
    }
    assert {signal.spec.unit_value for signal in record.signals} == {ureg.Unit("millistandard_gravity")}
    assert {signal.time_axis.period_us for signal in record.signals} == {Fraction(1_000_000, 64)}
    assert daphnet.label_set == GAIT_FREEZING
    assert daphnet.targets is not None
    assert read_value(daphnet.targets[0]) == "No freeze"
    assert record.metadata == {"file": "classification", "row": 2, "role": "series_1"}


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            _row(_WISDM_QUESTION, _WISDM_VALUES[:-1], "Based on the given information, the activity is Jogging."),
            "a wisdm question prints 30 values, not 29",
        ),
        (
            _row(_DAPHNET_QUESTION, _DAPHNET_VALUES, "Based on the given information, the answer is Jogging."),
            "label 'Jogging' is not one of",
        ),
        (
            _row("Unknown data. The input Time Serie is [{series}].", _DAPHNET_VALUES, "x"),
            "the question starts like none of the sources",
        ),
    ],
    ids=["one value too few", "a label of the other source", "an unknown source"],
)
def test_a_malformed_row_raises_with_its_location(write_part, row, message):
    with pytest.raises(TimeNetConnectorError, match=rf"classification\.csv data row 3: {message}"):
        _convert(write_part, [*_ROWS, row])
