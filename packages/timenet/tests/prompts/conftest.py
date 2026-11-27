from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import RegularAxis
from timenet.types import (
    Annotation,
    ClassificationTask,
    Label,
    LabelSet,
    Split,
    TimeInterval,
    TimeSeriesSpec,
    ureg,
)


WALMSLEY = {
    "light": "Waking behavior below 3 METs that is not sedentary behavior.",
    "moderate-vigorous": "All behavior at 3 METs or more.",
    "sedentary": "Waking behavior at 1.5 METs or less in a sitting, lying or reclining posture.",
    "sleep": "Non-waking behavior.",
}
ACCELERATION = TimeSeriesSpec(spec_type="acceleration", name="Wrist acceleration", unit_value=ureg.standard_gravity)


def label_set(*, multi_label: bool = False, labels: dict[str, str | None] | None = None) -> LabelSet:
    return LabelSet(
        [Label(name=name, description=text) for name, text in (labels or WALMSLEY).items()],
        multi_label=multi_label,
        annotations=[Annotation("name", value="walmsley_2020"), Annotation("description", value="Movement behavior.")],
    )


def acceleration(name: str, *, metadata: dict | None = None) -> Signal:
    return Signal(
        spec=ACCELERATION,
        time_axis=RegularAxis.from_rate_hz(100),
        name=name,
        data=np.arange(3000, dtype=np.float32) / 1000,
        metadata=metadata,
    )


@pytest.fixture
def make():
    """Factories for label sets and acceleration signals."""
    return SimpleNamespace(label_set=label_set, acceleration=acceleration)


@pytest.fixture
def capture():
    """A CAPTURE-24-like task: 30 s of wrist acceleration, scoped to the window from 10 s to 20 s."""
    x = acceleration("x", metadata={"placement": "dominant wrist", "calibration": "c1"})
    y, z = acceleration("y"), acceleration("z")
    source = Source(
        "Axivity AX3",
        signals=[z, x, y],
        metadata={"device": "Axivity AX3", "placement": "dominant wrist", "serial": "12345"},
    )
    activity = source.annotate(Annotation("activity", value="7030 sleeping;MET 0.95", span=TimeInterval.seconds(0, 30)))
    axis_note = x.annotate(Annotation("axis_note", value="x points along the forearm"))
    record = Record(sources=[source], metadata={"file": "P001.csv.gz"})
    record.annotate(Annotation("subject_ids", value=["P001"]))
    age_band = record.annotate(Annotation("age_band", value="38-52"))
    other = Record(sources=[Source("other", signals=[acceleration("w")])])
    instructions = other.annotate(Annotation("instructions", value="Use only the acceleration."))
    task = ClassificationTask(
        inputs=[record],
        scope=TimeInterval.seconds(10, 20),
        targets="sleep",
        label_set=label_set(),
        input_annotations=[axis_note, age_band, instructions],
        split=Split.TRAIN,
        metadata={"window": 3},
    )
    return SimpleNamespace(
        task=task,
        record=record,
        source=source,
        x=x,
        y=y,
        z=z,
        activity=activity,
        axis_note=axis_note,
        age_band=age_band,
        instructions=instructions,
        replace=lambda **changes: replace(task, **changes),
        label_set=label_set,
    )
