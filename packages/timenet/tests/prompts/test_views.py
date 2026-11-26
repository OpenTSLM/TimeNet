import dataclasses
from fractions import Fraction
import json

import numpy as np
import pytest

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import IrregularAxis, RegularAxis
from timenet.errors import TimeFValidationError
from timenet.prompts import views
from timenet.prompts.views import ContextItem, LabelView, WindowView, build_input_view, build_target_view
from timenet.testing import make_dataset
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    StepInterval,
    TimeInterval,
    TimePoint,
    TimeSeriesSpec,
)


def test_input_view_of_a_scoped_task(capture):
    values = build_input_view(capture.task, dataset=make_dataset().metadata).values

    assert values.dataset.name == "Hello World"
    assert values.window == WindowView(duration_s=10.0)
    [record] = values.records
    [source] = record.sources
    assert source.name == "Axivity AX3"
    assert source.metadata == {"device": "Axivity AX3", "placement": "dominant wrist"}
    assert [channel.name for channel in source.channels] == ["x", "y", "z"]
    assert record.channels == source.channels
    x = source.channels[0]
    assert (x.unit, x.unit_symbol, x.sampling_rate_hz, x.n_values) == ("standard_gravity", "g", 100, 1000)
    assert (x.spec_name, x.spec_type) == ("Wrist acceleration", "acceleration")
    assert x.context == (ContextItem(key="axis_note", text="x points along the forearm", is_list=False),)
    assert [item.line for item in record.context] == ["age_band: 38-52"]
    assert [item.line for item in values.context] == ["instructions: Use only the acceleration."]
    assert values.prompt is None
    assert values.label_set.name == "walmsley_2020"
    assert values.label_set.description == "Movement behavior."
    assert values.label_set.options_line == "light, moderate-vigorous, sedentary, sleep"
    assert values.label_set.labels[0] == LabelView(
        name="light", description="Waking behavior below 3 METs that is not sedentary behavior."
    )


def test_input_view_holds_no_answer_and_no_ids(capture):
    text = json.dumps(dataclasses.asdict(build_input_view(capture.task).values))

    for hidden in ("sleeping", "P001", "12345", "csv", "train", "calibration", "targets", "rationale"):
        assert hidden not in text


def test_input_view_holds_only_plain_values(capture):
    def walk(value: object) -> None:
        if dataclasses.is_dataclass(value):
            assert type(value).__module__ == views.__name__
            for field in dataclasses.fields(value):
                walk(getattr(value, field.name))
        elif isinstance(value, tuple | list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                assert isinstance(key, str)
                walk(item)
        else:
            assert value is None or isinstance(value, str | int | float | bool)

    walk(build_input_view(capture.task).values)


def test_window_is_a_stripped_copy(capture):
    view = build_input_view(capture.task)
    window = view.windows[id(view.values.records[0].channels[0])]

    assert window.n_values == 1000
    assert window.annotations == ()
    assert window.metadata == {"placement": "dominant wrist"}
    assert window.time_axis == RegularAxis(period_us=Fraction(10_000))
    np.testing.assert_array_equal(window.to_numpy(), capture.x.to_numpy()[1000:2000])


def test_irregular_window_starts_at_zero(make):
    heart_rate = TimeSeriesSpec(spec_type="heart_rate", name="Heart rate", unit_value=None)
    signal = Signal.from_irregular(
        [60.0, 61.0, 62.0, 63.0],
        time_offsets_us=[5_000_000, 5_500_000, 7_000_000, 12_000_000],
        spec=heart_rate,
        name="hr",
    )
    record = Record(sources=[Source("watch", signals=[signal])])
    task = ClassificationTask(
        inputs=[record], scope=TimeInterval.seconds(5, 10), targets="sleep", label_set=make.label_set()
    )

    view = build_input_view(task)
    [channel] = view.values.records[0].channels

    assert channel.sampling_rate_hz is None
    assert channel.unit is None
    np.testing.assert_array_equal(view.windows[id(channel)].time_offsets_us(), [0, 500_000, 2_000_000])
    np.testing.assert_array_equal(view.windows[id(channel)].to_numpy(), [60.0, 61.0, 62.0])


def test_scope_on_one_signal(capture):
    task = capture.replace(scope=StepInterval(time_series=capture.y, start=0, stop=100))

    values = build_input_view(task).values
    [record] = values.records

    assert [channel.name for channel in record.channels] == ["y"]
    assert record.channels[0].n_values == 100
    assert values.window.duration_s is None
    assert [item.key for item in record.context] == ["axis_note", "age_band"]


def test_two_inputs_and_a_nested_source(make):
    child = Source("child", signals=[make.acceleration("a")])
    note = child.annotate(Annotation("note", value="on the ankle"))
    first = Record(sources=[Source("parent", sources=[child], signals=[make.acceleration("b")])])
    second = Record(sources=[Source("solo", signals=[make.acceleration("c")])])
    task = ClassificationTask(
        inputs=[second, first], targets="sleep", label_set=make.label_set(), input_annotations=[note]
    )

    records = build_input_view(task).values.records

    assert [[source.name for source in record.sources] for record in records] == [["solo"], ["parent", "child"]]
    assert [channel.name for channel in records[1].channels] == ["b", "a"]
    assert [item.line for item in records[1].context] == ["note: on the ankle"]


def test_text_list_annotation(capture):
    steps = capture.record.annotate(Annotation("steps", value=["sit down", "stand up"]))
    task = capture.replace(input_annotations=[steps])

    [item] = build_input_view(task).values.records[0].context

    assert item == ContextItem(key="steps", text="1. sit down\n2. stand up", is_list=True)
    assert item.line == "steps:\n1. sit down\n2. stand up"


def _point_scope(c):
    return c.replace(scope=TimePoint.seconds(1))


def _repeated_input(c):
    return c.replace(inputs=[c.record, c.record])


def _input_is_target(c):
    return c.replace(targets=None, target_annotations=[c.age_band])


def _scoped_annotation(c):
    spike = c.x.annotate(Annotation("spike", value="big", span=TimeInterval.seconds(11, 12)))
    return c.replace(input_annotations=[spike])


def _blank_annotation(c):
    return c.replace(input_annotations=[c.record.annotate(Annotation("note", value="  "))])


def _number_annotation(c):
    return c.replace(input_annotations=[c.record.annotate(Annotation("age", value=64))])


def _broken_list(c):
    return c.replace(input_annotations=[c.record.annotate(Annotation("steps", value=["a", "b\nc"]))])


def _blank_prompt(c):
    return c.replace(prompt="  ")


def _scope_outside(c):
    return c.replace(scope=TimeInterval.seconds(100, 110))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_point_scope, "point"),
        (_repeated_input, "more than once"),
        (_input_is_target, "input and target"),
        (_scoped_annotation, "span"),
        (_blank_annotation, "blank"),
        (_number_annotation, "text"),
        (_broken_list, "line break"),
        (_blank_prompt, "prompt"),
        (_scope_outside, "no input signal"),
    ],
)
def test_input_view_rejects(capture, change, message):
    with pytest.raises(TimeFValidationError, match=message):
        build_input_view(change(capture))


def test_other_task_types_are_rejected(capture):
    task = AnswerTask(inputs=[capture.record], prompt="What?", targets="sleep")

    with pytest.raises(TimeFValidationError, match="ClassificationTask"):
        build_input_view(task)


def _target_view(task: ClassificationTask) -> views.ClassificationTargetView:
    return build_target_view(task, build_input_view(task).values)


def test_target_view(capture):
    task = capture.replace(rationale="Still for 10 s.")

    target = _target_view(task)

    assert (target.targets, target.rationale) == (("sleep",), "Still for 10 s.")
    input_view = build_input_view(task).values
    for field in dataclasses.fields(input_view):
        assert getattr(target, field.name) == getattr(input_view, field.name)


def test_target_view_from_target_annotations(capture):
    first = capture.record.annotate(Annotation("label", value="sleep"))
    second = capture.record.annotate(Annotation("label", value="light"))
    task = capture.replace(
        targets=None, target_annotations=[first, second], label_set=capture.label_set(multi_label=True)
    )

    assert _target_view(task).targets == ("sleep", "light")


def test_empty_multi_label_target(capture):
    target = _target_view(capture.replace(targets=(), label_set=capture.label_set(multi_label=True)))

    assert (target.targets, target.rationale) == ((), None)


@pytest.mark.parametrize(
    ("unit", "symbol"),
    [(None, ""), ("dimensionless", ""), ("standard_gravity", "g"), ("millivolt", "mV"), ("degree_Celsius", "°C")],
)
def test_unit_symbol(unit, symbol):
    channel = views.ChannelView(
        name="a", spec_name="A", spec_type="a", unit=unit, sampling_rate_hz=None, n_values=1, context=()
    )

    assert channel.unit_symbol == symbol


def test_options_line_rejects_a_comma():
    labels = views.LabelSetView(
        name=None, description=None, multi_label=False, labels=(LabelView(name="a, b", description=None),)
    )

    with pytest.raises(TimeFValidationError, match="several options"):
        _ = labels.options_line


def _both_targets(c):
    return c.replace(target_annotations=[c.record.annotate(Annotation("label", value="sleep"))])


def _no_target(c):
    return c.replace(targets=None)


def _label_with_line_break(c):
    return c.replace(targets="a\nb", label_set=c.label_set(labels={"a\nb": None}))


def _repeated_label(c):
    return c.replace(targets=("light", "light"), label_set=c.label_set(multi_label=True))


def _blank_rationale(c):
    return c.replace(rationale="  ")


def _list_target(c):
    return c.replace(targets=None, target_annotations=[c.record.annotate(Annotation("labels", value=["sleep"]))])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_both_targets, "both"),
        (_no_target, "needs"),
        (_label_with_line_break, "line break"),
        (_repeated_label, "repeat"),
        (_blank_rationale, "rationale"),
        (_list_target, "text list"),
    ],
)
def test_target_view_rejects(capture, change, message):
    with pytest.raises(TimeFValidationError, match=message):
        _target_view(change(capture))


def test_channels_share_the_window_start(make):
    heart_rate = TimeSeriesSpec(spec_type="heart_rate", name="Heart rate", unit_value=None)
    a = Signal.from_irregular([1.0, 2.0], time_offsets_us=[7_000_000, 9_000_000], spec=heart_rate, name="a")
    b = Signal.from_irregular(
        [1.0, 2.0, 3.0], time_offsets_us=[5_500_000, 6_000_000, 9_500_000], spec=heart_rate, name="b"
    )
    record = Record(sources=[Source("watch", signals=[a, b])])
    task = ClassificationTask(
        inputs=[record], scope=TimeInterval.seconds(5, 10), targets="sleep", label_set=make.label_set()
    )

    view = build_input_view(task)
    first, second = (view.windows[id(channel)] for channel in view.values.records[0].channels)

    np.testing.assert_array_equal(first.time_offsets_us(), [2_000_000, 4_000_000])
    np.testing.assert_array_equal(second.time_offsets_us(), [500_000, 1_000_000, 4_500_000])
    assert first.time_axis == IrregularAxis(first_us=2_000_000, last_us=4_000_000)


def test_unscoped_segments_keep_their_distance(make):
    early = make.acceleration("early")
    late = Signal(
        spec=early.spec,
        time_axis=RegularAxis.from_rate_hz(100, offset_us=100_000_000),
        name="late",
        data=np.zeros(10, dtype=np.float32),
    )
    record = Record(sources=[Source("device", signals=[early, late])])
    task = ClassificationTask(inputs=[record], targets="sleep", label_set=make.label_set())

    view = build_input_view(task)
    windows = [view.windows[id(channel)] for channel in view.values.records[0].channels]

    axes = [window.time_axis for window in windows]
    assert axes == [
        RegularAxis(period_us=Fraction(10_000)),
        RegularAxis(period_us=Fraction(10_000), offset_us=100_000_000),
    ]
